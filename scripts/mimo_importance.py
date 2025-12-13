import argparse, os, sys, glob
import torch
import torch.nn as nn  # 追加
import numpy as np
import random
import re
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision import transforms
from ldm.util import instantiate_from_config
# DDIMSamplerは修正したddim.pyからインポートされる前提
from ldm.models.diffusion.ddim import DDIMSampler 
from torchvision import utils as vutil
import lpips
import matplotlib.pyplot as plt
import shutil

# ==========================================
#  [NEW] Importance Prediction Model Classes
#  (Copied from train_importance.py)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class LatentImportancePredictor(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=64, num_blocks=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.tail = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

# ==========================================
#  [NEW] Importance & Power Manager
# ==========================================
class ImportanceManager:
    def __init__(self, model_path, t_antennas, power_ratio=0.9, total_power=1.0, device='cuda'):
        """
        重要度に基づく並べ替えと電力配分を管理するクラス
        """
        self.device = device
        self.t_antennas = t_antennas
        
        # モデル読み込み
        self.imp_model = LatentImportancePredictor(in_channels=4).to(device)
        if os.path.exists(model_path):
            print(f"[ImportanceManager] Loading importance model from {model_path}")
            # strict=Falseにして、学習時の不要なキーが含まれていてもロードできるようにする
            self.imp_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        else:
            print(f"[ImportanceManager] WARNING: Model not found at {model_path}. Using random weights (Simulation ONLY).")
        self.imp_model.eval()

        # 電力配分行列 Phi の対角成分計算 (Eq. 5)
        # ストリーム1に power_ratio (例: 0.9) を割り当て、残りを等分
        p1 = total_power * power_ratio
        if t_antennas > 1:
            p_others = (total_power - p1) / (t_antennas - 1)
            self.powers = [p1] + [p_others] * (t_antennas - 1)
        else:
            self.powers = [total_power]
        
        # 振幅スケーリング係数 (sqrt(P))
        self.phi_diag = torch.sqrt(torch.tensor(self.powers, device=device))
        print(f"[ImportanceManager] Power Allocation (Phi^2): {self.powers}")

        # 並べ替えインデックスの保存用 (受信側で使うサイド情報)
        self.perm_indices = None
        self.inv_perm_indices = None
        self.last_shape = None

    def predict_and_sort(self, z, Block):
        """
        Ground Truth z を入力とし、重要度予測 -> ソート -> Permutation を行う。
        返り値は (B, total_dim) のフラット化＆並べ替え済みテンソル。
        """
        self.last_shape = z.shape
        B, C, H, W = z.shape
        
        with torch.no_grad():
            # 重要度予測 (学習済みモデル使用)
            imp_map = self.imp_model(z) # (B, C, H, W)
        # チャネルで総和 (B, 1, H, W)
        imp_map_channel_sum = torch.sum(imp_map, dim=(1))

        # ブロック化 (B, C, H, W) -> (B, C, H/Nt, W/Nt)
        # サイズ 
        # フラット化
        imp_flat = imp_map.view(B, -1)
        z_flat = z.view(B, -1)
        
        # 重要度順に降順ソート
        # indices: (B, total_dim)
        _, sorted_indices = torch.sort(imp_flat, dim=1, descending=True)
        
        # 並べ替え実行 (Permutation)
        z_permuted = torch.gather(z_flat, 1, sorted_indices)
        
        # 逆置換インデックスの計算 (受信側での復元用サイド情報)
        inv_indices = torch.zeros_like(sorted_indices)
        src = torch.arange(z_flat.shape[1], device=self.device).expand(B, -1)
        inv_indices.scatter_(1, sorted_indices, src)
        
        # 状態保存
        self.perm_indices = sorted_indices
        self.inv_perm_indices = inv_indices
        
        return z_permuted

    def map_to_streams_with_power(self, z_permuted):
        """
        並べ替え済み z (Flat) -> 複素シンボル化 -> 電力割り当て -> x_tx
        """
        B = z_permuted.shape[0]
        total_elements = z_permuted.shape[1]
        
        # 複素数化 (前半を実部、後半を虚部。既存コードのロジックを踏襲)
        # 割り切れるように調整
        L_complex = total_elements // (self.t_antennas * 2)
        cutoff = L_complex * self.t_antennas * 2
        z_used = z_permuted[:, :cutoff]
        
        # 既存コードでは z / sqrt(2) していたので、ここで同様のスケーリングを行う
        z_scaled = z_used / np.sqrt(2.0)
        
        z_view = z_scaled.view(B, self.t_antennas, -1) # (B, Antennas, 2*L_sub)
        real_part, imag_part = torch.chunk(z_view, 2, dim=2)
        s = torch.complex(real_part, imag_part) # (B, Antennas, L_sub)
        
        # 電力行列 Phi を適用 (ブロードキャスト)
        # phi_diag: (Antennas) -> (1, Antennas, 1)
        phi = self.phi_diag.view(1, self.t_antennas, 1)
        
        # x_tx = Phi * s
        x_tx = s * phi
        
        return x_tx

    def forward_process_for_gradient(self, z_in):
        """
        勾配計算(Mapper)用: z -> Permutation(Fixed) -> Power -> x_tx
        """
        B_curr = z_in.shape[0]
        z_flat = z_in.view(B_curr, -1)
        
        # 保存されているインデックスで並べ替え
        indices = self.perm_indices
        if indices.shape[0] != B_curr:
            # バッチサイズ不一致時の保険 (基本的に一致する設計)
            indices = indices[:B_curr]
            
        z_permuted = torch.gather(z_flat, 1, indices)
        
        # 電力配分
        x_tx = self.map_to_streams_with_power(z_permuted)
        return x_tx, z_in.shape

    def inverse_map_and_permute(self, x_tx, shape=None):
        """
        受信信号(推定) x_tx -> 電力除去 -> 複素数分解 -> 逆並べ替え -> z
        """
        if shape is None:
            shape = self.last_shape
        B, C, H, W = shape
        
        # 1. 電力除去 (逆行列: 対角行列なので逆数)
        phi = self.phi_diag.view(1, self.t_antennas, 1)
        # ゼロ除算防止
        s_est = x_tx / (phi + 1e-8)
        
        # 2. 実数化 & フラット化
        real_part = s_est.real
        imag_part = s_est.imag
        z_view = torch.cat([real_part, imag_part], dim=2)
        z_permuted_flat = z_view.view(B, -1)
        
        # sqrt(2)倍して元のスケールに戻す
        z_permuted_flat = z_permuted_flat * np.sqrt(2.0)
        
        # パディング復元 (cutoffで切り捨てた分)
        target_len = C * H * W
        curr_len = z_permuted_flat.shape[1]
        if curr_len < target_len:
            pad = torch.zeros(B, target_len - curr_len, device=self.device)
            z_permuted_flat = torch.cat([z_permuted_flat, pad], dim=1)
            
        # 3. 逆並べ替え (Inverse Permutation)
        z_restored_flat = torch.gather(z_permuted_flat, 1, self.inv_perm_indices)
        
        return z_restored_flat.view(B, C, H, W)


# ==========================================
#  Helper Functions (Visualization etc.)
# ==========================================

def get_adaptive_h_lr(current_snr, snr_min=-5, snr_max=25, lr_max=20.0, lr_min=1.0):
    if current_snr <= snr_min:
        return lr_max
    if current_snr >= snr_max:
        return lr_min
    slope = (lr_min - lr_max) / (snr_max - snr_min)
    lr = lr_max + (current_snr - snr_min) * slope
    return lr

def get_optimal_steps(snr):
    steps = 28.33 * np.exp(-0.0879 * snr) - 1.45
    return int(np.clip(np.round(steps), 1, 200))

def plot_channel_evolution(H_true, H_init, H_final, save_path, batch_idx=0):
    h_gt = H_true[batch_idx].detach().cpu().numpy().flatten()
    h_ls = H_init[batch_idx].detach().cpu().numpy().flatten()
    h_gcr = H_final[batch_idx].detach().cpu().numpy().flatten()

    plt.figure(figsize=(8, 8))
    plt.scatter([], [], c='red', marker='x', s=100, linewidths=2, label='Ground Truth')
    plt.scatter([], [], c='blue', marker='^', s=80, label='Initial LS')
    plt.scatter([], [], c='none', edgecolors='green', marker='o', s=120, linewidths=2, label='Final Burst+GCR')

    num_elements = len(h_gt)
    for i in range(num_elements):
        plt.scatter(h_gt[i].real, h_gt[i].imag, c='red', marker='x', s=100, linewidths=2)
        plt.scatter(h_ls[i].real, h_ls[i].imag, c='blue', marker='^', s=80)
        plt.scatter(h_gcr[i].real, h_gcr[i].imag, c='none', edgecolors='green', marker='o', s=120, linewidths=2)
        plt.plot([h_ls[i].real, h_gcr[i].real], [h_ls[i].imag, h_gcr[i].imag], color='gray', linestyle=':', alpha=0.5)

    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.title(f"Channel Estimation Evolution (Batch[{batch_idx}])")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_channel_trajectory(H_history, H_true, H_init, save_path, split_index=None, batch_idx=0):
    steps = len(H_history)
    traj = torch.stack(H_history).cpu().numpy()[:, batch_idx, :, :].reshape(steps, -1)
    h_gt = H_true[batch_idx].detach().cpu().numpy().flatten()
    h_ls = H_init[batch_idx].detach().cpu().numpy().flatten()
    
    plt.figure(figsize=(10, 10))
    num_elements = traj.shape[1]
    
    for i in range(num_elements):
        if split_index is not None and split_index < steps:
            plt.plot(traj[:split_index+1, i].real, traj[:split_index+1, i].imag, color='orange', linewidth=2.0, alpha=0.8)
            plt.plot(traj[split_index:, i].real, traj[split_index:, i].imag, color='green', linewidth=2.0, alpha=0.8)
        else:
            plt.plot(traj[:, i].real, traj[:, i].imag, color='gray', linewidth=1, alpha=0.5)

        plt.scatter(h_ls[i].real, h_ls[i].imag, c='blue', marker='^', s=60, zorder=4)
        plt.scatter(traj[-1, i].real, traj[-1, i].imag, c='green', marker='o', s=80, zorder=4)
        plt.scatter(h_gt[i].real, h_gt[i].imag, c='red', marker='x', s=100, linewidths=2, zorder=5)

    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f"Channel Trajectory (Batch[{batch_idx}])\nOrange: Burst, Green: Main")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_h_loss_evolution(burst_loss, main_loss, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(burst_loss, color='orange', linewidth=1.5)
    ax1.set_title("Phase 1: Burst Calibration Loss")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Squared Error")
    ax1.grid(True)

    ax2.plot(main_loss, color='green', linewidth=1.5)
    ax2.set_title("Phase 3: Main GCR Sampling Loss")
    ax2.set_xlabel("Step")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def calculate_metrics_single(target_img_01, pred_img, lpips_fn):
    pred_clamped = torch.clamp(pred_img, -1.0, 1.0)
    pred_01 = (pred_clamped + 1.0) / 2.0
    pred_01 = torch.clamp(pred_01, 0.0, 1.0)
    
    mse = torch.mean((target_img_01 - pred_01) ** 2)
    psnr = 20 * torch.log10(1.0 / (torch.sqrt(mse) + 1e-8))
    
    target_m11 = target_img_01 * 2.0 - 1.0
    with torch.no_grad():
        lpips_val = lpips_fn(target_m11, pred_clamped).item()
        
    return psnr.item(), lpips_val

def plot_metrics_evolution(psnr_list, lpips_list, save_path, snr, batch_idx=0):
    steps = range(len(psnr_list))
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('PSNR (dB)', color=color1)
    line1 = ax1.plot(steps, psnr_list, color=color1, label='PSNR')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('LPIPS', color=color2) 
    line2 = ax2.plot(steps, lpips_list, color=color2, linestyle='--', label='LPIPS')
    ax2.tick_params(axis='y', labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
    
    label_str = batch_idx if isinstance(batch_idx, str) else f"Batch[{batch_idx}]"
    plt.title(f"Image Quality Evolution - SNR {snr}dB ({label_str})", y=1.1)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_images_as_tensors(dir_path, image_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    image_paths = []
    supported_formats = ["*.jpg", "*.jpeg", "*.png"]
    for fmt in supported_formats:
        image_paths.extend(glob.glob(os.path.join(dir_path, fmt)))
    
    if not image_paths:
        return torch.empty(0)

    image_paths.sort(key=lambda f: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', os.path.basename(f))])
    
    tensors_list = []
    for path in tqdm(image_paths, desc=f"Loading Images from {dir_path}"):
        try:
            img = Image.open(path).convert("RGB")
            tensors_list.append(transform(img))
        except Exception as e:
            print(f"Error loading {path}: {e}")

    return torch.stack(tensors_list, dim=0)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if verbose:
        if len(m) > 0: print("missing keys:", m)
        if len(u) > 0: print("unexpected keys:", u)
    model.cuda()
    model.eval()
    return model

def save_img_individually(img, path):
    if len(img.shape) == 3: img = img.unsqueeze(0)
    dirname = os.path.dirname(path)
    basename = os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1]
    os.makedirs(dirname, exist_ok=True)
    for i in range(img.shape[0]):
        vutil.save_image(img[i], os.path.join(dirname, f"{basename}_{i}{ext}"))

# ==========================================
#  Main Script
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # MIMO Parameters
    t_mimo = 2 
    r_mimo = 2 
    N_pilot = 2 
    
    P_power = 1.0 
    Perfect_Estimate = False 
    
    base_experiment_name = f"MIMO_Burst_Reset_Importance/t={t_mimo}_r={r_mimo}"
    
    parser.add_argument("--input_path", type=str, default="input_img")
    parser.add_argument("--outdir", type=str, default=f"outputs/{base_experiment_name}")
    parser.add_argument("--nosample_outdir", type=str, default=f"outputs/{base_experiment_name}/nosample")
    parser.add_argument("--sentimgdir", type=str, default="./sentimg")
    
    # [NEW] Importance Model Path (Change default to your trained path)
    parser.add_argument("--imp_model_path", type=str, default="checkpoints/importance_model/student_epoch_10.pth",
                        help="Path to the trained Importance Predictor model (.pth)")
    
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--dps_scale", type=float, default=0.3)
    
    # Burst & Reset Parameters
    parser.add_argument("--burst_iterations", type=int, default=20)
    parser.add_argument("--burst_lr", type=float, default=0.05)
    parser.add_argument("--anchor_lambda", type=float, default=1.0)
    
    # Adaptive Learning Rate
    parser.add_argument("--h_lr_max", type=float, default=20.0)
    parser.add_argument("--h_lr_min", type=float, default=0.05)
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--monitor_range", type=int, nargs=2, default=[0, 5])
    
    opt = parser.parse_args()

    seed_everything(opt.seed)
    
    suffix = "perfect" if Perfect_Estimate else "estimated"
    base_out_path = f"outputs/{base_experiment_name}"
    if os.path.exists(base_out_path):
        print(f"Removing previous experiment results at: {base_out_path}")
        shutil.rmtree(base_out_path)
    opt.outdir = os.path.join(opt.outdir, suffix)
    opt.nosample_outdir = os.path.join(opt.nosample_outdir, suffix)
    channel_outdir = os.path.join(base_out_path, "channel_plots", suffix)
    intermediates_base_dir = os.path.join(base_out_path, f"{suffix}_process")

    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(opt.sentimgdir, exist_ok=True)
    os.makedirs(opt.nosample_outdir, exist_ok=True)
    os.makedirs(channel_outdir, exist_ok=True)
    os.makedirs(intermediates_base_dir, exist_ok=True)

    # Load LDM
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    # Load LPIPS
    print("Loading LPIPS model...")
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    # Load Images
    existing_imgs = glob.glob(os.path.join(opt.sentimgdir, "*.png")) + \
                    glob.glob(os.path.join(opt.sentimgdir, "*.jpg"))

    if len(existing_imgs) > 0:
        print(f"Found existing images in {opt.sentimgdir}. Loading from there to preserve order...")
        img = load_images_as_tensors(opt.sentimgdir).to(device)
    else:
        print(f"No existing images in {opt.sentimgdir}. Loading from {opt.input_path}...")
        img = load_images_as_tensors(opt.input_path).to(device)
        save_img_individually(img, opt.sentimgdir + "/original.png")

    if img.shape[0] == 0:
        raise ValueError("No images loaded! Please check input paths.")
        
    batch_size = img.shape[0]
    
    start_idx, end_idx = opt.monitor_range
    end_idx = min(end_idx, batch_size)
    monitor_indices = list(range(start_idx, end_idx))
    print(f"Monitoring batches: {monitor_indices} (Total {len(monitor_indices)})")

    # =========================================================================
    # [NEW] Initialize Importance Manager
    # =========================================================================
    imp_manager = ImportanceManager(
        model_path=opt.imp_model_path,
        t_antennas=t_mimo,
        power_ratio=0.9, # Ratio for the 1st stream (configurable if needed)
        total_power=P_power,
        device=device
    )

    # Encode & Normalize Latent
    z = model.encode_first_stage(img)
    z = model.get_first_stage_encoding(z).detach()
    
    z_mean = z.mean(dim=(1, 2, 3), keepdim=True)
    z_var = torch.var(z, dim=(1, 2, 3)).view(-1, 1, 1, 1)
    eps = 1e-7
    z_norm = (z - z_mean) / (torch.sqrt(z_var) + eps)
    
    z_mean_target_all = z_mean
    z_var_target_all = z_var

    # =========================================================================
    # [MODIFIED] Transmit Signal Generation (Sorting & Power Allocation)
    # =========================================================================
    # 1. 重要度に基づいて並べ替え (Ground Truth使用)
    #    この時点で imp_manager 内部に perm_indices が保存される
    z_permuted_gt = imp_manager.predict_and_sort(z_norm)
    
    # 2. MIMOストリームへのマッピングと電力配分
    #    x_tx_gt は Phi * s の状態になっている
    x_tx_gt = imp_manager.map_to_streams_with_power(z_permuted_gt)
    x_tx_gt = x_tx_gt.to(device)
    
    L_len = x_tx_gt.shape[2]
    print(f"MIMO Transmission Prepared: {t_mimo}x{L_len} symbols (Weighted & Sorted)")

    # Pilot Signal Setup
    t_vec = torch.arange(t_mimo, device=device)
    N_vec = torch.arange(N_pilot, device=device)
    tt, NN = torch.meshgrid(t_vec, N_vec, indexing='ij')
    P = torch.sqrt(torch.tensor(P_power/(N_pilot*t_mimo))) * torch.exp(1j*2*torch.pi*tt*NN/N_pilot)
    P = P.to(device) 

    # Simulation Loop
    min_snr_sim = -5
    max_snr_sim = 25
    gt_imgs = img 

    for snr in range(min_snr_sim, max_snr_sim + 1, 3): 
        print(f"\n======== SNR = {snr} dB ========")
        print(f"Monitoring: {monitor_indices}")
        
        noise_variance = t_mimo / (10**(snr/10))
        sigma_n = np.sqrt(noise_variance / 2.0)

        # Channel Setup
        H_real = torch.randn(batch_size, r_mimo, t_mimo, device=device) * np.sqrt(0.5)
        H_imag = torch.randn(batch_size, r_mimo, t_mimo, device=device) * np.sqrt(0.5)
        H = torch.complex(H_real, H_imag)

        V_real = torch.randn(batch_size, r_mimo, N_pilot, device=device) * np.sqrt(noise_variance/2)
        V_imag = torch.randn(batch_size, r_mimo, N_pilot, device=device) * np.sqrt(noise_variance/2)
        V = torch.complex(V_real, V_imag)
        S_pilot = torch.matmul(H, P) + V
        
        if Perfect_Estimate:
            H_hat = H 
            sigma_e2 = 0.0
        else:
            P_herm = P.mH
            inv_PP = torch.inverse(torch.matmul(P, P_herm))
            H_hat = torch.matmul(S_pilot, torch.matmul(P_herm, inv_PP))
            sigma_e2 = noise_variance / (P_power/t_mimo)

        W_real = torch.randn(batch_size, r_mimo, L_len, device=device) * sigma_n
        W_imag = torch.randn(batch_size, r_mimo, L_len, device=device) * sigma_n
        W = torch.complex(W_real, W_imag)
        
        # [MODIFIED] Transmission with Power-Allocated Signal
        Y = torch.matmul(H, x_tx_gt) + W
        
        # =====================================================================
        # [MODIFIED] MMSE Initialization & Reset
        # =====================================================================
        # H_hat から x_tx (電力付きシンボル) を推定する
        eff_noise = sigma_e2 + noise_variance
        H_hat_H = H_hat.mH
        Gram = torch.matmul(H_hat_H, H_hat) 
        Reg = eff_noise * torch.eye(t_mimo, device=device).unsqueeze(0)
        inv_mat = torch.inverse(Gram + Reg)
        W_mmse = torch.matmul(inv_mat, H_hat_H) 
        
        # x_tx の推定値
        x_tx_est = torch.matmul(W_mmse, Y) 
        
        # Latentの復元 (電力除去 + 逆並べ替え)
        z_init_real = imp_manager.inverse_map_and_permute(x_tx_est, z_norm.shape)
        z_init_mmse = z_init_real # 既にスケール調整済み
        
        # Save MMSE Result
        z_nosample = z_init_mmse * (torch.sqrt(z_var) + eps) + z_mean
        rec_nosample = model.decode_first_stage(z_nosample)
        save_img_individually(rec_nosample, f"{opt.nosample_outdir}/mmse_snr{snr}.png")
        
        # =====================================================================
        # [MODIFIED] Sampler Preparation
        # =====================================================================
        # MMSE後の残留ノイズ分散推定
        W_W_H = torch.matmul(W_mmse, W_mmse.mH) 
        noise_power_factor = W_W_H.diagonal(dim1=-2, dim2=-1).real.mean()
        post_mmse_noise_var_raw = eff_noise * noise_power_factor
        actual_std = z_init_mmse.std(dim=(1, 2, 3), keepdim=True)
        actual_var_flat = (actual_std.flatten()) ** 2
        effective_noise_variance = (post_mmse_noise_var_raw / actual_var_flat).mean()

        eff_var_scalar = noise_variance + sigma_e2
        Sigma_inv = 1.0 / eff_var_scalar
        
        # --- Define Mappers for Sampler ---
        # 重要: 勾配計算では「電力配分」と「並べ替え」を含んだForwardを行う
        def forward_mapper(z):
            # z (normalized) -> Sort -> Power -> x_tx
            return imp_manager.forward_process_for_gradient(z)
        
        def backward_mapper(x_tx, shape):
            # x_tx -> Inverse Power -> Inverse Sort -> z (normalized)
            return imp_manager.inverse_map_and_permute(x_tx, shape)

        z_init_normalized = z_init_mmse / (actual_std + 1e-8)
        cond = model.get_learned_conditioning(batch_size * [""])

        current_zeta = opt.dps_scale
        if snr < 5:
            current_zeta *= 0.1
        
        adaptive_h_lr = get_adaptive_h_lr(
            snr, snr_min=min_snr_sim, snr_max=max_snr_sim,
            lr_max=opt.h_lr_max, lr_min=opt.h_lr_min
        )

        opt_steps = get_optimal_steps(snr)

        print(f"Starting Burst-Reset Sampling... Steps={opt.ddim_steps}")
        print(f"  > Effective Noise Var: {effective_noise_variance.item():.5f}")
        print(f"  > Phase3 Steps: {opt_steps}")
        
        # --- CALL SAMPLER ---
        samples, H_final_est, H_history, burst_loss, main_loss, img_history = sampler.gcr_burst_sampling_importance(
            S=opt.ddim_steps,
            batch_size=batch_size,
            shape=z.shape[1:4], 
            conditioning=cond,
            y=Y,                 
            H_hat=H_hat, 
            Sigma_inv=torch.tensor(Sigma_inv, device=device),
            z_init=z_init_normalized, 
            burst_iterations=opt.burst_iterations,
            burst_lr=opt.burst_lr,
            anchor_lambda=opt.anchor_lambda,
            zeta=current_zeta,
            h_lr=adaptive_h_lr, 
            mapper=forward_mapper,       # Updated
            inv_mapper=backward_mapper,  # Updated
            initial_noise_variance=effective_noise_variance,
            H_true=H,  
            eta=0.0,
            verbose=True,
            phase3_num_steps=opt_steps,
            monitor_indices=monitor_indices 
        )
        
        # ---------------------------------------------------------------------------------
        # Visualization & Logging (Same as before)
        # ---------------------------------------------------------------------------------
        for k, real_batch_idx in enumerate(monitor_indices):
            print(f"  -> Processing plots for Batch {real_batch_idx}")
            batch_plot_dir = os.path.join(channel_outdir, f"batch_{real_batch_idx}")
            os.makedirs(batch_plot_dir, exist_ok=True)

            traj_plot_path = os.path.join(batch_plot_dir, f"trajectory_snr{snr}.png")
            plot_channel_trajectory(H_history, H, H_hat, traj_plot_path, 
                                    split_index=opt.burst_iterations, batch_idx=real_batch_idx)

            plot_path = os.path.join(batch_plot_dir, f"channel_plot_snr{snr}.png")
            plot_channel_evolution(H, H_hat, H_final_est, plot_path, batch_idx=real_batch_idx)

            if k == 0:
                loss_plot_path_root = os.path.join(channel_outdir, f"loss_evolution_snr{snr}_total.png")
                plot_h_loss_evolution(burst_loss, main_loss, loss_plot_path_root)

        z_restored = samples * (torch.sqrt(z_var) + eps) + z_mean
        rec_proposed = model.decode_first_stage(z_restored)
        save_img_individually(rec_proposed, f"{opt.outdir}/burst_reset_snr{snr}.png")

        # Analyze Intermediate Images
        print(f"Analyzing intermediate steps for SNR {snr}...")
        num_steps = len(img_history)
        all_batches_psnr_history = []
        all_batches_lpips_history = []

        for k, real_batch_idx in enumerate(monitor_indices):
            inter_dir = os.path.join(intermediates_base_dir, f"snr{snr}", f"batch_{real_batch_idx}")
            os.makedirs(inter_dir, exist_ok=True)
            
            psnr_history = []
            lpips_history = []
            gt_img_target = gt_imgs[real_batch_idx:real_batch_idx+1]
            z_mean_target = z_mean_target_all[real_batch_idx:real_batch_idx+1]
            z_var_target = z_var_target_all[real_batch_idx:real_batch_idx+1]
            
            for idx in range(num_steps):
                z_step_batch = img_history[idx]
                z_step_single = z_step_batch[k]
                z_step_gpu = z_step_single.to(device).unsqueeze(0)
                z_step_restored = z_step_gpu * (torch.sqrt(z_var_target) + eps) + z_mean_target
                
                with torch.no_grad():
                    rec_step = model.decode_first_stage(z_step_restored)
                
                p, l = calculate_metrics_single(gt_img_target, rec_step, lpips_fn)
                psnr_history.append(p)
                lpips_history.append(l)
                save_img_individually(rec_step, os.path.join(inter_dir, f"step_{idx:03d}.png"))
            
            all_batches_psnr_history.append(psnr_history)
            all_batches_lpips_history.append(lpips_history)
            
            batch_plot_dir = os.path.join(channel_outdir, f"batch_{real_batch_idx}")
            metrics_plot_path = os.path.join(batch_plot_dir, f"metrics_evolution_snr{snr}.png")
            plot_metrics_evolution(psnr_history, lpips_history, metrics_plot_path, snr, batch_idx=real_batch_idx)
        
        if len(all_batches_psnr_history) > 0:
            avg_psnr_history = np.mean(np.array(all_batches_psnr_history), axis=0)
            avg_lpips_history = np.mean(np.array(all_batches_lpips_history), axis=0)
            avg_plot_path = os.path.join(channel_outdir, f"metrics_evolution_snr{snr}_AVERAGE.png")
            plot_metrics_evolution(avg_psnr_history, avg_lpips_history, avg_plot_path, snr, batch_idx="Average")

        print(f"Saved all results for SNR {snr}")