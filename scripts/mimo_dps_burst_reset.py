import argparse, os, sys, glob
import torch
import numpy as np
import random
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

# ==========================================
#  Helper Classes & Functions
# ==========================================

def get_adaptive_h_lr(current_snr, snr_min=-5, snr_max=25, lr_max=20.0, lr_min=1.0):
    """
    SNRに応じて学習率を線形補間する関数（メインループ用）
    """
    if current_snr <= snr_min:
        return lr_max
    if current_snr >= snr_max:
        return lr_min
    
    slope = (lr_min - lr_max) / (snr_max - snr_min)
    lr = lr_max + (current_snr - snr_min) * slope
    return lr

def plot_channel_evolution(H_true, H_init, H_final, save_path, batch_idx=0):
    """
    初期値(LS)と最終値(GCR)の点のみをプロット (指定バッチ)
    """
    # 指定バッチのデータを取り出し
    h_gt = H_true[batch_idx].detach().cpu().numpy().flatten()
    h_ls = H_init[batch_idx].detach().cpu().numpy().flatten()
    h_gcr = H_final[batch_idx].detach().cpu().numpy().flatten()

    plt.figure(figsize=(8, 8))
    
    # 凡例用のダミー
    plt.scatter([], [], c='red', marker='x', s=100, linewidths=2, label='Ground Truth')
    plt.scatter([], [], c='blue', marker='^', s=80, label='Initial LS')
    plt.scatter([], [], c='none', edgecolors='green', marker='o', s=120, linewidths=2, label='Final Burst+GCR')

    num_elements = len(h_gt)
    for i in range(num_elements):
        plt.scatter(h_gt[i].real, h_gt[i].imag, c='red', marker='x', s=100, linewidths=2)
        plt.text(h_gt[i].real, h_gt[i].imag, f" {i}", fontsize=12, color='red', fontweight='bold', ha='left', va='bottom')

        plt.scatter(h_ls[i].real, h_ls[i].imag, c='blue', marker='^', s=80)
        plt.text(h_ls[i].real, h_ls[i].imag, f" {i}", fontsize=10, color='blue', ha='right', va='top')

        plt.scatter(h_gcr[i].real, h_gcr[i].imag, c='none', edgecolors='green', marker='o', s=120, linewidths=2)
        plt.text(h_gcr[i].real, h_gcr[i].imag, f" {i}", fontsize=10, color='green', ha='left', va='top')

        plt.plot([h_ls[i].real, h_gcr[i].real], [h_ls[i].imag, h_gcr[i].imag], 
                 color='gray', linestyle=':', alpha=0.5)

    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.title(f"Channel Estimation Evolution (Batch[{batch_idx}])\nMethod: Burst Calibration")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved channel plot to {save_path}")

def plot_channel_trajectory(H_history, H_true, H_init, save_path, split_index=None, batch_idx=0):
    """
    Hの推移を軌跡としてプロットする関数 (指定バッチのみ)
    split_index: Burst Phase (Orange) と Main Phase (Green) の境界
    """
    steps = len(H_history)
    
    # Batch idx を取り出し、CPU numpyへ
    # H_historyは [Tensor(B, r, t), ...] のリスト
    traj = torch.stack(H_history).cpu().numpy()[:, batch_idx, :, :].reshape(steps, -1)
    
    h_gt = H_true[batch_idx].detach().cpu().numpy().flatten()
    h_ls = H_init[batch_idx].detach().cpu().numpy().flatten()
    
    plt.figure(figsize=(10, 10))
    
    num_elements = traj.shape[1]
    
    for i in range(num_elements):
        # 軌跡のプロット
        if split_index is not None and split_index < steps:
            # Burst Phase: Orange
            plt.plot(traj[:split_index+1, i].real, traj[:split_index+1, i].imag, 
                     color='orange', linewidth=2.0, alpha=0.8, label='Burst Phase' if i==0 else "")
            # Main Phase: Green
            plt.plot(traj[split_index:, i].real, traj[split_index:, i].imag, 
                     color='green', linewidth=2.0, alpha=0.8, label='Main Phase' if i==0 else "")
            
            # Phase切り替え地点
            plt.scatter(traj[split_index, i].real, traj[split_index, i].imag, 
                        c='orange', marker='s', s=40, zorder=3)
        else:
            plt.plot(traj[:, i].real, traj[:, i].imag, color='gray', linewidth=1, alpha=0.5)

        # 1. Initial LS (Start) - Blue
        plt.scatter(h_ls[i].real, h_ls[i].imag, c='blue', marker='^', s=60, zorder=4, label='Initial LS' if i==0 else "")
        plt.text(h_ls[i].real, h_ls[i].imag, f"{i}", fontsize=10, color='blue', ha='right', va='bottom', fontweight='bold')
        
        # 2. Final Est (End) - Green
        plt.scatter(traj[-1, i].real, traj[-1, i].imag, c='green', marker='o', s=80, zorder=4, label='Final Est' if i==0 else "")
        plt.text(traj[-1, i].real, traj[-1, i].imag, f"{i}", fontsize=10, color='green', ha='left', va='top', fontweight='bold')
        
        # 3. Ground Truth - Red
        plt.scatter(h_gt[i].real, h_gt[i].imag, c='red', marker='x', s=100, linewidths=2, zorder=5, label='Ground Truth' if i==0 else "")
        plt.text(h_gt[i].real, h_gt[i].imag, f"{i}", fontsize=12, color='red', fontweight='bold', ha='left', va='bottom')

    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f"Channel Estimation Trajectory (Batch[{batch_idx}])\nOrange: Burst Calibration, Green: Main GCR Loop")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved trajectory plot to {save_path}")

def plot_h_loss_evolution(burst_loss, main_loss, save_path):
    """
    Burst PhaseとMain PhaseのHのSquared Error (|H_true - H|^2) の推移
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- 1. Burst Phase ---
    ax1.plot(burst_loss, color='orange', linewidth=1.5)
    ax1.set_title("Phase 1: Burst Calibration Loss")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(r"$||H_{true} - \hat{H}||^2$")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 最後の値を表示
    if len(burst_loss) > 0:
        ax1.text(len(burst_loss)*0.7, burst_loss[0]*0.9, f"Start: {burst_loss[0]:.4f}", color='black')
        ax1.text(len(burst_loss)*0.7, burst_loss[-1]*1.1, f"End: {burst_loss[-1]:.4f}", color='red')

    # --- 2. Main Phase ---
    ax2.plot(main_loss, color='green', linewidth=1.5)
    ax2.set_title("Phase 3: Main GCR Sampling Loss")
    ax2.set_xlabel("Sampling Step (Process Order)")
    ax2.set_ylabel(r"$||H_{true} - \hat{H}||^2$")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    if len(main_loss) > 0:
        ax2.text(len(main_loss)*0.05, main_loss[0], f"Start: {main_loss[0]:.4f}", color='black', verticalalignment='bottom')
        ax2.text(len(main_loss)*0.7, main_loss[-1], f"End: {main_loss[-1]:.4f}", color='red', verticalalignment='top')

    plt.suptitle("Evolution of Channel Estimation Error (Squared Norm)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved loss evolution plot to {save_path}")

# ==========================================
#  [NEW] Metric Calculation & Plotting
# ==========================================

def calculate_metrics_single(target_img_01, pred_img, lpips_fn):
    """
    単一画像(1, C, H, W)のPSNRとLPIPSを計算する
    target_img_01: [1, 3, H, W] in [0, 1] (Original)
    pred_img: [1, 3, H, W] in [-1, 1] or [0, 1] (Decoder output)
    """
    # pred_imgを[0, 1]に正規化 (PSNR用)
    if pred_img.min() < 0:
        pred_01 = (pred_img + 1.0) / 2.0
    else:
        pred_01 = pred_img
    pred_01 = torch.clamp(pred_01, 0.0, 1.0)
    
    # PSNR
    mse = torch.mean((target_img_01 - pred_01) ** 2)
    psnr = 20 * torch.log10(1.0 / (torch.sqrt(mse) + 1e-8))
    
    # LPIPS用: 入力は[-1, 1]である必要がある
    target_m11 = target_img_01 * 2.0 - 1.0
    
    if pred_img.min() >= 0:
        pred_m11 = pred_img * 2.0 - 1.0
    else:
        pred_m11 = pred_img
    pred_m11 = torch.clamp(pred_m11, -1.0, 1.0)
    
    with torch.no_grad():
        lpips_val = lpips_fn(target_m11, pred_m11).item()
        
    return psnr.item(), lpips_val

def plot_metrics_evolution(psnr_list, lpips_list, save_path, snr, batch_idx=0):
    """
    PSNRとLPIPSの推移をプロット
    凡例を上部にまとめて表示
    """
    steps = range(len(psnr_list))
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- 左軸: PSNR (Blue) ---
    color1 = 'tab:blue'
    ax1.set_xlabel('Sampling Step (Process Order)')
    ax1.set_ylabel('PSNR (dB)', color=color1)
    
    line1 = ax1.plot(steps, psnr_list, color=color1, label='PSNR (Left Axis)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- 右軸: LPIPS (Red) ---
    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('LPIPS', color=color2) 
    
    line2 = ax2.plot(steps, lpips_list, color=color2, linestyle='--', label='LPIPS (Right Axis)')
    ax2.tick_params(axis='y', labelcolor=color2)

    # --- 凡例 ---
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

    plt.title(f"Evolution of Image Quality - SNR {snr}dB (Batch[{batch_idx}])", y=1.1)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved metrics plot to {save_path}")

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
        print(f"Warning: No images found in {dir_path}")
        return torch.empty(0)

    tensors_list = []
    for path in tqdm(image_paths, desc="Loading Images"):
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
    
    if img.min() < 0:
        img = (img + 1.0) / 2.0
    img = torch.clamp(img, 0.0, 1.0)
        
    for i in range(img.shape[0]):
        vutil.save_image(img[i], os.path.join(dirname, f"{basename}_{i}{ext}"))
    print(f"Saved images to {dirname}/")

def remove_png(path):
    for file in glob.glob(f'{path}/*.png'):
        try: os.remove(file)
        except: pass

# ==========================================
#  Mappers (Latent <-> MIMO Streams)
# ==========================================
def latent_to_mimo_streams(z_real, t_antennas):
    B, C, H, W = z_real.shape
    z_flat = z_real.view(B, -1)
    
    total_elements = z_flat.shape[1]
    L_complex = total_elements // (t_antennas * 2)
    cutoff = L_complex * t_antennas * 2
    z_used = z_flat[:, :cutoff]
    
    z_view = z_used.view(B, t_antennas, -1)
    real_part, imag_part = torch.chunk(z_view, 2, dim=2)
    s = torch.complex(real_part, imag_part)
    
    return s, (B, C, H, W)

def mimo_streams_to_latent(s, original_shape):
    real_part = s.real
    imag_part = s.imag
    z_view = torch.cat([real_part, imag_part], dim=2) # (B, t, 2L)
    z_flat = z_view.view(s.shape[0], -1)
    
    target_size = np.prod(original_shape[1:])
    current_size = z_flat.shape[1]
    
    if current_size < target_size:
        padding = torch.zeros(s.shape[0], target_size - current_size, device=s.device)
        z_flat = torch.cat([z_flat, padding], dim=1)
    
    return z_flat.view(original_shape)

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
    
    base_experiment_name = f"MIMO_Burst_Reset/t={t_mimo}_r={r_mimo}"
    
    parser.add_argument("--input_path", type=str, default="input_img")
    parser.add_argument("--outdir", type=str, default=f"outputs/{base_experiment_name}")
    parser.add_argument("--nosample_outdir", type=str, default=f"outputs/{base_experiment_name}/nosample")
    parser.add_argument("--sentimgdir", type=str, default="./sentimg")
    
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--dps_scale", type=float, default=0.3)
    
    # Burst & Reset Parameters
    parser.add_argument("--burst_iterations", type=int, default=20)
    parser.add_argument("--burst_lr", type=float, default=0.1)
    parser.add_argument("--anchor_lambda", type=float, default=1.0)
    
    # Adaptive Learning Rate
    parser.add_argument("--h_lr_max", type=float, default=20.0)
    parser.add_argument("--h_lr_min", type=float, default=0.05)
    
    parser.add_argument("--seed", type=int, default=42)
    
    # [NEW] 監視するバッチインデックス
    parser.add_argument("--monitor_idx", type=int, default=2, help="Index of the batch to monitor intermediate results and plots")
    
    opt = parser.parse_args()

    seed_everything(opt.seed)

    # Directory Setup
    suffix = "perfect" if Perfect_Estimate else "estimated"
    base_out_path = f"outputs/{base_experiment_name}"
    
    opt.outdir = os.path.join(opt.outdir, suffix)
    opt.nosample_outdir = os.path.join(opt.nosample_outdir, suffix)
    channel_outdir = os.path.join(base_out_path, "channel_plots", suffix)
    
    # 中間生成物用ディレクトリ
    intermediates_base_dir = os.path.join(base_out_path, f"{suffix}_process")

    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(opt.sentimgdir, exist_ok=True)
    os.makedirs(opt.nosample_outdir, exist_ok=True)
    os.makedirs(channel_outdir, exist_ok=True)
    os.makedirs(intermediates_base_dir, exist_ok=True)
    
    remove_png(opt.outdir)
    remove_png(channel_outdir)

    # Load Model
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    # Initialize LPIPS
    print("Loading LPIPS model...")
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    # Load Images
    img = load_images_as_tensors(opt.input_path).to(device)
    batch_size = img.shape[0]
    save_img_individually(img, opt.sentimgdir + "/original.png")
    
    # [NEW] 指定された monitor_idx がバッチサイズを超えていないか確認
    if opt.monitor_idx >= batch_size:
        print(f"Warning: monitor_idx {opt.monitor_idx} is out of bounds for batch size {batch_size}. Resetting to 0.")
        opt.monitor_idx = 0

    # 評価用のターゲット画像 (Ground Truth) を抽出
    # Shape: [1, 3, H, W]
    gt_img_target = img[opt.monitor_idx:opt.monitor_idx+1]

    # Encode & Normalize
    z = model.encode_first_stage(img)
    z = model.get_first_stage_encoding(z).detach()
    
    z_mean = z.mean(dim=(1, 2, 3), keepdim=True)
    z_var = torch.var(z, dim=(1, 2, 3)).view(-1, 1, 1, 1)
    eps = 1e-7
    z_norm = (z - z_mean) / (torch.sqrt(z_var) + eps)
    
    # [NEW] デコード用にターゲット画像の統計量を抽出
    z_mean_target = z_mean[opt.monitor_idx:opt.monitor_idx+1]
    z_var_target = z_var[opt.monitor_idx:opt.monitor_idx+1]

    # Map to MIMO Streams
    s_0_real = z_norm / np.sqrt(2.0)
    s_0, latent_shape = latent_to_mimo_streams(s_0_real, t_mimo)
    s_0 = s_0.to(device)
    
    L_len = s_0.shape[2]
    print(f"MIMO Streams: {t_mimo}x{L_len} complex symbols")

    # Pilot Signal Setup
    t_vec = torch.arange(t_mimo, device=device)
    N_vec = torch.arange(N_pilot, device=device)
    tt, NN = torch.meshgrid(t_vec, N_vec, indexing='ij')
    P = torch.sqrt(torch.tensor(P_power/(N_pilot*t_mimo))) * torch.exp(1j*2*torch.pi*tt*NN/N_pilot)
    P = P.to(device) 

    # Simulation Loop
    min_snr_sim = -5
    max_snr_sim = 25
    
    for snr in range(min_snr_sim, max_snr_sim + 1, 3): 
        print(f"\n======== SNR = {snr} dB (Monitoring Batch [{opt.monitor_idx}]) ========")
        
        noise_variance = t_mimo / (10**(snr/10))
        sigma_n = np.sqrt(noise_variance / 2.0)

        # Channel & Transmission
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
        Y = torch.matmul(H, s_0) + W
        
        # MMSE Initialization
        eff_noise = sigma_e2 + noise_variance
        H_hat_H = H_hat.mH
        Gram = torch.matmul(H_hat_H, H_hat) 
        Reg = eff_noise * torch.eye(t_mimo, device=device).unsqueeze(0)
        inv_mat = torch.inverse(Gram + Reg)
        W_mmse = torch.matmul(inv_mat, H_hat_H) 
        s_mmse = torch.matmul(W_mmse, Y) 
        
        z_init_real = mimo_streams_to_latent(s_mmse, latent_shape)
        z_init_mmse = z_init_real * np.sqrt(2.0)
        
        # Save MMSE Result
        z_nosample = z_init_mmse * (torch.sqrt(z_var) + eps) + z_mean
        rec_nosample = model.decode_first_stage(z_nosample)
        save_img_individually(rec_nosample, f"{opt.nosample_outdir}/mmse_snr{snr}.png")
        
        # Sampling Prep
        W_W_H = torch.matmul(W_mmse, W_mmse.mH) 
        noise_power_factor = W_W_H.diagonal(dim1=-2, dim2=-1).real.mean()
        post_mmse_noise_var_raw = eff_noise * noise_power_factor
        actual_std = z_init_mmse.std(dim=(1, 2, 3), keepdim=True)
        actual_var_flat = (actual_std.flatten()) ** 2
        effective_noise_variance = (post_mmse_noise_var_raw / actual_var_flat).mean()

        eff_var_scalar = noise_variance + sigma_e2
        Sigma_inv = 1.0 / eff_var_scalar
        
        def forward_mapper(z):
            return latent_to_mimo_streams(z / np.sqrt(2.0), t_mimo)
        
        def backward_mapper(s, shape):
            z = mimo_streams_to_latent(s, shape)
            return z * np.sqrt(2.0)

        z_init_normalized = z_init_mmse / (actual_std + 1e-8)
        cond = model.get_learned_conditioning(batch_size * [""])

        current_zeta = opt.dps_scale
        if snr < 5:
            current_zeta *= 0.1
        
        adaptive_h_lr = get_adaptive_h_lr(
            snr, snr_min=min_snr_sim, snr_max=max_snr_sim,
            lr_max=opt.h_lr_max, lr_min=opt.h_lr_min
        )

        print(f"Starting Burst-Reset Sampling... Steps={opt.ddim_steps}")
        print(f"  > Effective Noise Var: {effective_noise_variance.item():.5f}")

        # --- CALL SAMPLER (with monitor_batch_idx) ---
        samples, H_final_est, H_history, burst_loss, main_loss, img_history = sampler.gcr_burst_sampling(
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
            mapper=forward_mapper,
            inv_mapper=backward_mapper,
            initial_noise_variance=effective_noise_variance,
            H_true=H,  
            eta=0.0,
            verbose=True,
            monitor_batch_idx=opt.monitor_idx # [NEW] 監視対象のインデックスを渡す
        )
        
        # 1. 軌跡のプロット (指定バッチ)
        traj_plot_path = os.path.join(channel_outdir, f"trajectory_snr{snr}.png")
        plot_channel_trajectory(H_history, H, H_hat, traj_plot_path, 
                                split_index=opt.burst_iterations, batch_idx=opt.monitor_idx)

        # 2. 始点・終点のプロット (指定バッチ)
        plot_path = os.path.join(channel_outdir, f"channel_plot_snr{snr}.png")
        plot_channel_evolution(H, H_hat, H_final_est, plot_path, batch_idx=opt.monitor_idx)

        # 3. Loss Evolutionのプロット (Batch平均のままが一般的だが、H_lossなどは加算されている可能性があるためそのまま)
        loss_plot_path = os.path.join(channel_outdir, f"loss_evolution_snr{snr}.png")
        plot_h_loss_evolution(burst_loss, main_loss, loss_plot_path)

        # 4. 画像の保存 (全バッチの最終結果)
        z_restored = samples * (torch.sqrt(z_var) + eps) + z_mean
        rec_proposed = model.decode_first_stage(z_restored)
        save_img_individually(rec_proposed, f"{opt.outdir}/burst_reset_snr{snr}.png")

        # -------------------------------------------------------------
        # 5. Analyze Intermediate Images (Monitored Batch Only)
        # -------------------------------------------------------------
        print(f"Analyzing intermediate steps for SNR {snr} (Batch {opt.monitor_idx})...")
        
        # 兄弟ディレクトリのSNRサブフォルダに保存
        inter_dir = os.path.join(intermediates_base_dir, f"snr{snr}")
        os.makedirs(inter_dir, exist_ok=True)
        
        psnr_history = []
        lpips_history = []
        
        # img_history内の要素は [C, H, W] (monitor_idxのものだけ)
        for idx, z_step in enumerate(tqdm(img_history, desc="Decoding Intermediates")):
            # GPUへ移動し、バッチ次元[1, C, H, W]を追加
            z_step_gpu = z_step.to(device).unsqueeze(0)
            
            # Latent復元: ターゲット画像(monitor_idx)の統計量を使用
            z_step_restored = z_step_gpu * (torch.sqrt(z_var_target) + eps) + z_mean_target
            
            with torch.no_grad():
                rec_step = model.decode_first_stage(z_step_restored)
            
            # Metrics Calculation (ターゲット画像と比較)
            p, l = calculate_metrics_single(gt_img_target, rec_step, lpips_fn)
            psnr_history.append(p)
            lpips_history.append(l)
            
            # 保存
            save_img_individually(rec_step, os.path.join(inter_dir, f"step_{idx:03d}.png"))
        
        # Plot Metrics
        metrics_plot_path = os.path.join(channel_outdir, f"metrics_evolution_snr{snr}.png")
        plot_metrics_evolution(psnr_history, lpips_history, metrics_plot_path, snr, batch_idx=opt.monitor_idx)

        print(f"Saved result for SNR {snr}")