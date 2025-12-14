import argparse, os, sys, glob
import torch
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
from ldm.models.diffusion.ddim import DDIMSampler 
from torchvision import utils as vutil
import lpips
import matplotlib.pyplot as plt
import shutil

# ==========================================
#  Helper Classes & Functions
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
    # バッチ内のインデックスを取り出すため、テンソルがバッチ次元を持っているか確認して処理
    h_gt = H_true.detach().cpu().numpy().flatten()
    h_ls = H_init.detach().cpu().numpy().flatten()
    h_gcr = H_final.detach().cpu().numpy().flatten()
    
    plt.figure(figsize=(8, 8))
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
        plt.plot([h_ls[i].real, h_gcr[i].real], [h_ls[i].imag, h_gcr[i].imag], color='gray', linestyle=':', alpha=0.5)
    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.title(f"Channel Estimation Evolution (ID[{batch_idx}])\nMethod: Burst Calibration")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved channel plot to {save_path}")

def plot_channel_trajectory(H_history, H_true, H_init, save_path, split_index=None, batch_idx=0, local_batch_index=0):
    steps = len(H_history)
    # H_history: list of tensors [Batch, r, t]. Extract specific batch item.
    traj = torch.stack(H_history).cpu().numpy()[:, local_batch_index, :, :].reshape(steps, -1)
    h_gt = H_true.detach().cpu().numpy().flatten()
    h_ls = H_init.detach().cpu().numpy().flatten()
    
    plt.figure(figsize=(10, 10))
    num_elements = traj.shape[1]
    for i in range(num_elements):
        if split_index is not None and split_index < steps:
            plt.plot(traj[:split_index+1, i].real, traj[:split_index+1, i].imag, color='orange', linewidth=2.0, alpha=0.8, label='Burst Phase' if i==0 else "")
            plt.plot(traj[split_index:, i].real, traj[split_index:, i].imag, color='green', linewidth=2.0, alpha=0.8, label='Main Phase' if i==0 else "")
            plt.scatter(traj[split_index, i].real, traj[split_index, i].imag, c='orange', marker='s', s=40, zorder=3)
        else:
            plt.plot(traj[:, i].real, traj[:, i].imag, color='gray', linewidth=1, alpha=0.5)
        plt.scatter(h_ls[i].real, h_ls[i].imag, c='blue', marker='^', s=60, zorder=4, label='Initial LS' if i==0 else "")
        plt.text(h_ls[i].real, h_ls[i].imag, f"{i}", fontsize=10, color='blue', ha='right', va='bottom', fontweight='bold')
        plt.scatter(traj[-1, i].real, traj[-1, i].imag, c='green', marker='o', s=80, zorder=4, label='Final Est' if i==0 else "")
        plt.text(traj[-1, i].real, traj[-1, i].imag, f"{i}", fontsize=10, color='green', ha='left', va='top', fontweight='bold')
        plt.scatter(h_gt[i].real, h_gt[i].imag, c='red', marker='x', s=100, linewidths=2, zorder=5, label='Ground Truth' if i==0 else "")
        plt.text(h_gt[i].real, h_gt[i].imag, f"{i}", fontsize=12, color='red', fontweight='bold', ha='left', va='bottom')
    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f"Channel Estimation Trajectory (ID[{batch_idx}])\nOrange: Burst Calibration, Green: Main GCR Loop")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved trajectory plot to {save_path}")

def plot_h_loss_evolution(burst_loss, main_loss, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(burst_loss, color='orange', linewidth=1.5)
    ax1.set_title("Phase 1: Burst Calibration Loss (Batch Sum)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(r"$||H_{true} - \hat{H}||^2$")
    ax1.grid(True, linestyle='--', alpha=0.6)
    if len(burst_loss) > 0:
        ax1.text(len(burst_loss)*0.7, burst_loss[0]*0.9, f"Start: {burst_loss[0]:.4f}", color='black')
        ax1.text(len(burst_loss)*0.7, burst_loss[-1]*1.1, f"End: {burst_loss[-1]:.4f}", color='red')
    ax2.plot(main_loss, color='green', linewidth=1.5)
    ax2.set_title("Phase 3: Main GCR Sampling Loss (Batch Sum)")
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


def calculate_metrics_single(target_img_01, pred_img, lpips_fn):
    """
    単一画像(1, C, H, W)のPSNRとLPIPSを計算する
    target_img_01: [1, 3, H, W] in [0, 1] (Original)
    pred_img: [1, 3, H, W] in [-1, 1] (Decoder output)
    """
    # 1. まず [-1, 1] に収める（外れ値対策）
    pred_clamped = torch.clamp(pred_img, -1.0, 1.0)
    
    # 2. [-1, 1] -> [0, 1] に変換
    pred_01 = (pred_clamped + 1.0) / 2.0
    
    # 念のため [0, 1] クリップ
    pred_01 = torch.clamp(pred_01, 0.0, 1.0)
    
    # PSNR
    mse = torch.mean((target_img_01 - pred_01) ** 2)
    psnr = 20 * torch.log10(1.0 / (torch.sqrt(mse) + 1e-8))
    
    # LPIPS用: 入力は[-1, 1]である必要がある
    target_m11 = target_img_01 * 2.0 - 1.0
    
    # LPIPSには pred_clamped ([-1, 1]) をそのまま使う
    with torch.no_grad():
        lpips_val = lpips_fn(target_m11, pred_clamped).item()
        
    return psnr.item(), lpips_val

def plot_metrics_evolution(psnr_list, lpips_list, save_path, snr, batch_idx=0):
    steps = range(len(psnr_list))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1 = 'tab:blue'
    ax1.set_xlabel('Sampling Step (Process Order)')
    ax1.set_ylabel('PSNR (dB)', color=color1)
    line1 = ax1.plot(steps, psnr_list, color=color1, label='PSNR (Left Axis)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('LPIPS', color=color2) 
    line2 = ax2.plot(steps, lpips_list, color=color2, linestyle='--', label='LPIPS (Right Axis)')
    ax2.tick_params(axis='y', labelcolor=color2)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    if isinstance(batch_idx, str):
        batch_label = batch_idx
    else:
        batch_label = f"ID[{batch_idx}]"
    plt.title(f"Evolution of Image Quality - SNR {snr}dB ({batch_label})", y=1.1)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved metrics plot to {save_path}")

# ==========================================
#  Latent Change Plotting
# ==========================================
def plot_latent_change(diff_list, save_path, snr, batch_idx=0):
    """
    潜在変数の変化量 |x_t - x_{t-1}| の推移をプロット
    """
    steps = range(1, len(diff_list) + 1)
    plt.figure(figsize=(10, 6))
    
    plt.plot(steps, diff_list, color='purple', marker='.', linestyle='-', linewidth=1.0, label='Latent Change |x_t - x_{t-1}|')
    
    plt.xlabel('Sampling Step (Process Order)')
    plt.ylabel('L2 Norm of Difference')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # タイトル設定
    if isinstance(batch_idx, str):
        batch_label = batch_idx
    else:
        batch_label = f"ID[{batch_idx}]"

    plt.title(f"Latent Update Magnitude - SNR {snr}dB ({batch_label})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved latent change plot to {save_path}")

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

def save_img_individually(img, path, start_idx=0):
    """
    画像を保存する関数。start_idx を指定することで、バッチ処理時でも
    通し番号 (ID) を維持したファイル名で保存する。
    """
    if len(img.shape) == 3: img = img.unsqueeze(0)
    dirname = os.path.dirname(path)
    basename = os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1]
    os.makedirs(dirname, exist_ok=True)
    for i in range(img.shape[0]):
        # ここで通し番号 start_idx + i を使用
        global_idx = start_idx + i
        vutil.save_image(img[i], os.path.join(dirname, f"{basename}_{global_idx}{ext}"))

def remove_png(path):
    for file in glob.glob(f'{path}/*.png'):
        try: os.remove(file)
        except: pass

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
    z_view = torch.cat([real_part, imag_part], dim=2) 
    z_flat = z_view.view(s.shape[0], -1)
    target_size = np.prod(original_shape[1:])
    current_size = z_flat.shape[1]
    if current_size < target_size:
        padding = torch.zeros(s.shape[0], target_size - current_size, device=s.device)
        z_flat = torch.cat([z_flat, padding], dim=1)
    return z_flat.view(original_shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    t_mimo = 2 
    r_mimo = 2 
    N_pilot = 2 
    P_power = 1.0 
    Perfect_Estimate = False 
    
    parser.add_argument("--input_path", type=str, default="input_img")
    # 初期値は None に設定し、後で自動生成ロジックを通す
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--nosample_outdir", type=str, default=None)
    parser.add_argument("--sentimgdir", type=str, default="./sentimg")
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--dps_scale", type=float, default=0.3)
    parser.add_argument("--burst_iterations", type=int, default=20)
    parser.add_argument("--burst_lr", type=float, default=0.05)
    parser.add_argument("--anchor_lambda", type=float, default=0.0)
    parser.add_argument("--h_lr_max", type=float, default=20.0)
    parser.add_argument("--h_lr_min", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--monitor_range", type=int, nargs=2, default=[0, 5], 
                        help="Start and End index of batches to monitor (e.g. 0 10 for batch 0 to 9)")
    
    # 【追加】バッチ処理用の引数
    parser.add_argument("--process_batch_size", type=int, default=20, 
                        help="Number of images to process at once on GPU")
    
    opt = parser.parse_args()

    seed_everything(opt.seed)
    
    # --- 修正点: ここでハイパーパラメータを含むディレクトリ名を動的に作成 ---
    # 例: t=2_r=2_steps=200_burst=20_blr=0.05_lam=0.0_zeta=0.3
    param_str = (f"t={t_mimo}_r={r_mimo}_"
                 f"steps={opt.ddim_steps}_"
                 f"burst={opt.burst_iterations}_"
                 f"blr={opt.burst_lr}_"
                 f"lam={opt.anchor_lambda}_"
                 f"zeta={opt.dps_scale}")

    base_experiment_name = f"MIMO_Burst_Reset/{param_str}"
    
    # 引数でoutdirが指定されていなければ、自動生成したパスを使用
    if opt.outdir is None:
        opt.outdir = f"outputs/{base_experiment_name}"
    
    if opt.nosample_outdir is None:
        opt.nosample_outdir = f"outputs/{base_experiment_name}/nosample"

    # base_out_path もこの新しいパスに合わせる
    base_out_path = opt.outdir 
    
    # -------------------------------------------------------------
    
    suffix = "perfect" if Perfect_Estimate else "estimated"
    
    # 既存フォルダの削除は慎重に行う（バッチ処理で追記する可能性がある場合は注意だが、
    # ここでは実行時に一括で初期化する挙動とする）
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
    
    print(f"Experiment outputs will be saved to: {opt.outdir}")

    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    print("Loading LPIPS model...")
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    # まず画像をCPUメモリ上にロードする
    existing_imgs = glob.glob(os.path.join(opt.sentimgdir, "*.png")) + \
                    glob.glob(os.path.join(opt.sentimgdir, "*.jpg"))

    if len(existing_imgs) > 0:
        print(f"Found existing images in {opt.sentimgdir}. Loading from there to preserve order...")
        all_imgs_cpu = load_images_as_tensors(opt.sentimgdir) # deviceには送らない
    else:
        print(f"No existing images in {opt.sentimgdir}. Loading from {opt.input_path}...")
        all_imgs_cpu = load_images_as_tensors(opt.input_path) # deviceには送らない
        # オリジナルの保存（一括で保存してもCPUメモリなら大丈夫な範囲と仮定、多すぎる場合はここもループ推奨）
        save_img_individually(all_imgs_cpu, opt.sentimgdir + "/original.png", start_idx=0)

    if all_imgs_cpu.shape[0] == 0:
        raise ValueError("No images loaded! Please check input paths.")
    
    total_images = all_imgs_cpu.shape[0]
    print(f"Total images loaded: {total_images}. Processing in batches of {opt.process_batch_size}...")

    # モニタリングしたいグローバルインデックスのリストを作成
    global_monitor_indices = list(range(opt.monitor_range[0], opt.monitor_range[1]))
    
    # 定数定義
    t_vec = torch.arange(t_mimo, device=device)
    N_vec = torch.arange(N_pilot, device=device)
    tt, NN = torch.meshgrid(t_vec, N_vec, indexing='ij')
    P = torch.sqrt(torch.tensor(P_power/(N_pilot*t_mimo))) * torch.exp(1j*2*torch.pi*tt*NN/N_pilot)
    P = P.to(device) 

    # ==========================================
    #  Outer Loop: Process Image Batches
    # ==========================================
    for batch_start_idx in range(0, total_images, opt.process_batch_size):
        batch_end_idx = min(batch_start_idx + opt.process_batch_size, total_images)
        current_batch_size = batch_end_idx - batch_start_idx
        
        print(f"\n################################################################")
        print(f" Processing Global Batch: Images {batch_start_idx} to {batch_end_idx - 1} (Count: {current_batch_size})")
        print(f"################################################################")

        # GPUへ転送
        img_01 = all_imgs_cpu[batch_start_idx:batch_end_idx].to(device)
        
        # モデル入力用に [-1, 1] に変換
        img_m11 = img_01 * 2.0 - 1.0 
        gt_imgs = img_01
        
        # 現在のバッチに含まれるモニタリング対象を探す
        current_batch_monitor_indices = [] # グローバルID
        current_batch_monitor_local_indices = [] # バッチ内相対ID (0 ~ batch_size-1)
        
        for g_idx in global_monitor_indices:
            if batch_start_idx <= g_idx < batch_end_idx:
                current_batch_monitor_indices.append(g_idx)
                current_batch_monitor_local_indices.append(g_idx - batch_start_idx)
        
        print(f"Monitoring Global Indices in this batch: {current_batch_monitor_indices}")

        # エンコード
        z = model.encode_first_stage(img_m11)
        z = model.get_first_stage_encoding(z).detach()
        
        z_mean = z.mean(dim=(1, 2, 3), keepdim=True)
        z_var = torch.var(z, dim=(1, 2, 3)).view(-1, 1, 1, 1)
        eps = 1e-7
        z_norm = (z - z_mean) / (torch.sqrt(z_var) + eps)
        
        z_mean_target_all = z_mean
        z_var_target_all = z_var

        s_0_real = z_norm / np.sqrt(2.0)
        s_0, latent_shape = latent_to_mimo_streams(s_0_real, t_mimo)
        s_0 = s_0.to(device)
        
        L_len = s_0.shape[2]

        min_snr_sim = 0
        max_snr_sim = 15

        # ==========================================
        #  Inner Loop: Iterate SNRs
        # ==========================================
        for snr in range(min_snr_sim, max_snr_sim + 1, 1): 
            print(f"  --- SNR = {snr} dB (Images {batch_start_idx}-{batch_end_idx-1}) ---")
            
            noise_variance = t_mimo / (10**(snr/10))
            sigma_n = np.sqrt(noise_variance / 2.0)

            H_real = torch.randn(current_batch_size, r_mimo, t_mimo, device=device) * np.sqrt(0.5)
            H_imag = torch.randn(current_batch_size, r_mimo, t_mimo, device=device) * np.sqrt(0.5)
            H = torch.complex(H_real, H_imag)

            V_real = torch.randn(current_batch_size, r_mimo, N_pilot, device=device) * np.sqrt(noise_variance/2)
            V_imag = torch.randn(current_batch_size, r_mimo, N_pilot, device=device) * np.sqrt(noise_variance/2)
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

            W_real = torch.randn(current_batch_size, r_mimo, L_len, device=device) * sigma_n
            W_imag = torch.randn(current_batch_size, r_mimo, L_len, device=device) * sigma_n
            W = torch.complex(W_real, W_imag)
            Y = torch.matmul(H, s_0) + W
            
            eff_noise = sigma_e2 + noise_variance
            H_hat_H = H_hat.mH
            Gram = torch.matmul(H_hat_H, H_hat) 
            Reg = eff_noise * torch.eye(t_mimo, device=device).unsqueeze(0)
            inv_mat = torch.inverse(Gram + Reg)
            W_mmse = torch.matmul(inv_mat, H_hat_H) 
            s_mmse = torch.matmul(W_mmse, Y) 
            
            z_init_real = mimo_streams_to_latent(s_mmse, latent_shape)
            z_init_mmse = z_init_real * np.sqrt(2.0)
            
            z_nosample = z_init_mmse * (torch.sqrt(z_var) + eps) + z_mean
            rec_nosample = model.decode_first_stage(z_nosample)
            
            # [-1, 1] -> [0, 1] に変換してから保存
            rec_nosample_01 = torch.clamp((rec_nosample + 1.0) / 2.0, 0.0, 1.0)
            # 【重要】start_idx を渡してグローバルIDで保存
            save_img_individually(rec_nosample_01, f"{opt.nosample_outdir}/mmse_snr{snr}.png", start_idx=batch_start_idx)
            
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
            cond = model.get_learned_conditioning(current_batch_size * [""])

            current_zeta = opt.dps_scale
            if snr < 5:
                current_zeta *= 0.1
            
            adaptive_h_lr = get_adaptive_h_lr(
                snr, snr_min=min_snr_sim, snr_max=max_snr_sim,
                lr_max=opt.h_lr_max, lr_min=opt.h_lr_min
            )

            opt_steps = get_optimal_steps(snr)
            
            # 監視対象がある場合のみ monitor_indices を渡す（ただしローカルインデックスで）
            samples, H_final_est, H_history, burst_loss, main_loss, img_history = sampler.gcr_burst_sampling(
                S=opt.ddim_steps,
                batch_size=current_batch_size,
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
                monitor_indices=current_batch_monitor_local_indices # ローカルインデックスを渡す
            )
            
            # プロット処理 (監視対象が含まれている場合のみ)
            for k, real_global_idx in enumerate(current_batch_monitor_indices):
                local_idx = current_batch_monitor_local_indices[k]
                
                print(f"  -> Processing plots for Global Batch {real_global_idx} (Local {local_idx})")
                
                batch_plot_dir = os.path.join(channel_outdir, f"batch_{real_global_idx}")
                os.makedirs(batch_plot_dir, exist_ok=True)

                traj_plot_path = os.path.join(batch_plot_dir, f"trajectory_snr{snr}.png")
                # H (True), H_hat (Init) は全バッチ分あるので local_idx でアクセス
                plot_channel_trajectory(H_history, H[local_idx:local_idx+1], H_hat[local_idx:local_idx+1], traj_plot_path, 
                                        split_index=opt.burst_iterations, batch_idx=real_global_idx, local_batch_index=local_idx)

                plot_path = os.path.join(batch_plot_dir, f"channel_plot_snr{snr}.png")
                plot_channel_evolution(H[local_idx:local_idx+1], H_hat[local_idx:local_idx+1], H_final_est[local_idx:local_idx+1], plot_path, batch_idx=real_global_idx)

                # Lossのプロットは1回だけ (最初の監視対象のみ代表して出力、もしくはBatch Sumなのでそのまま)
                if k == 0:
                    loss_plot_path_root = os.path.join(channel_outdir, f"loss_evolution_snr{snr}_batch_start_{batch_start_idx}.png")
                    plot_h_loss_evolution(burst_loss, main_loss, loss_plot_path_root)

            z_restored = samples * (torch.sqrt(z_var) + eps) + z_mean
            rec_proposed = model.decode_first_stage(z_restored)
            
            # [-1, 1] -> [0, 1] に変換してから保存
            rec_proposed_01 = torch.clamp((rec_proposed + 1.0) / 2.0, 0.0, 1.0)
            # 【重要】start_idx を渡す
            save_img_individually(rec_proposed_01, f"{opt.outdir}/burst_reset_snr{snr}.png", start_idx=batch_start_idx)

            # 中間生成物の分析
            if len(current_batch_monitor_indices) > 0:
                print(f"Analyzing intermediate steps for SNR {snr} (Indices: {current_batch_monitor_indices})...")
                
                num_steps = len(img_history)
                
                all_batches_psnr_history = []
                all_batches_lpips_history = []
                all_batches_latent_diff_history = []

                for k, real_global_idx in enumerate(current_batch_monitor_indices):
                    local_idx = current_batch_monitor_local_indices[k]
                    
                    inter_dir = os.path.join(intermediates_base_dir, f"snr{snr}", f"batch_{real_global_idx}")
                    os.makedirs(inter_dir, exist_ok=True)
                    
                    psnr_history = []
                    lpips_history = []
                    latent_diff_history = []
                    
                    gt_img_target = gt_imgs[local_idx:local_idx+1]
                    z_mean_target = z_mean_target_all[local_idx:local_idx+1]
                    z_var_target = z_var_target_all[local_idx:local_idx+1]
                    
                    for idx in range(num_steps):
                        z_step_batch = img_history[idx]
                        # img_history は monitor_indices 分しか保存されていないため、インデックスは k (list index) を使う
                        z_step_single = z_step_batch[k]
                        
                        # Latent Difference Calculation
                        if idx > 0:
                            z_prev_single = img_history[idx-1][k]
                            diff_val = torch.norm(z_step_single - z_prev_single).item()
                            latent_diff_history.append(diff_val)

                        z_step_gpu = z_step_single.to(device).unsqueeze(0)
                        z_step_restored = z_step_gpu * (torch.sqrt(z_var_target) + eps) + z_mean_target
                        
                        with torch.no_grad():
                            rec_step = model.decode_first_stage(z_step_restored) # range [-1, 1]
                        
                        # calculate_metrics_single は [-1, 1] を期待
                        p, l = calculate_metrics_single(gt_img_target, rec_step, lpips_fn)
                        psnr_history.append(p)
                        lpips_history.append(l)
                        
                        # 保存用に [-1, 1] -> [0, 1] に変換
                        rec_step_01 = torch.clamp((rec_step + 1.0) / 2.0, 0.0, 1.0)
                        # ステップごとの保存は元々連番ファイルではないのでそのまま
                        save_img_individually(rec_step_01, os.path.join(inter_dir, f"step_{idx:03d}.png"), start_idx=0) # ファイル名自体にidxが入るのでstart_idx=0でOK
                    
                    all_batches_psnr_history.append(psnr_history)
                    all_batches_lpips_history.append(lpips_history)
                    all_batches_latent_diff_history.append(latent_diff_history)

                    batch_plot_dir = os.path.join(channel_outdir, f"batch_{real_global_idx}")
                    metrics_plot_path = os.path.join(batch_plot_dir, f"metrics_evolution_snr{snr}.png")
                    plot_metrics_evolution(psnr_history, lpips_history, metrics_plot_path, snr, batch_idx=real_global_idx)

                    # Plot Latent Change
                    latent_plot_path = os.path.join(batch_plot_dir, f"latent_change_snr{snr}.png")
                    plot_latent_change(latent_diff_history, latent_plot_path, snr, batch_idx=real_global_idx)
            
        print(f"Finished processing batch {batch_start_idx} - {batch_end_idx-1}")
        # GPUメモリのキャッシュクリア
        torch.cuda.empty_cache()

    print("All processing finished.")