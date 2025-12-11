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
# DDIMSamplerは修正したddim.pyからインポートされる前提
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

        plt.plot([h_ls[i].real, h_gcr[i].real], [h_ls[i].imag, h_gcr[i].imag], 
                 color='gray', linestyle=':', alpha=0.5)

    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    title_idx = batch_idx if isinstance(batch_idx, str) else f"Batch[{batch_idx}]"
    plt.title(f"Channel Estimation Evolution ({title_idx})\nMethod: Burst Calibration")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    # print(f"Saved channel plot to {save_path}")

def plot_channel_trajectory(H_history, H_true, H_init, save_path, split_index=None, local_batch_idx=0, global_batch_label=0):
    steps = len(H_history)
    
    # H_history: list of tensors (Steps, B, ...). Pick specific batch index.
    traj = torch.stack(H_history).cpu().numpy()[:, local_batch_idx, :, :].reshape(steps, -1)
    
    h_gt = H_true[local_batch_idx].detach().cpu().numpy().flatten()
    h_ls = H_init[local_batch_idx].detach().cpu().numpy().flatten()
    
    plt.figure(figsize=(10, 10))
    
    num_elements = traj.shape[1]
    
    for i in range(num_elements):
        if split_index is not None and split_index < steps:
            plt.plot(traj[:split_index+1, i].real, traj[:split_index+1, i].imag, 
                     color='orange', linewidth=2.0, alpha=0.8, label='Burst Phase' if i==0 else "")
            plt.plot(traj[split_index:, i].real, traj[split_index:, i].imag, 
                     color='green', linewidth=2.0, alpha=0.8, label='Main Phase' if i==0 else "")
            plt.scatter(traj[split_index, i].real, traj[split_index, i].imag, 
                        c='orange', marker='s', s=40, zorder=3)
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
    title_idx = global_batch_label if isinstance(global_batch_label, str) else f"Batch[{global_batch_label}]"
    plt.title(f"Channel Estimation Trajectory ({title_idx})\nOrange: Burst Calibration, Green: Main GCR Loop")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

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
    
    batch_label = batch_idx if isinstance(batch_idx, str) else f"Batch[{batch_idx}]"
    plt.title(f"Evolution of Image Quality - SNR {snr}dB ({batch_label})", y=1.1)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# --- [Modified] Batch Loading Helpers ---
def get_image_paths(dir_path):
    image_paths = []
    supported_formats = ["*.jpg", "*.jpeg", "*.png"]
    for fmt in supported_formats:
        image_paths.extend(glob.glob(os.path.join(dir_path, fmt)))
    # 自然順ソート
    image_paths.sort(key=lambda f: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', os.path.basename(f))])
    return image_paths

def load_images_from_paths(paths, image_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    tensors_list = []
    for path in tqdm(paths, desc="Loading Batch", leave=False):
        try:
            img = Image.open(path).convert("RGB")
            tensors_list.append(transform(img))
        except Exception as e:
            print(f"Error loading {path}: {e}")
    if not tensors_list:
        return torch.empty(0)
    return torch.stack(tensors_list, dim=0)
# ----------------------------------------

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

def save_img_individually(img, path, start_index=0):
    """
    start_index: このバッチの開始インデックス (通し番号用)
    """
    if len(img.shape) == 3: img = img.unsqueeze(0)
    
    dirname = os.path.dirname(path)
    basename = os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1]
    os.makedirs(dirname, exist_ok=True)
    
    img = torch.clamp(img, 0.0, 1.0)
    
    for i in range(img.shape[0]):
        global_idx = start_index + i
        vutil.save_image(img[i], os.path.join(dirname, f"{basename}_{global_idx}{ext}"))

def remove_png(path):
    if not os.path.exists(path): return
    for file in glob.glob(f'{path}/*.png'):
        try: os.remove(file)
        except: pass

# ==========================================
#  Mappers
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
    z_view = torch.cat([real_part, imag_part], dim=2) 
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
    
    # Batch Size Config
    BATCH_SIZE = 20
    
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
    parser.add_argument("--burst_lr", type=float, default=0.05)
    parser.add_argument("--anchor_lambda", type=float, default=1.0)
    
    # Adaptive Learning Rate
    parser.add_argument("--h_lr_max", type=float, default=20.0)
    parser.add_argument("--h_lr_min", type=float, default=0.05)
    
    parser.add_argument("--seed", type=int, default=42)
    
    # 監視するバッチインデックスの範囲指定 (0~19など)
    parser.add_argument("--monitor_range", type=int, nargs=2, default=[0, 5], 
                        help="Start and End index of GLOBAL indices to monitor")
    
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

    # Load Model (Once)
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    print("Loading LPIPS model...")
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    # ------------------------------------------------------------------
    # Determine Source & Files
    # ------------------------------------------------------------------
    existing_imgs = glob.glob(os.path.join(opt.sentimgdir, "*.png")) + \
                    glob.glob(os.path.join(opt.sentimgdir, "*.jpg"))

    source_dir = ""
    is_new_data = False
    
    if len(existing_imgs) > 0:
        print(f"Found existing images in {opt.sentimgdir}. Loading from there...")
        source_dir = opt.sentimgdir
    else:
        print(f"No existing images in {opt.sentimgdir}. Loading from {opt.input_path}...")
        source_dir = opt.input_path
        is_new_data = True

    all_image_paths = get_image_paths(source_dir)
    total_images = len(all_image_paths)
    
    if total_images == 0:
        raise ValueError("No images found! Please check input paths.")

    print(f"Total images: {total_images}. Processing in batches of {BATCH_SIZE}...")
    
    # 監視対象のインデックス (Global)
    global_monitor_start, global_monitor_end = opt.monitor_range
    print(f"Global Monitor Range: {global_monitor_start} to {global_monitor_end - 1}")

    # ==========================================
    #  Batch Loop
    # ==========================================
    for batch_start_idx in range(0, total_images, BATCH_SIZE):
        batch_end_idx = min(batch_start_idx + BATCH_SIZE, total_images)
        current_batch_paths = all_image_paths[batch_start_idx : batch_end_idx]
        
        print(f"\nProcessing Batch: {batch_start_idx} to {batch_end_idx - 1} ({len(current_batch_paths)} images)")
        
        # Load Images
        img = load_images_from_paths(current_batch_paths).to(device)
        batch_size = img.shape[0]
        
        # 新規データの場合は保存
        if is_new_data:
            save_img_individually(img, opt.sentimgdir + "/original.png", start_index=batch_start_idx)
            
        # --- 監視対象かどうかの判定 ---
        # このバッチに含まれる Global Index リスト
        current_global_indices = list(range(batch_start_idx, batch_end_idx))
        
        # このバッチ内で監視すべきローカルインデックスと、そのGlobalインデックス
        local_monitor_indices = []
        global_monitor_indices_in_batch = []
        
        for local_i, global_i in enumerate(current_global_indices):
            if global_monitor_start <= global_i < global_monitor_end:
                local_monitor_indices.append(local_i)
                global_monitor_indices_in_batch.append(global_i)
        
        if local_monitor_indices:
            print(f"  -> Monitoring Indices in this batch: Local {local_monitor_indices} (Global {global_monitor_indices_in_batch})")
        else:
            # print("  -> No monitoring for this batch.")
            pass

        # -----------------------------
        # 1. Encode & Normalize
        # -----------------------------
        z = model.encode_first_stage(img)
        z = model.get_first_stage_encoding(z).detach()
        
        z_mean = z.mean(dim=(1, 2, 3), keepdim=True)
        z_var = torch.var(z, dim=(1, 2, 3)).view(-1, 1, 1, 1)
        eps = 1e-7
        z_norm = (z - z_mean) / (torch.sqrt(z_var) + eps)
        
        # 統計量保存 (このバッチ用)
        z_mean_batch = z_mean
        z_var_batch = z_var

        # 2. Map to MIMO Streams
        s_0_real = z_norm / np.sqrt(2.0)
        s_0, latent_shape = latent_to_mimo_streams(s_0_real, t_mimo)
        s_0 = s_0.to(device)
        
        L_len = s_0.shape[2]
        
        # 3. Pilot Signal
        t_vec = torch.arange(t_mimo, device=device)
        N_vec = torch.arange(N_pilot, device=device)
        tt, NN = torch.meshgrid(t_vec, N_vec, indexing='ij')
        P = torch.sqrt(torch.tensor(P_power/(N_pilot*t_mimo))) * torch.exp(1j*2*torch.pi*tt*NN/N_pilot)
        P = P.to(device) 

        # -----------------------------
        # SNR Loop
        # -----------------------------
        min_snr_sim = -5
        max_snr_sim = 25
        
        for snr in range(min_snr_sim, max_snr_sim + 1, 3): 
            # print(f"  SNR = {snr} dB")
            
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
            
            # MMSE
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
            z_nosample = z_init_mmse * (torch.sqrt(z_var_batch) + eps) + z_mean_batch
            rec_nosample = model.decode_first_stage(z_nosample)
            save_img_individually(rec_nosample, f"{opt.nosample_outdir}/mmse_snr{snr}.png", start_index=batch_start_idx)
            
            # Sampling Setup
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
            
            adaptive_h_lr = get_adaptive_h_lr(snr, lr_max=opt.h_lr_max, lr_min=opt.h_lr_min)
            opt_steps = get_optimal_steps(snr)

            # Sampling Execution
            # local_monitor_indices を渡して、必要なログのみ取得する
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
                verbose=(batch_start_idx==0), # 最初のバッチのみVerbose
                phase3_num_steps=opt_steps, 
                monitor_indices=local_monitor_indices 
            )
            
            # Save Final Result
            z_restored = samples * (torch.sqrt(z_var_batch) + eps) + z_mean_batch
            rec_proposed = model.decode_first_stage(z_restored)
            save_img_individually(rec_proposed, f"{opt.outdir}/burst_reset_snr{snr}.png", start_index=batch_start_idx)

            # -------------------------------------------------------------
            # Analysis & Plots (Only if monitor indices exist in this batch)
            # -------------------------------------------------------------
            if local_monitor_indices:
                # リストのインデックス(k) と 対応するローカルIndex(local_idx) と グローバルIndex(global_idx) を回す
                for k, (local_idx, global_idx) in enumerate(zip(local_monitor_indices, global_monitor_indices_in_batch)):
                    
                    batch_plot_dir = os.path.join(channel_outdir, f"batch_{global_idx}")
                    os.makedirs(batch_plot_dir, exist_ok=True)

                    # 1. 軌跡
                    traj_plot_path = os.path.join(batch_plot_dir, f"trajectory_snr{snr}.png")
                    plot_channel_trajectory(H_history, H, H_hat, traj_plot_path, 
                                            split_index=opt.burst_iterations, 
                                            local_batch_idx=local_idx, 
                                            global_batch_label=global_idx)

                    # 2. 始点・終点
                    # H_final_estなどは (Batch, ...) なので local_idx でアクセス
                    plot_path = os.path.join(batch_plot_dir, f"channel_plot_snr{snr}.png")
                    # plot_channel_evolution用に関数を少し修正するか、単にデータ抽出してから渡す
                    # ここでは既存関数に tensor[local_idx:local_idx+1] を渡すなどの工夫が必要だが、
                    # 既存関数が [batch_idx] でアクセスする仕様なので、tensor全体を渡して local_idx を指定すればOK
                    plot_channel_evolution(H, H_hat, H_final_est, plot_path, batch_idx=local_idx) 
                    # ※注意: plot_channel_evolution内部のタイトル表示用に global_idx を渡したいが、
                    # 関数シグネチャを変えないなら、後で手直しが必要。
                    # 今回は関数の batch_idx 引数がそのままタイトルに使われると仮定して文字列を渡したいが、
                    # tensorアクセスにも使われているので、integerである必要がある。
                    # => plot_channel_evolution は内部で H_true[batch_idx] しているので local_idx を渡すのが必須。
                    
                    # 3. Loss Evolution (Batch Sumなので、各バッチフォルダに保存するのは少し変だが、参考として保存)
                    if k == 0:
                        loss_plot_path = os.path.join(batch_plot_dir, f"loss_evolution_snr{snr}_batchchunk.png")
                        plot_h_loss_evolution(burst_loss, main_loss, loss_plot_path)

                # Intermediate Images & Metrics
                num_steps = len(img_history)
                all_batches_psnr_history = []
                all_batches_lpips_history = []

                for k, (local_idx, global_idx) in enumerate(zip(local_monitor_indices, global_monitor_indices_in_batch)):
                    inter_dir = os.path.join(intermediates_base_dir, f"snr{snr}", f"batch_{global_idx}")
                    os.makedirs(inter_dir, exist_ok=True)
                    
                    psnr_history = []
                    lpips_history = []
                    
                    gt_img_target = img[local_idx:local_idx+1]
                    z_mean_target = z_mean_batch[local_idx:local_idx+1]
                    z_var_target = z_var_batch[local_idx:local_idx+1]
                    
                    for step_i in range(num_steps):
                        # img_history[step_i] is (Num_Monitored, C, H, W)
                        # k番目がこの global_idx に対応
                        z_step_single = img_history[step_i][k].to(device).unsqueeze(0)
                        
                        z_step_restored = z_step_single * (torch.sqrt(z_var_target) + eps) + z_mean_target
                        with torch.no_grad():
                            rec_step = model.decode_first_stage(z_step_restored)
                        
                        p, l = calculate_metrics_single(gt_img_target, rec_step, lpips_fn)
                        psnr_history.append(p)
                        lpips_history.append(l)
                        
                        save_img_individually(rec_step, os.path.join(inter_dir, f"step_{step_i:03d}.png"))
                    
                    all_batches_psnr_history.append(psnr_history)
                    all_batches_lpips_history.append(lpips_history)

                    batch_plot_dir = os.path.join(channel_outdir, f"batch_{global_idx}")
                    metrics_plot_path = os.path.join(batch_plot_dir, f"metrics_evolution_snr{snr}.png")
                    plot_metrics_evolution(psnr_history, lpips_history, metrics_plot_path, snr, batch_idx=global_idx)

                # Batch Average Metrics (for this chunk)
                if len(all_batches_psnr_history) > 0:
                    avg_psnr = np.mean(np.array(all_batches_psnr_history), axis=0)
                    avg_lpips = np.mean(np.array(all_batches_lpips_history), axis=0)
                    
                    # チャンクごとの平均として保存
                    avg_plot_path = os.path.join(channel_outdir, f"metrics_evolution_snr{snr}_chunk_{batch_start_idx}_AVG.png")
                    plot_metrics_evolution(avg_psnr, avg_lpips, avg_plot_path, snr, batch_idx=f"Avg_Chunk_{batch_start_idx}")

        # メモリ解放
        torch.cuda.empty_cache()

    print("All batches processed.")