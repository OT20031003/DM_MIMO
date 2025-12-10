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

def plot_channel_evolution(H_true, H_init, H_final, save_path):
    """
    初期値(LS)と最終値(GCR)の点のみをプロット
    """
    h_gt = H_true[0].detach().cpu().numpy().flatten()
    h_ls = H_init[0].detach().cpu().numpy().flatten()
    h_gcr = H_final[0].detach().cpu().numpy().flatten()

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
    plt.title(f"Channel Estimation Evolution (Batch[0])\nMethod: Burst Calibration")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved channel plot to {save_path}")

def plot_channel_trajectory(H_history, H_true, H_init, save_path, split_index=None):
    """
    Hの推移を軌跡としてプロットする関数 (Batch[0]のみ)
    split_index: Burst Phase (Orange) と Main Phase (Green) の境界
    すべての点にインデックス番号を付与
    """
    # H_history: list of Tensor(B, r, t) -> Convert to (Steps, r*t)
    steps = len(H_history)
    
    # Batch 0 を取り出し、CPU numpyへ
    traj = torch.stack(H_history).cpu().numpy()[:, 0, :, :].reshape(steps, -1)
    
    h_gt = H_true[0].detach().cpu().numpy().flatten()
    h_ls = H_init[0].detach().cpu().numpy().flatten()
    
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
        # 番号を追加 (右・上寄り)
        plt.text(h_ls[i].real, h_ls[i].imag, f"{i}", fontsize=10, color='blue', ha='right', va='bottom', fontweight='bold')
        
        # 2. Final Est (End) - Green
        plt.scatter(traj[-1, i].real, traj[-1, i].imag, c='green', marker='o', s=80, zorder=4, label='Final Est' if i==0 else "")
        # 番号を追加 (左・下寄り)
        plt.text(traj[-1, i].real, traj[-1, i].imag, f"{i}", fontsize=10, color='green', ha='left', va='top', fontweight='bold')
        
        # 3. Ground Truth - Red
        plt.scatter(h_gt[i].real, h_gt[i].imag, c='red', marker='x', s=100, linewidths=2, zorder=5, label='Ground Truth' if i==0 else "")
        # 番号を追加 (左・上寄り)
        plt.text(h_gt[i].real, h_gt[i].imag, f"{i}", fontsize=12, color='red', fontweight='bold', ha='left', va='bottom')

    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title("Channel Estimation Trajectory\nOrange: Burst Calibration, Green: Main GCR Loop")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved trajectory plot to {save_path}")

# ==========================================
#  NEW: Loss Evolution Plotter
# ==========================================
def plot_h_loss_evolution(burst_loss, main_loss, save_path):
    """
    Burst PhaseとMain PhaseのHのSquared Error (|H_true - H|^2) の推移を
    横並びのグラフにして保存する。
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
    # 通常のDiffusionはStepが逆順で進行するが、配列はappend順（処理順）になっているため
    # x軸は「Processing Step」として表示する
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
    # python -m scripts.mimo_dps_burst_reset > output_burst.txt
    # ----------------------------------------------------
    # Experiment Naming
    # ----------------------------------------------------
    base_experiment_name = f"MIMO_Burst_Reset/t={t_mimo}_r={r_mimo}"
    
    parser.add_argument("--input_path", type=str, default="input_img")
    parser.add_argument("--outdir", type=str, default=f"outputs/{base_experiment_name}")
    parser.add_argument("--nosample_outdir", type=str, default=f"outputs/{base_experiment_name}/nosample")
    parser.add_argument("--sentimgdir", type=str, default="./sentimg")
    
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--dps_scale", type=float, default=0.3)
    
    # ----------------------------------------------------
    # Burst & Reset Parameters (New!)
    # ----------------------------------------------------
    parser.add_argument("--burst_iterations", type=int, default=20, help="Number of H updates in Burst Phase")
    parser.add_argument("--burst_lr", type=float, default=0.1, help="Learning rate for H in Burst Phase")
    parser.add_argument("--anchor_lambda", type=float, default=1.0, help="Weight for H-Anchor regularization")
    
    # Adaptive Learning Rate Settings for Main Phase
    parser.add_argument("--h_lr_max", type=float, default=20.0)
    parser.add_argument("--h_lr_min", type=float, default=0.05)
    
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    
    opt = parser.parse_args()

    # Seed Setting
    seed_everything(opt.seed)

    # Directory Setup
    suffix = "perfect" if Perfect_Estimate else "estimated"
    base_out_path = f"outputs/{base_experiment_name}"
    opt.outdir = os.path.join(opt.outdir, suffix)
    opt.nosample_outdir = os.path.join(opt.nosample_outdir, suffix)
    channel_outdir = os.path.join(base_out_path, "channel_plots", suffix)

    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(opt.sentimgdir, exist_ok=True)
    os.makedirs(opt.nosample_outdir, exist_ok=True)
    os.makedirs(channel_outdir, exist_ok=True)
    
    remove_png(opt.outdir)
    remove_png(channel_outdir)

    # Load Model
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    # Load Images
    img = load_images_as_tensors(opt.input_path).to(device)
    batch_size = img.shape[0]
    save_img_individually(img, opt.sentimgdir + "/original.png")

    # Encode & Normalize
    z = model.encode_first_stage(img)
    z = model.get_first_stage_encoding(z).detach()
    
    z_mean = z.mean(dim=(1, 2, 3), keepdim=True)
    z_var = torch.var(z, dim=(1, 2, 3)).view(-1, 1, 1, 1)
    eps = 1e-7
    z_norm = (z - z_mean) / (torch.sqrt(z_var) + eps)
    
    # 1. Map Latent to MIMO Streams
    s_0_real = z_norm / np.sqrt(2.0)
    s_0, latent_shape = latent_to_mimo_streams(s_0_real, t_mimo)
    s_0 = s_0.to(device)
    
    L_len = s_0.shape[2]
    print(f"MIMO Streams: {t_mimo}x{L_len} complex symbols")

    # 2. Pilot Signal Setup
    t_vec = torch.arange(t_mimo, device=device)
    N_vec = torch.arange(N_pilot, device=device)
    tt, NN = torch.meshgrid(t_vec, N_vec, indexing='ij')
    P = torch.sqrt(torch.tensor(P_power/(N_pilot*t_mimo))) * torch.exp(1j*2*torch.pi*tt*NN/N_pilot)
    P = P.to(device) 

    # Simulation Loop
    min_snr_sim = -5
    max_snr_sim = 25
    
    for snr in range(min_snr_sim, max_snr_sim + 1, 3): 
        print(f"\n======== SNR = {snr} dB ========")
        
        noise_variance = t_mimo / (10**(snr/10))
        sigma_n = np.sqrt(noise_variance / 2.0)

        # A. Channel Generation
        H_real = torch.randn(batch_size, r_mimo, t_mimo, device=device) * np.sqrt(0.5)
        H_imag = torch.randn(batch_size, r_mimo, t_mimo, device=device) * np.sqrt(0.5)
        H = torch.complex(H_real, H_imag)

        # B. Pilot Transmission & Estimation
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

        # C. Data Transmission
        W_real = torch.randn(batch_size, r_mimo, L_len, device=device) * sigma_n
        W_imag = torch.randn(batch_size, r_mimo, L_len, device=device) * sigma_n
        W = torch.complex(W_real, W_imag)
        
        Y = torch.matmul(H, s_0) + W
        
        # D. MMSE Initialization (Original)
        eff_noise = sigma_e2 + noise_variance
        
        H_hat_H = H_hat.mH
        Gram = torch.matmul(H_hat_H, H_hat) 
        Reg = eff_noise * torch.eye(t_mimo, device=device).unsqueeze(0)
        
        inv_mat = torch.inverse(Gram + Reg)
        W_mmse = torch.matmul(inv_mat, H_hat_H) 
        
        s_mmse = torch.matmul(W_mmse, Y) 
        
        # Save MMSE Result
        z_init_real = mimo_streams_to_latent(s_mmse, latent_shape)
        z_init_mmse = z_init_real * np.sqrt(2.0)
        
        z_nosample = z_init_mmse * (torch.sqrt(z_var) + eps) + z_mean
        rec_nosample = model.decode_first_stage(z_nosample)
        save_img_individually(rec_nosample, f"{opt.nosample_outdir}/mmse_snr{snr}.png")
        
        # E. Prepare for Burst-GCR Sampling
        
        # -------------------------------------------------------------
        # [Corrected] Noise Variance Normalization (Consistency Fix)
        # -------------------------------------------------------------
        W_W_H = torch.matmul(W_mmse, W_mmse.mH) 
        noise_power_factor = W_W_H.diagonal(dim1=-2, dim2=-1).real.mean()
        
        # MMSE後の物理的な残留ノイズ分散
        post_mmse_noise_var_raw = eff_noise * noise_power_factor
        
        # z_initの信号分散で正規化して、Samplerが期待する「相対ノイズ分散」に変換
        actual_std = z_init_mmse.std(dim=(1, 2, 3), keepdim=True)
        actual_var_flat = (actual_std.flatten()) ** 2
        
        # これがbench_MMSE.pyと同じスケールのノイズ分散になる
        effective_noise_variance = (post_mmse_noise_var_raw / actual_var_flat).mean()

        # Regularization for solver
        eff_var_scalar = noise_variance + sigma_e2
        Sigma_inv = 1.0 / eff_var_scalar
        
        def forward_mapper(z):
            return latent_to_mimo_streams(z / np.sqrt(2.0), t_mimo)
        
        def backward_mapper(s, shape):
            z = mimo_streams_to_latent(s, shape)
            return z * np.sqrt(2.0)

        # Input Normalization
        z_init_normalized = z_init_mmse / (actual_std + 1e-8)
        
        cond = model.get_learned_conditioning(batch_size * [""])

        # Adaptive Settings based on SNR
        current_zeta = opt.dps_scale
        if snr < 5:
            current_zeta *= 0.1
            print(f"[Info] Low SNR ({snr}dB): Reducing Zeta to {current_zeta:.4f}")
        
        adaptive_h_lr = get_adaptive_h_lr(
            snr, 
            snr_min=min_snr_sim, snr_max=max_snr_sim,
            lr_max=opt.h_lr_max, lr_min=opt.h_lr_min
        )

        print(f"Starting Burst-Reset Sampling... Steps={opt.ddim_steps}, Burst={opt.burst_iterations}, H_LR={adaptive_h_lr:.2f}")
        print(f"  > Effective Noise Var: {effective_noise_variance.item():.5f} (Matched with Benchmark)")

        # --- NEW SAMPLING CALL (Updated to unpack loss histories) ---
        samples, H_final_est, H_history, burst_loss, main_loss = sampler.gcr_burst_sampling(
            S=opt.ddim_steps,
            batch_size=batch_size,
            shape=z.shape[1:4], 
            conditioning=cond,
            
            y=Y,                 
            H_hat=H_hat, 
            Sigma_inv=torch.tensor(Sigma_inv, device=device),
            z_init=z_init_normalized, 
            
            # Burst Parameters
            burst_iterations=opt.burst_iterations,
            burst_lr=opt.burst_lr,
            anchor_lambda=opt.anchor_lambda,
            
            # Main Loop Parameters
            zeta=current_zeta,
            h_lr=adaptive_h_lr, 
            
            mapper=forward_mapper,
            inv_mapper=backward_mapper,
            
            # ここで修正済みの正規化された分散を渡す
            initial_noise_variance=effective_noise_variance,
            
            # Evaluation
            H_true=H,  
            
            eta=0.0,
            verbose=True 
        )
        
        # 1. 軌跡のプロット (BurstとMainの境界をsplit_indexで指定)
        traj_plot_path = os.path.join(channel_outdir, f"trajectory_snr{snr}.png")
        plot_channel_trajectory(H_history, H, H_hat, traj_plot_path, split_index=opt.burst_iterations)

        # 2. 始点・終点のプロット
        plot_path = os.path.join(channel_outdir, f"channel_plot_snr{snr}.png")
        plot_channel_evolution(H, H_hat, H_final_est, plot_path)

        # 3. NEW: Loss Evolutionのプロット
        loss_plot_path = os.path.join(channel_outdir, f"loss_evolution_snr{snr}.png")
        plot_h_loss_evolution(burst_loss, main_loss, loss_plot_path)

        # 4. 画像の保存
        z_restored = samples * (torch.sqrt(z_var) + eps) + z_mean
        rec_proposed = model.decode_first_stage(z_restored)
        
        save_img_individually(rec_proposed, f"{opt.outdir}/burst_reset_snr{snr}.png")
        print(f"Saved result for SNR {snr}")