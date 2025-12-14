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

# --- Sionna Imports (v1.2.1+ 構造に対応) ---
import tensorflow as tf
try:
    # GPUメモリの動的確保
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

import sionna
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator
from sionna.phy.channel import RayleighBlockFading, OFDMChannel
from sionna.phy.utils import flatten_last_dims, expand_to_rank
from sionna.phy.channel.tr38901 import TDL

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
    if H_true.dim() > 3: 
        h_gt = H_true[batch_idx, :, :, 0, 0].detach().cpu().numpy().flatten()
        h_ls = H_init[batch_idx, :, :, 0, 0].detach().cpu().numpy().flatten()
        h_gcr = H_final[batch_idx, :, :, 0, 0].detach().cpu().numpy().flatten()
    else:
        h_gt = H_true[batch_idx].detach().cpu().numpy().flatten()
        h_ls = H_init[batch_idx].detach().cpu().numpy().flatten()
        h_gcr = H_final[batch_idx].detach().cpu().numpy().flatten()

    plt.figure(figsize=(8, 8))
    plt.scatter([], [], c='red', marker='x', s=100, linewidths=2, label='Ground Truth')
    plt.scatter([], [], c='blue', marker='^', s=80, label='Initial LS')
    plt.scatter([], [], c='none', edgecolors='green', marker='o', s=120, linewidths=2, label='Final Burst+GCR')
    
    num_elements = len(h_gt)
    limit = min(num_elements, 16) 
    
    for i in range(limit):
        plt.scatter(h_gt[i].real, h_gt[i].imag, c='red', marker='x', s=100, linewidths=2)
        plt.scatter(h_ls[i].real, h_ls[i].imag, c='blue', marker='^', s=80)
        plt.scatter(h_gcr[i].real, h_gcr[i].imag, c='none', edgecolors='green', marker='o', s=120, linewidths=2)
        plt.plot([h_ls[i].real, h_gcr[i].real], [h_ls[i].imag, h_gcr[i].imag], color='gray', linestyle=':', alpha=0.5)
    
    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.title(f"Channel Est (Subcarrier 0) Batch[{batch_idx}]\nMethod: Burst Calibration")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_h_loss_evolution(burst_loss, main_loss, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(burst_loss, color='orange', linewidth=1.5)
    ax1.set_title("Phase 1: Burst Calibration Loss")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(r"$||H_{true} - \hat{H}||^2$")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.plot(main_loss, color='green', linewidth=1.5)
    ax2.set_title("Phase 3: Main GCR Sampling Loss")
    ax2.set_xlabel("Sampling Step")
    ax2.set_ylabel(r"$||H_{true} - \hat{H}||^2$")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle("Evolution of Channel Estimation Error", fontsize=14)
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
    ax1.set_xlabel('Sampling Step')
    ax1.set_ylabel('PSNR (dB)', color=color1)
    line1 = ax1.plot(steps, psnr_list, color=color1, label='PSNR')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('LPIPS', color=color2) 
    line2 = ax2.plot(steps, lpips_list, color=color2, linestyle='--', label='LPIPS')
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.title(f"Image Quality - SNR {snr}dB ({batch_idx})")
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_latent_change(diff_list, save_path, snr, batch_idx=0):
    steps = range(1, len(diff_list) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, diff_list, color='purple', marker='.', linestyle='-')
    plt.xlabel('Sampling Step')
    plt.ylabel('L2 Norm of Difference')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(f"Latent Update Magnitude - SNR {snr}dB ({batch_idx})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed) 
    torch.backends.cudnn.deterministic = True

def load_images_as_tensors(dir_path, image_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    image_paths = glob.glob(os.path.join(dir_path, "*.png")) + glob.glob(os.path.join(dir_path, "*.jpg"))
    if not image_paths: return torch.empty(0)
    image_paths.sort()
    tensors_list = []
    for path in tqdm(image_paths, desc=f"Loading Images"):
        try:
            img = Image.open(path).convert("RGB")
            tensors_list.append(transform(img))
        except Exception as e:
            print(f"Error {path}: {e}")
    return torch.stack(tensors_list, dim=0)

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def save_img_individually(img, path):
    if len(img.shape) == 3: img = img.unsqueeze(0)
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    vutil.save_image(img[0], path)

# ==========================================
#  Sionna / OFDM Helper Functions
# ==========================================

def get_sionna_config(batch_size, num_tx, num_rx, fft_size=64, cp_len=16, num_data_symbols=14):
    """
    Sionna設定: 
    - rx_tx_association は単位行列(np.eye)を使用し、論理的な1対1対応を設定。
    """
    rx_tx_assoc = np.eye(num_rx, num_tx, dtype=int)
    sm = StreamManagement(rx_tx_assoc, num_streams_per_tx=1)
    
    rg = ResourceGrid(num_ofdm_symbols=num_data_symbols,
                      fft_size=fft_size,
                      subcarrier_spacing=30e3,
                      num_tx=num_tx,
                      num_streams_per_tx=1,
                      cyclic_prefix_length=cp_len,
                      pilot_pattern=None, 
                      pilot_ofdm_symbol_indices=[])
    
    return sm, rg

def latent_to_ofdm_grid(z_real, num_tx, fft_size, num_ofdm_symbols):
    """Latent Vector [B,C,H,W] -> OFDM Grid [B,T,F,S] (Complex)"""
    B, C, H, W = z_real.shape
    z_flat = z_real.view(B, -1) 
    
    total_reals = z_flat.shape[1]
    complex_capacity = num_tx * fft_size * num_ofdm_symbols
    
    if (total_reals // 2) > complex_capacity:
        raise ValueError(f"Latent size {total_reals//2} exceeds Grid capacity {complex_capacity}")

    needed_reals = complex_capacity * 2
    if total_reals < needed_reals:
        padding = torch.zeros(B, needed_reals - total_reals, device=z_real.device)
        z_padded = torch.cat([z_flat, padding], dim=1)
    else:
        z_padded = z_flat[:, :needed_reals]
        
    z_view = z_padded.view(B, num_tx, fft_size, num_ofdm_symbols, 2)
    s_complex = torch.complex(z_view[..., 0], z_view[..., 1]) # [B, T, F, S]
    
    return s_complex, (B, C, H, W)

def ofdm_grid_to_latent(s_complex, original_shape):
    """OFDM Grid [B,T,F,S] -> Latent Vector [B,C,H,W]"""
    B = s_complex.shape[0]
    real_part = s_complex.real
    imag_part = s_complex.imag
    
    z_view = torch.stack([real_part, imag_part], dim=-1) # [B, T, F, S, 2]
    z_flat = z_view.view(B, -1)
    
    target_len = np.prod(original_shape[1:])
    z_restored = z_flat[:, :target_len]
    
    return z_restored.view(original_shape)

# ==========================================
#  Main Script
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # MIMO Params
    t_mimo = 2 
    r_mimo = 2 
    
    # OFDM Params (Sionna)
    fft_size = 64
    cp_len = 16
    subcarrier_spacing = 30e3
    
    # Simulation Params
    P_power = 1.0 
    Perfect_Estimate = False 

    parser.add_argument("--input_path", type=str, default="input_img")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--nosample_outdir", type=str, default=None)
    parser.add_argument("--sentimgdir", type=str, default="./sentimg")
    parser.add_argument("--ddim_steps", type=int, default=100)
    parser.add_argument("--burst_iterations", type=int, default=20)
    parser.add_argument("--burst_lr", type=float, default=0.05)
    parser.add_argument("--anchor_lambda", type=float, default=0.0)
    parser.add_argument("--h_lr_max", type=float, default=10.0)
    parser.add_argument("--h_lr_min", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--monitor_range", type=int, nargs=2, default=[0, 2])
    
    opt = parser.parse_args()
    seed_everything(opt.seed)

    latent_elements = 4 * 64 * 64 
    complex_syms_needed = latent_elements // 2
    num_ofdm_symbols = int(np.ceil(complex_syms_needed / (t_mimo * fft_size)))
    
    print(f"MIMO-OFDM Configuration via Sionna (phy namespace):")
    print(f"  Tx: {t_mimo}, Rx: {r_mimo}, FFT: {fft_size}")
    print(f"  Required OFDM Symbols: {num_ofdm_symbols}")

    param_str = (f"OFDM_t={t_mimo}_r={r_mimo}_fft={fft_size}_"
                 f"steps={opt.ddim_steps}_burst={opt.burst_iterations}")
    base_experiment_name = f"MIMO_OFDM_Sionna/{param_str}"
    
    if opt.outdir is None: opt.outdir = f"outputs/{base_experiment_name}/estimated"
    if opt.nosample_outdir is None: opt.nosample_outdir = f"outputs/{base_experiment_name}/estimated/nosample"
    
    base_out_path = os.path.dirname(opt.outdir)
    if os.path.exists(base_out_path): shutil.rmtree(base_out_path)
    
    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(opt.sentimgdir, exist_ok=True)
    os.makedirs(opt.nosample_outdir, exist_ok=True)
    channel_outdir = os.path.join(base_out_path, "channel_plots")
    os.makedirs(channel_outdir, exist_ok=True)
    intermediates_base_dir = os.path.join(base_out_path, "process")
    os.makedirs(intermediates_base_dir, exist_ok=True)

    # --- Load LDM ---
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")
    device = torch.device("cuda")
    model = model.to(device)
    sampler = DDIMSampler(model)
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    # --- Load Data ---
    img_01 = load_images_as_tensors(opt.input_path).to(device)
    if img_01.shape[0] == 0: raise ValueError("No images found.")
    save_img_individually(img_01, opt.sentimgdir + "/original.png")
    
    batch_size = img_01.shape[0]
    img_m11 = img_01 * 2.0 - 1.0 
    gt_imgs = img_01
    
    monitor_indices = list(range(opt.monitor_range[0], min(opt.monitor_range[1], batch_size)))
    
    # --- Encode ---
    with torch.no_grad():
        z = model.encode_first_stage(img_m11)
        z = model.get_first_stage_encoding(z).detach()
        z_mean = z.mean(dim=(1, 2, 3), keepdim=True)
        z_var = torch.var(z, dim=(1, 2, 3)).view(-1, 1, 1, 1)
        z_norm = (z - z_mean) / (torch.sqrt(z_var) + 1e-7)

    s_0_real = z_norm / np.sqrt(2.0)
    s_0, latent_shape = latent_to_ofdm_grid(s_0_real, t_mimo, fft_size, num_ofdm_symbols)
    
    # --- Sionna Setup ---
    sm, rg = get_sionna_config(batch_size, t_mimo, r_mimo, fft_size, cp_len, num_ofdm_symbols)
    
    # TDL (3GPP Channel Model)
    channel_model = TDL(model="A", 
                        delay_spread=100e-9, 
                        min_speed=0.0, 
                        carrier_frequency=3.5e9, 
                        num_rx_ant=r_mimo,       
                        num_tx_ant=t_mimo)       
    
    def apply_channel_torch(H_freq, X_freq):
        # H: [Batch, Rx, Tx, Freq, Sym]
        return torch.einsum('brtfs,btfs->brfs', H_freq, X_freq)

    min_snr_sim = 10
    max_snr_sim = 20

    for snr in range(min_snr_sim, max_snr_sim + 1, 5): 
        print(f"\n======== SNR = {snr} dB ========")
        
        # 1. Generate Channel (Sionna / TensorFlow)
        # a: [batch, 1, rx_ant, 1, tx_ant, num_paths, num_time_steps]
        # tau: [batch, 1, 1, num_paths]
        a, tau = channel_model(batch_size, num_ofdm_symbols, 1/subcarrier_spacing)
        
        # --- 周波数応答の手動計算 (Rank > 5 回避版) ---
        # TensorFlowは高次元(>5)の演算でエラーが出ることがあるため、
        # [Batch, Rx, Tx] をまとめてフラット化して計算を行う。
        
        # a の整形: [B, 1, R, 1, T, P, S] -> [B, R, T, P, S]
        a_sq = tf.squeeze(a, axis=[1, 3]) 
        # 次元数の取得
        dim_combined = batch_size * r_mimo * t_mimo
        
        # フラット化: [Dim, Paths, Syms] (Rank 3)
        a_flat = tf.reshape(a_sq, [dim_combined, -1, num_ofdm_symbols]) 
        a_flat = tf.cast(a_flat, tf.complex64)

        # tau の整形とブロードキャスト
        # tau: [B, 1, 1, P] -> [B, R, T, P] に拡張
        tau_bc = tf.broadcast_to(tau, [batch_size, r_mimo, t_mimo, tau.shape[-1]])
        # フラット化: [Dim, Paths] (Rank 2)
        tau_flat = tf.reshape(tau_bc, [dim_combined, -1]) 
        tau_flat = tf.cast(tau_flat, tf.float32)

        # 位相項の計算: exp(-j * 2pi * f * tau)
        # freq: [F]
        frequencies = tf.range(fft_size, dtype=tf.float32) * subcarrier_spacing
        freq_grid = tf.reshape(frequencies, [1, 1, fft_size]) # [1, 1, F]
        tau_expanded = tf.expand_dims(tau_flat, axis=-1)      # [Dim, P, 1]
        
        # arg: [Dim, P, F] (Rank 3)
        arg = -2.0 * np.pi * tau_expanded * freq_grid
        phase = tf.exp(tf.complex(0.0, arg)) # [Dim, P, F]

        # 周波数応答の合成: H = sum_p (a * phase)
        # a_flat: [Dim, P, S] -> [Dim, P, 1, S]
        a_ready = tf.expand_dims(a_flat, axis=2) 
        # phase: [Dim, P, F] -> [Dim, P, F, 1]
        phase_ready = tf.expand_dims(phase, axis=3)
        
        # 積の計算: [Dim, P, F, S] (Rank 4) - ここでRank 5未満なのでエラー回避可能
        # sum over paths (axis 1): -> [Dim, F, S]
        h_flat = tf.reduce_sum(a_ready * phase_ready, axis=1) 
        
        # 元の形状に戻す: [B, R, T, F, S]
        h_freq_tf = tf.reshape(h_flat, [batch_size, r_mimo, t_mimo, fft_size, num_ofdm_symbols])

        # PyTorchへ転送
        H_freq_gt = torch.from_numpy(h_freq_tf.numpy()).to(device) 
        
        no = 10**(-snr/10.0) 
        sigma_n = np.sqrt(no / 2.0)
        
        # 2. Simulate Received Signal
        s_0_device = s_0.to(device)
        Y_clean = apply_channel_torch(H_freq_gt, s_0_device)
        
        noise_real = torch.randn_like(Y_clean.real) * sigma_n
        noise_imag = torch.randn_like(Y_clean.imag) * sigma_n
        Noise = torch.complex(noise_real, noise_imag)
        Y = Y_clean + Noise
        
        # 3. Initial Estimate (MMSE)
        if Perfect_Estimate:
            H_hat = H_freq_gt
            sigma_e2 = 0.0
        else:
            h_noise_real = torch.randn_like(H_freq_gt.real) * (sigma_n * 0.5) 
            h_noise_imag = torch.randn_like(H_freq_gt.imag) * (sigma_n * 0.5)
            H_hat = H_freq_gt + torch.complex(h_noise_real, h_noise_imag)
            sigma_e2 = (sigma_n * 0.5)**2 

        B_dim, R_dim, T_dim, F_dim, S_dim = H_hat.shape
        H_hat_flat = rearrange(H_hat, 'b r t f s -> (b f s) r t')
        Y_flat = rearrange(Y, 'b r f s -> (b f s) r').unsqueeze(-1) 
        
        H_herm = H_hat_flat.mH
        Gram = torch.matmul(H_herm, H_hat_flat)
        Eye = torch.eye(T_dim, device=device).unsqueeze(0)
        eff_noise = no + sigma_e2
        Inv = torch.inverse(Gram + eff_noise * Eye)
        W_mmse = torch.matmul(Inv, H_herm)
        
        s_mmse_flat = torch.matmul(W_mmse, Y_flat)
        s_mmse = rearrange(s_mmse_flat.squeeze(-1), '(b f s) t -> b t f s', b=B_dim, f=F_dim, s=S_dim)
        
        z_init_real = ofdm_grid_to_latent(s_mmse, latent_shape)
        z_init_mmse = z_init_real * np.sqrt(2.0)
        
        z_nosample = z_init_mmse * (torch.sqrt(z_var) + 1e-7) + z_mean
        rec_nosample = model.decode_first_stage(z_nosample)
        save_img_individually(torch.clamp((rec_nosample+1)/2, 0, 1), f"{opt.nosample_outdir}/mmse_snr{snr}.png")
        
        # 4. Burst Reset Sampling
        def forward_mapper(z):
            s, _ = latent_to_ofdm_grid(z / np.sqrt(2.0), t_mimo, fft_size, num_ofdm_symbols)
            return s
            
        def backward_mapper(s, shape):
            z = ofdm_grid_to_latent(s, shape)
            return z * np.sqrt(2.0)

        effective_noise_variance = torch.tensor(no, device=device) 
        adaptive_h_lr = get_adaptive_h_lr(snr)
        
        print(f"Starting Diffusion with MIMO-OFDM Physics (Sionna Channel)...")
        
        samples, H_final, H_hist, b_loss, m_loss, img_hist = sampler.gcr_burst_sampling(
            S=opt.ddim_steps,
            batch_size=batch_size,
            shape=z.shape[1:4],
            conditioning=model.get_learned_conditioning(batch_size * [""]),
            y=Y,
            H_hat=H_hat, 
            Sigma_inv=torch.tensor(1.0/eff_noise, device=device),
            z_init=z_init_mmse / z_init_mmse.std(),
            burst_iterations=opt.burst_iterations,
            burst_lr=opt.burst_lr,
            anchor_lambda=opt.anchor_lambda,
            zeta=0.3,
            h_lr=adaptive_h_lr,
            mapper=forward_mapper,
            inv_mapper=backward_mapper,
            initial_noise_variance=effective_noise_variance,
            H_true=H_freq_gt,
            measurement_fn=apply_channel_torch 
        )

        rec_final = model.decode_first_stage(samples * (torch.sqrt(z_var)+1e-7) + z_mean)
        save_img_individually(torch.clamp((rec_final+1)/2, 0, 1), f"{opt.outdir}/burst_sionna_snr{snr}.png")
        
        for k in monitor_indices:
            batch_plot_dir = os.path.join(channel_outdir, f"batch_{k}")
            os.makedirs(batch_plot_dir, exist_ok=True)
            plot_channel_evolution(H_freq_gt, H_hat, H_final, 
                                   os.path.join(batch_plot_dir, f"channel_snr{snr}.png"), batch_idx=k)

        print(f"Done SNR {snr}")