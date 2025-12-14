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
import time

# 追加インポート (DDIMSamplerのパッチ用)
from ldm.modules.diffusionmodules.util import noise_like

# ==========================================
#  Sionna & TensorFlow Imports / Setup
# ==========================================
import tensorflow as tf

# GPUメモリの動的確保
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"TensorFlow GPU Setup Error: {e}")

from sionna.phy import Block
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, \
                            OFDMModulator, OFDMDemodulator, RZFPrecoder, RemoveNulledSubcarriers
from sionna.phy.channel.tr38901 import AntennaArray, CDL
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, \
                               time_lag_discrete_time_channel, ApplyOFDMChannel, ApplyTimeChannel, \
                               OFDMChannel, TimeChannel
from sionna.phy.utils import ebnodb2no

# ==========================================
#  [PATCH] DDIM Sampler Fix for OFDM (4D H)
# ==========================================
# ddim.py の gcr_burst_sampling をオーバーライドし、4次元 H (OFDM) に対応させます
@torch.no_grad()
def gcr_burst_sampling_fixed(self,
                               S,
                               batch_size,
                               shape,
                               conditioning=None,
                               y=None, 
                               H_hat=None,          
                               Sigma_inv=None,      
                               z_init=None,
                               burst_iterations=50,   
                               burst_lr=0.1,          
                               anchor_lambda=1.0,     
                               h_lr=0.01,             
                               zeta=1.0,              
                               mapper=None,         
                               inv_mapper=None,     
                               initial_noise_variance=None,
                               eta=0., 
                               verbose=True, 
                               unconditional_guidance_scale=1., 
                               unconditional_conditioning=None,
                               H_true=None,
                               monitor_indices=None,
                               phase3_num_steps=None,
                               **kwargs
                               ):
    """
    Early Burst Calibration & Latent Reset Sampling (Patched for 4D Channel Input)
    """
    
    if monitor_indices is None:
        monitor_indices = [0]
    
    # --- Helper: MMSE Solver for Reset (Fixed for 4D H) ---
    def compute_new_initial_latent(y_batch, H_batch, noise_pwr, target_mean, target_std):
        # 【修正箇所】Hの次元数に応じたアンパック
        if H_batch.ndim == 4:
            # [Batch, REs, Rx, Tx] -> 4D
            B_local, REs, r, t = H_batch.shape
            # B_local はバッチサイズそのもの
        else:
            # [Batch, Rx, Tx] -> 3D (Standard MIMO)
            B_local, r, t = H_batch.shape
        
        H_herm = H_batch.mH
        Gram = torch.matmul(H_herm, H_batch)
        
        # Reg項の作成（ブロードキャスト対応）
        # Gram: [B, REs, Tx, Tx] or [B, Tx, Tx]
        Tx = Gram.shape[-1]
        eye = torch.eye(Tx, device=H_batch.device).unsqueeze(0) # [1, Tx, Tx]
        
        # 4Dの場合、Regを [1, 1, Tx, Tx] に拡張するか、自動ブロードキャストに任せる
        # torchのブロードキャストは右揃えなので [1, Tx, Tx] は [B, REs, Tx, Tx] に足せる
        Reg = noise_pwr * eye
        
        inv_mat = torch.inverse(Gram + Reg)
        W_mmse = torch.matmul(inv_mat, H_herm)
        
        s_new = torch.matmul(W_mmse, y_batch)
        
        # inv_mapperは外部で定義されたものを使用
        z_raw = inv_mapper(s_new, (B_local, *shape))
        
        # --- Batch-wise Normalization ---
        z_flat = z_raw.view(B_local, -1)
        batch_mean = z_flat.mean(dim=1, keepdim=True)
        batch_std = z_flat.std(dim=1, keepdim=True)
        
        view_shape = (B_local,) + (1,) * (z_raw.ndim - 1)
        batch_mean = batch_mean.view(view_shape)
        batch_std = batch_std.view(view_shape)

        z_new = (z_raw - batch_mean) / (batch_std + 1e-8)
        z_new = z_new * target_std + target_mean
        
        return z_new

    # 1. Setup & Schedule
    self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=False)
    device = self.model.betas.device
    
    # ノイズ分散の推定
    if initial_noise_variance is not None:
        est_noise_var = initial_noise_variance.mean().item() if torch.is_tensor(initial_noise_variance) else initial_noise_variance
    else:
        avg_precision = Sigma_inv.abs().mean().item()
        est_noise_var = 1.0 / (avg_precision + 1e-8)
    
    # 開始ステップの決定
    if phase3_num_steps is not None:
        requested_index = int(phase3_num_steps) - 1
        start_index = max(0, min(requested_index, S - 1))
        if verbose:
            print(f"[Phase 3 Override] Force sampling for {phase3_num_steps} steps (Index: {start_index}/{S-1})")
    else:
        target_alpha = 1.0 / (1.0 + est_noise_var)
        diffs = torch.abs(self.alphas_cumprod.to(device) - target_alpha)
        start_t_ddpm = torch.argmin(diffs).item()
        
        ddim_timesteps_tensor = torch.from_numpy(self.ddim_timesteps).to(device)
        abs_diff = torch.abs(ddim_timesteps_tensor - start_t_ddpm)
        start_index = torch.argmin(abs_diff).item()
        
        if verbose:
            print(f"[Phase 3 Auto] Estimated start index: {start_index}/{S-1} based on variance {est_noise_var:.4f}")
    
    # 2. Initialization
    z_init = z_init.to(device)
    img = z_init.clone()
    
    if not torch.is_tensor(H_hat):
            raise ValueError("H_hat must be a torch.Tensor.")
    
    current_H = H_hat.clone().detach().requires_grad_(True)
    H_anchor = H_hat.clone().detach()
    H_history = [current_H.detach().cpu().clone()]
    
    burst_loss_history = []
    main_loss_history = []

    # ============================================================
    # Phase 1: Early Burst Calibration
    # ============================================================
    if verbose:
        print(f"--> [Phase 1] Starting Burst Calibration ({burst_iterations} iters)...")

    t_start = self.ddim_timesteps[start_index]
    ts = torch.full((batch_size,), t_start, device=device, dtype=torch.long)
    
    with torch.enable_grad():
        img_in = img.detach().requires_grad_(False)
        
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(img_in, ts, conditioning)
        else:
            x_in = torch.cat([img_in] * 2)
            t_in = torch.cat([ts] * 2)
            c_in = torch.cat([unconditional_conditioning, conditioning])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = self.ddim_alphas
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        a_t = torch.full((batch_size, 1, 1, 1), alphas[start_index], device=device)
        sqrt_one_minus_at = torch.full((batch_size, 1, 1, 1), sqrt_one_minus_alphas[start_index], device=device)
        
        pred_z0_target = (img_in - sqrt_one_minus_at * e_t) / a_t.sqrt()
        s_hat_target, _ = mapper(pred_z0_target.detach()) 
    
    optimizer_H = torch.optim.Adam([current_H], lr=burst_lr)
    pbar = range(burst_iterations)
    
    for i in pbar:
        with torch.enable_grad():
            optimizer_H.zero_grad()
            
            y_est = torch.matmul(current_H, s_hat_target)
            residual = y - y_est
            weighted_res = residual * Sigma_inv
            
            # 損失計算 (次元数によらず全要素平均)
            loss_data = 0.5 * torch.sum(torch.conj(residual) * weighted_res).real / residual.numel() * residual.shape[0] 
            # Note: numelで割るとバッチ平均にならないので、batch倍するか、単純にmeanをとるか調整が必要ですが
            # 元コードに合わせています (K_dim計算が元コードにある場合はそれに従う)
            
            # 元コードの K_dim = residual.shape[1] * residual.shape[2] は 3D前提。
            # 4Dの場合、 REs * Rx * 1 なので shape[1]*shape[2] でよい
            K_dim = np.prod(residual.shape[1:]) 
            loss_data = 0.5 * torch.sum(torch.conj(residual) * weighted_res).real / (batch_size * K_dim) * batch_size

            loss_anchor = 0.5 * torch.nn.functional.mse_loss(
                torch.view_as_real(current_H), 
                torch.view_as_real(H_anchor)
            )
            
            total_loss = loss_data + (anchor_lambda * loss_anchor)
            total_loss.backward()
            optimizer_H.step()

        if H_true is not None:
            with torch.no_grad():
                h_err = torch.norm(current_H - H_true)**2
                burst_loss_history.append(h_err.item())
    
    H_history.append(current_H.detach().cpu().clone())

    if verbose and H_true is not None:
        err_after = torch.norm(current_H - H_true).item()
        print(f"    Post-Burst H Error: {err_after:.4f}")

    # ============================================================
    # Phase 2: Latent Reset
    # ============================================================
    if verbose:
        print("--> [Phase 2] Resetting Latent with Improved H...")
    
    with torch.no_grad():
        z_mean = z_init.mean()
        z_std = z_init.std()
        img = compute_new_initial_latent(
            y, current_H.detach(), est_noise_var, 
            z_mean, z_std
        )

    # ============================================================
    # Phase 3: Main GCR Sampling Loop
    # ============================================================
    timesteps = self.ddim_timesteps[:start_index+1]
    time_range = np.flip(timesteps)
    iterator = tqdm(time_range, desc='GCR (Burst+Reset)', total=len(time_range))

    img_history = [img[monitor_indices].detach().cpu().clone()]

    for i, step in enumerate(iterator):
        index = np.where(self.ddim_timesteps == step)[0][0]
        ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

        # --- A. Gradient Computation ---
        with torch.enable_grad():
            img_in = img.detach().requires_grad_(True)
            
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(img_in, ts, conditioning)
            else:
                x_in = torch.cat([img_in] * 2)
                t_in = torch.cat([ts] * 2)
                c_in = torch.cat([unconditional_conditioning, conditioning])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            alphas = self.ddim_alphas
            sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
            a_t = torch.full((batch_size, 1, 1, 1), alphas[index], device=device)
            sqrt_one_minus_at = torch.full((batch_size, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
            pred_z0 = (img_in - sqrt_one_minus_at * e_t) / a_t.sqrt()

            s_hat, _ = mapper(pred_z0) 
            
            y_est = torch.matmul(current_H, s_hat)
            residual = y - y_est
            weighted_res = residual * Sigma_inv 
            K_dim = np.prod(residual.shape[1:])
            loss_val = 0.5 * torch.sum(torch.conj(residual) * weighted_res).real / (batch_size * K_dim) * batch_size
            
            grads = torch.autograd.grad(loss_val, [img_in, current_H])
            guidance_grad = grads[0]
            h_grad = grads[1]

        # --- B. Update H ---
        if h_lr > 0:
            with torch.no_grad():
                h_grad_norm_tensor = torch.linalg.norm(h_grad)
                if h_grad_norm_tensor > 1e-8:
                    norm_h_grad = h_grad / h_grad_norm_tensor
                else:
                    norm_h_grad = torch.zeros_like(h_grad)
                current_H = current_H - h_lr * norm_h_grad
                current_H = current_H.detach().requires_grad_(True)
        
        H_history.append(current_H.detach().cpu().clone())

        if H_true is not None:
            with torch.no_grad():
                h_err = torch.norm(current_H - H_true)**2
                main_loss_history.append(h_err.item())

        # --- C. Update z ---
        max_timestep = self.ddim_timesteps[-1]
        decay_factor = step / max_timestep
        current_zeta = zeta * decay_factor

        scaled_grad = guidance_grad * current_zeta
        scaled_grad = torch.clamp(scaled_grad, min=-1.0, max=1.0)
        
        # --- D. DDIM Step ---
        with torch.no_grad():
            alphas_prev = self.ddim_alphas_prev
            sigmas = self.ddim_sigmas
            a_prev = torch.full((batch_size, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((batch_size, 1, 1, 1), sigmas[index], device=device)
            
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(img.shape, device, False) * eta
            img_prev_ddim = a_prev.sqrt() * pred_z0 + dir_xt + noise
            
            img = img_prev_ddim - scaled_grad
            img = torch.clamp(img, min=-3.0, max=3.0)
            
            img_history.append(img[monitor_indices].detach().cpu().clone())

            if verbose and (i % 1 == 0):
                logs = {"Loss": f"{loss_val.item():.2f}"}
                if H_true is not None:
                    logs["H_err"] = f"{torch.norm(current_H - H_true).item():.3f}"
                iterator.set_postfix(logs)

    return img, current_H, H_history, burst_loss_history, main_loss_history, img_history

# パッチの適用
DDIMSampler.gcr_burst_sampling = gcr_burst_sampling_fixed


# ==========================================
#  Helper Classes & Functions
# ==========================================

def get_adaptive_h_lr(current_snr, snr_min=0, snr_max=20, lr_max=20.0, lr_min=1.0):
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
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center')
    
    if isinstance(batch_idx, int):
        title = f"Evolution of Image Quality - SNR {snr}dB (Batch {batch_idx})"
    else:
        title = f"Evolution of Image Quality - SNR {snr}dB ({batch_idx})"
    plt.title(title)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_latent_change(diff_list, save_path, snr, batch_idx=0):
    steps = range(1, len(diff_list) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, diff_list, color='purple', marker='.', linestyle='-', linewidth=1.0)
    plt.xlabel('Sampling Step')
    plt.ylabel('L2 Norm of Difference')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(f"Latent Update Magnitude - SNR {snr}dB")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    tf.random.set_seed(seed)

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

# ==========================================
#  OFDM / Latent Mapping Helpers
# ==========================================

def pad_to_length(tensor, target_length, dim=1):
    current_length = tensor.shape[dim]
    if current_length < target_length:
        pad_size = target_length - current_length
        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_size
        padding = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
        return torch.cat([tensor, padding], dim=dim)
    return tensor

def latent_to_complex_symbols(z):
    """
    z: [B, C, H, W] -> Complex Symbols [B, N_syms]
    """
    B = z.shape[0]
    z_flat = z.view(B, -1)
    if z_flat.shape[1] % 2 != 0:
        z_flat = torch.cat([z_flat, torch.zeros(B, 1, device=z.device)], dim=1)
    
    s_real, s_imag = torch.chunk(z_flat, 2, dim=1)
    s_complex = torch.complex(s_real, s_imag) / np.sqrt(2.0)
    return s_complex

def complex_symbols_to_latent(s_complex, shape):
    """
    s_complex: [B, N_syms] -> z: [B, C, H, W]
    """
    z_real = s_complex.real * np.sqrt(2.0)
    z_imag = s_complex.imag * np.sqrt(2.0)
    
    z_flat = torch.cat([z_real, z_imag], dim=1)
    
    target_elements = np.prod(shape[1:])
    current_elements = z_flat.shape[1]
    
    if current_elements > target_elements:
        z_flat = z_flat[:, :target_elements]
    elif current_elements < target_elements:
        z_flat = pad_to_length(z_flat, target_elements, dim=1)
        
    return z_flat.view(shape)

# ==========================================
#  Main Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="input_img")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--nosample_outdir", type=str, default=None)
    parser.add_argument("--sentimgdir", type=str, default="./sentimg")
    
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--burst_iterations", type=int, default=20)
    parser.add_argument("--burst_lr", type=float, default=0.05)
    parser.add_argument("--anchor_lambda", type=float, default=0.0)
    parser.add_argument("--h_lr_max", type=float, default=20.0)
    parser.add_argument("--h_lr_min", type=float, default=0.05)
    parser.add_argument("--dps_scale", type=float, default=0.3)
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--monitor_range", type=int, nargs=2, default=[0, 5])
    
    # OFDM / MIMO Settings
    parser.add_argument("--num_tx", type=int, default=4, help="Number of TX antennas (streams)")
    parser.add_argument("--num_rx", type=int, default=8, help="Number of RX antennas")
    
    opt = parser.parse_args()

    seed_everything(opt.seed)
    
    # ---------------------------------------------------------
    # 1. Sionna OFDM System Configuration
    # ---------------------------------------------------------
    t_mimo = opt.num_tx
    r_mimo = opt.num_rx
    num_streams_per_tx = t_mimo 
    
    # [Config] Resource Grid (FFT size, Guards)
    carrier_frequency = 2.6e9 
    subcarrier_spacing = 30e3 
    fft_size = 76 
    num_guard_carriers = [5, 6] 
    num_ofdm_symbols = 14 
    cyclic_prefix_length = 6
    
    # Antenna Arrays
    ut_array = AntennaArray(num_rows=1, num_cols=int(t_mimo/2), polarization="dual", 
                            polarization_type="cross", antenna_pattern="38.901", 
                            carrier_frequency=carrier_frequency)
    bs_array = AntennaArray(num_rows=1, num_cols=int(r_mimo/2), polarization="dual", 
                            polarization_type="cross", antenna_pattern="38.901", 
                            carrier_frequency=carrier_frequency)
    
    # Stream Management
    rx_tx_association = np.ones([1, 1], dtype=int)
    sm = StreamManagement(rx_tx_association, num_streams_per_tx)
    
    # Resource Grid
    rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                      fft_size=fft_size,
                      subcarrier_spacing=subcarrier_spacing,
                      num_tx=1,
                      num_streams_per_tx=num_streams_per_tx,
                      cyclic_prefix_length=cyclic_prefix_length,
                      num_guard_carriers=num_guard_carriers,  
                      dc_null=True,
                      pilot_pattern="kronecker",
                      pilot_ofdm_symbol_indices=[2, 11])
    
    # CDL Channel
    cdl_model = "C" # NLOS
    delay_spread = 300e-9
    direction = "uplink"
    speed = 0.0
    cdl = CDL(cdl_model, delay_spread, carrier_frequency, ut_array, bs_array, direction, min_speed=speed)
    
    # Sionna Layers
    rg_mapper = ResourceGridMapper(rg)
    ls_est = LSChannelEstimator(rg, interpolation_type="nn")
    lmmse_equ = LMMSEEqualizer(rg, sm)
    channel_applier = ApplyOFDMChannel(add_awgn=True)
    remove_nulled_scs = RemoveNulledSubcarriers(rg) 
    
    # ---------------------------------------------------------
    # 2. Output Paths Setup
    # ---------------------------------------------------------
    param_str = (f"OFDM_t={t_mimo}_r={r_mimo}_steps={opt.ddim_steps}_"
                 f"burst={opt.burst_iterations}_blr={opt.burst_lr}")
    base_experiment_name = f"MIMO_OFDM_Burst/{param_str}"
    
    if opt.outdir is None: opt.outdir = f"outputs/{base_experiment_name}"
    if opt.nosample_outdir is None: opt.nosample_outdir = f"outputs/{base_experiment_name}/nosample"
    
    base_out_path = opt.outdir
    opt.outdir = os.path.join(opt.outdir, "estimated")
    opt.nosample_outdir = os.path.join(opt.nosample_outdir, "estimated")
    channel_outdir = os.path.join(base_out_path, "channel_plots", "estimated")
    intermediates_base_dir = os.path.join(base_out_path, "estimated_process")

    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(opt.sentimgdir, exist_ok=True)
    os.makedirs(opt.nosample_outdir, exist_ok=True)
    os.makedirs(channel_outdir, exist_ok=True)
    os.makedirs(intermediates_base_dir, exist_ok=True)

    print(f"Experiment outputs will be saved to: {opt.outdir}")

    # ---------------------------------------------------------
    # 3. Load Diffusion Model & Images
    # ---------------------------------------------------------
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    lpips_fn = lpips.LPIPS(net='alex').to(device)

    existing_imgs = glob.glob(os.path.join(opt.sentimgdir, "*.png")) + \
                    glob.glob(os.path.join(opt.sentimgdir, "*.jpg"))
    if len(existing_imgs) > 0:
        print(f"Loading existing images from {opt.sentimgdir}...")
        img_01 = load_images_as_tensors(opt.sentimgdir).to(device)
    else:
        print(f"Loading images from {opt.input_path}...")
        img_01 = load_images_as_tensors(opt.input_path).to(device)
        save_img_individually(img_01, opt.sentimgdir + "/original.png")

    batch_size = img_01.shape[0]
    img_m11 = img_01 * 2.0 - 1.0 
    gt_imgs = img_01
    
    start_idx, end_idx = opt.monitor_range
    end_idx = min(end_idx, batch_size)
    monitor_indices = list(range(start_idx, end_idx))
    
    # ---------------------------------------------------------
    # 4. Prepare Latent Variables (z)
    # ---------------------------------------------------------
    with torch.no_grad():
        z = model.encode_first_stage(img_m11)
        z = model.get_first_stage_encoding(z).detach()
    
    z_mean = z.mean(dim=(1, 2, 3), keepdim=True)
    z_var = torch.var(z, dim=(1, 2, 3)).view(-1, 1, 1, 1)
    eps = 1e-7
    z_norm = (z - z_mean) / (torch.sqrt(z_var) + eps) 

    s_complex_torch = latent_to_complex_symbols(z_norm) 
    
    # Check Capacity & Reshape
    num_data_symbols_per_stream = rg.num_data_symbols
    total_grid_capacity = num_data_symbols_per_stream * num_streams_per_tx
    required_symbols = s_complex_torch.shape[1]
    
    print(f"Latent Symbols: {required_symbols}, Grid Capacity: {total_grid_capacity}")
    
    if required_symbols > total_grid_capacity:
        print("Warning: Latent size exceeds grid capacity. Truncating.")
        s_complex_torch = s_complex_torch[:, :total_grid_capacity]
        pad_len = 0
    else:
        pad_len = total_grid_capacity - required_symbols
        s_complex_torch = pad_to_length(s_complex_torch, total_grid_capacity, dim=1)
    
    s_reshaped = s_complex_torch.view(batch_size, 1, num_streams_per_tx, num_data_symbols_per_stream)
    
    x_data_np = s_reshaped.cpu().numpy()
    x_data_tf = tf.convert_to_tensor(x_data_np, dtype=tf.complex64)
    
    # ---------------------------------------------------------
    # 5. SNR Loop
    # ---------------------------------------------------------
    min_snr_sim = 15
    max_snr_sim = 15
    
    for snr in range(min_snr_sim, max_snr_sim + 1, 1):
        print(f"\n======== SNR = {snr} dB (OFDM-MIMO) ========")
        
        no = 1.0 / (10**(snr/10.0))
        
        # --- A. Transmission ---
        x_rg = rg_mapper(x_data_tf) 
        
        cir = cdl(batch_size=batch_size, num_time_steps=rg.num_ofdm_symbols, sampling_frequency=1/rg.ofdm_symbol_duration)
        frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        h_freq = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
        
        y_rg = channel_applier(x_rg, h_freq, no)
        
        # --- B. Reception ---
        h_hat, err_var = ls_est(y_rg, no)
        x_hat_tf, no_eff = lmmse_equ(y_rg, h_hat, err_var, no) 
        
        # --- C. Initial Reconstruction (MMSE) ---
        x_hat_np = x_hat_tf.numpy()
        x_hat_torch = torch.from_numpy(x_hat_np).to(device)
        
        s_hat_flat = x_hat_torch.view(batch_size, -1)
        if pad_len > 0:
            s_hat_flat = s_hat_flat[:, :-pad_len]
            
        z_init_mmse = complex_symbols_to_latent(s_hat_flat, z.shape)
        
        z_nosample = z_init_mmse * (torch.sqrt(z_var) + eps) + z_mean
        rec_nosample = model.decode_first_stage(z_nosample)
        
        rec_nosample_01 = torch.clamp((rec_nosample + 1.0) / 2.0, 0.0, 1.0)
        save_img_individually(rec_nosample_01, f"{opt.nosample_outdir}/mmse_ofdm_snr{snr}.png")
        
        # --- D. GCR Burst Sampling (OFDM Adaptation) ---
        print("Preparing GCR for OFDM...")
        
        # ガードバンド/ヌル除去 (有効サブキャリアのみ抽出)
        y_eff_tf = remove_nulled_scs(y_rg)
        y_eff_np = y_eff_tf.numpy()
        
        h_hat_np = h_hat.numpy() 

        # 次元調整 (Squeeze singleton dims)
        if h_hat_np.ndim == 7: # [B, 1, Rx, 1, Tx, F, T]
            h_hat_np = h_hat_np[:, 0, :, 0, :, :, :]
        elif h_hat_np.ndim == 6: # [B, 1, Rx, Tx, F, T]
            if h_hat_np.shape[1] == 1:
                h_hat_np = h_hat_np.squeeze(1)

        if y_eff_np.ndim == 5: # [B, 1, Rx, F, T]
             if y_eff_np.shape[1] == 1:
                y_eff_np = y_eff_np.squeeze(1)

        H_torch_full = torch.from_numpy(h_hat_np).to(device) # [B, Rx, Tx, F, T]
        Y_torch_full = torch.from_numpy(y_eff_np).to(device)  # [B, Rx, F, T]
        
        if H_torch_full.ndim != 5:
             raise ValueError(f"H_torch_full dim mismatch: Expected 5, got {H_torch_full.ndim}. Shape: {H_torch_full.shape}")

        B, Rx, Tx, nF, nT = H_torch_full.shape
        num_REs = nF * nT
        
        H_for_sampler = H_torch_full.permute(0, 3, 4, 1, 2).reshape(B, num_REs, Rx, Tx)
        Y_for_sampler = Y_torch_full.permute(0, 2, 3, 1).reshape(B, num_REs, Rx, 1)
        
        eff_noise_var_scalar = np.mean(no_eff.numpy())
        Sigma_inv_scalar = 1.0 / (eff_noise_var_scalar + 1e-8)
        
        def forward_mapper_ofdm(z_in):
            s = latent_to_complex_symbols(z_in) 
            target_len = num_REs * num_streams_per_tx
            s_padded = pad_to_length(s, target_len, dim=1)
            s_view = s_padded.view(B, num_streams_per_tx, num_REs) # [B, Tx, REs]
            s_out = s_view.permute(0, 2, 1).unsqueeze(-1) # [B, REs, Tx, 1]
            return s_out, (B, *z.shape[1:])

        def backward_mapper_ofdm(s_in, shape):
            s_view = s_in.squeeze(-1).permute(0, 2, 1) # [B, Tx, REs]
            s_flat = s_view.reshape(B, -1) 
            return complex_symbols_to_latent(s_flat, shape)

        z_init_norm = z_init_mmse / (z_init_mmse.std() + 1e-8)
        
        adaptive_h_lr = get_adaptive_h_lr(snr, lr_max=opt.h_lr_max, lr_min=opt.h_lr_min)
        opt_steps = get_optimal_steps(snr)
        current_zeta = opt.dps_scale * (0.1 if snr < 5 else 1.0)
        
        print(f"Starting GCR Sampling... Steps={opt.ddim_steps}")
        
        try:
            samples, H_final, H_hist, b_loss, m_loss, img_hist = sampler.gcr_burst_sampling(
                S=opt.ddim_steps,
                batch_size=batch_size,
                shape=z.shape[1:],
                conditioning=model.get_learned_conditioning(batch_size * [""]),
                y=Y_for_sampler,
                H_hat=H_for_sampler,
                Sigma_inv=torch.tensor(Sigma_inv_scalar, device=device),
                z_init=z_init_norm,
                burst_iterations=opt.burst_iterations,
                burst_lr=opt.burst_lr,
                anchor_lambda=opt.anchor_lambda,
                zeta=current_zeta,
                h_lr=adaptive_h_lr,
                mapper=forward_mapper_ofdm,
                inv_mapper=backward_mapper_ofdm,
                initial_noise_variance=eff_noise_var_scalar,
                monitor_indices=monitor_indices,
                verbose=True
            )
            
            z_restored = samples * (torch.sqrt(z_var) + eps) + z_mean
            rec_final = model.decode_first_stage(z_restored)
            rec_final_01 = torch.clamp((rec_final + 1.0) / 2.0, 0.0, 1.0)
            save_img_individually(rec_final_01, f"{opt.outdir}/burst_reset_snr{snr}.png")
            
            print("Analyzing results...")
            
            all_psnr = []
            all_lpips = []
            
            for k, batch_idx in enumerate(monitor_indices):
                inter_dir = os.path.join(intermediates_base_dir, f"snr{snr}", f"batch_{batch_idx}")
                os.makedirs(inter_dir, exist_ok=True)
                
                psnr_hist = []
                lpips_hist = []
                latent_diff = []
                
                gt_img = gt_imgs[batch_idx:batch_idx+1]
                z_mean_b = z_mean[batch_idx:batch_idx+1]
                z_var_b = z_var[batch_idx:batch_idx+1]
                
                for step_i in range(len(img_hist)):
                    z_step = img_hist[step_i][k:k+1].to(device)
                    if step_i > 0:
                         z_prev = img_hist[step_i-1][k:k+1].to(device)
                         latent_diff.append(torch.norm(z_step - z_prev).item())
                    
                    z_step_res = z_step * (torch.sqrt(z_var_b) + eps) + z_mean_b
                    with torch.no_grad():
                        rec_step = model.decode_first_stage(z_step_res)
                    
                    p, l = calculate_metrics_single(gt_img, rec_step, lpips_fn)
                    psnr_hist.append(p)
                    lpips_hist.append(l)
                    
                    rec_step_01 = torch.clamp((rec_step + 1.0) / 2.0, 0.0, 1.0)
                    save_img_individually(rec_step_01, os.path.join(inter_dir, f"step_{step_i:03d}.png"))
                
                all_psnr.append(psnr_hist)
                all_lpips.append(lpips_hist)
                
                batch_plot_dir = os.path.join(channel_outdir, f"batch_{batch_idx}")
                os.makedirs(batch_plot_dir, exist_ok=True)
                plot_metrics_evolution(psnr_hist, lpips_hist, 
                                     os.path.join(batch_plot_dir, f"metrics_snr{snr}.png"), 
                                     snr, batch_idx)
                if len(latent_diff) > 0:
                    plot_latent_change(latent_diff, 
                                     os.path.join(batch_plot_dir, f"latent_diff_snr{snr}.png"), 
                                     snr, batch_idx)

            if len(all_psnr) > 0:
                avg_psnr = np.mean(np.array(all_psnr), axis=0)
                avg_lpips = np.mean(np.array(all_lpips), axis=0)
                plot_metrics_evolution(avg_psnr, avg_lpips, 
                                     os.path.join(channel_outdir, f"metrics_snr{snr}_AVG.png"), 
                                     snr, "Average")
        
        except Exception as e:
            print(f"Error during sampling loop at SNR {snr}: {e}")
            import traceback
            traceback.print_exc()

    print("Experiment Finished.")