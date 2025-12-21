import argparse, os, sys, glob
import torch
import numpy as np
import random
import re
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision import transforms
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler 
from torchvision import utils as vutil
import lpips
import matplotlib.pyplot as plt
import shutil

# ... (Previous helper functions: get_adaptive_h_lr, get_optimal_steps, plot_channel_evolution, etc. copied from mimo_dps_burst_reset.py) ...
# To save space in this response, I assume standard helpers are available. 
# I will include the NEW plotting function here.

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

def get_adaptive_h_lr(current_snr, snr_min=-5, snr_max=25, lr_max=20.0, lr_min=1.0):
    if current_snr <= snr_min: return lr_max
    if current_snr >= snr_max: return lr_min
    slope = (lr_min - lr_max) / (snr_max - snr_min)
    return lr_max + (current_snr - snr_min) * slope

def get_optimal_steps(snr):
    steps = 28.33 * np.exp(-0.0879 * snr) - 1.45
    return int(np.clip(np.round(steps), 1, 200))

def latent_to_mimo_streams(z_real, t_antennas):
    B, C, H, W = z_real.shape
    z_flat = z_real.view(B, -1)
    L_complex = z_flat.shape[1] // (t_antennas * 2)
    cutoff = L_complex * t_antennas * 2
    z_used = z_flat[:, :cutoff]
    z_view = z_used.view(B, t_antennas, -1)
    real_part, imag_part = torch.chunk(z_view, 2, dim=2)
    return torch.complex(real_part, imag_part), (B, C, H, W)

def mimo_streams_to_latent(s, original_shape):
    real_part, imag_part = s.real, s.imag
    z_view = torch.cat([real_part, imag_part], dim=2) 
    z_flat = z_view.view(s.shape[0], -1)
    target_size = np.prod(original_shape[1:])
    current_size = z_flat.shape[1]
    if current_size < target_size:
        padding = torch.zeros(s.shape[0], target_size - current_size, device=s.device)
        z_flat = torch.cat([z_flat, padding], dim=1)
    return z_flat.view(original_shape)

# ==========================================
#  NEW: Uncertainty Correlation Plotter
# ==========================================
def plot_uncertainty_correlation(uncertainty_history, img_history, z_true, save_path, batch_idx=0):
    """
    Plots the correlation between pixel-wise squared error |z - z_restored|^2 and the uncertainty map U_t.
    
    Args:
        uncertainty_history: List of tuples (step_index, map_tensor). 
                             map_tensor is (B_sub, 1, H, W).
        img_history: List of latent tensors (B_sub, C, H, W). All steps.
        z_true: Ground truth latent (B_sub, C, H, W).
    """
    
    correlations = []
    steps = []
    
    # Extract only the steps where uncertainty was calculated
    unc_step_indices = [u[0] for u in uncertainty_history]
    unc_maps = [u[1][batch_idx, 0] for u in uncertainty_history] # (H, W)
    
    # Precompute z_true magnitude for normalization (optional, here using raw SE)
    z_true_b = z_true[batch_idx] # (C, H, W)
    
    for i, step_idx in enumerate(unc_step_indices):
        if step_idx >= len(img_history): break
        
        # Get restored latent at this step
        z_restored = img_history[step_idx][batch_idx] # (C, H, W)
        
        # Calculate Squared Error Map (aggregated over channels)
        # Error = sum((z - z_hat)^2, dim=0) -> (H, W)
        error_map = torch.sum((z_true_b - z_restored)**2, dim=0)
        
        # Flatten for correlation
        u_flat = unc_maps[i].detach().cpu().numpy().flatten()
        e_flat = error_map.detach().cpu().numpy().flatten()
        
        # Calculate Pearson Correlation
        if np.std(u_flat) > 1e-6 and np.std(e_flat) > 1e-6:
            corr = np.corrcoef(u_flat, e_flat)[0, 1]
        else:
            corr = 0.0
            
        correlations.append(corr)
        steps.append(step_idx)
        
        # Visualization of the last step maps (optional debug)
        if i == len(unc_step_indices) - 1:
            fig_map, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(e_flat.reshape(unc_maps[i].shape), cmap='hot')
            ax[0].set_title(f"Squared Error Map (Step {step_idx})")
            ax[1].imshow(u_flat.reshape(unc_maps[i].shape), cmap='viridis')
            ax[1].set_title(f"Uncertainty Map (Step {step_idx})")
            plt.close(fig_map)

    # Plot Correlation Evolution
    plt.figure(figsize=(10, 6))
    plt.plot(steps, correlations, marker='o', linestyle='-', color='magenta', label='Correlation')
    plt.xlabel('Sampling Step')
    plt.ylabel('Correlation (Error vs Uncertainty)')
    plt.title(f'Correlation of Uncertainty Map and Squared Error\nBatch {batch_idx}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved uncertainty correlation plot to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Fixed parameters
    t_mimo = 2 
    r_mimo = 2 
    N_pilot = 2 
    P_power = 1.0 
    
    parser.add_argument("--input_path", type=str, default="input_img")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--dps_scale", type=float, default=0.3)
    parser.add_argument("--burst_iterations", type=int, default=20)
    parser.add_argument("--burst_lr", type=float, default=0.05)
    parser.add_argument("--anchor_lambda", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--monitor_range", type=int, nargs=2, default=[0, 2])
    
    # New Arguments for Uncertainty
    parser.add_argument("--uncertainty_interval", type=int, default=10, help="Compute uncertainty every N steps")
    parser.add_argument("--num_uncertainty_samples", type=int, default=4, help="Number of perturbed samples M")

    opt = parser.parse_args()

    seed_everything(opt.seed)
    
    param_str = (f"Uncertainty_t={t_mimo}_steps={opt.ddim_steps}_"
                 f"int={opt.uncertainty_interval}_M={opt.num_uncertainty_samples}")

    if opt.outdir is None:
        opt.outdir = f"outputs/{param_str}"
    
    os.makedirs(opt.outdir, exist_ok=True)
    channel_outdir = os.path.join(opt.outdir, "plots")
    os.makedirs(channel_outdir, exist_ok=True)

    print(f"Output: {opt.outdir}")

    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    # Load Images
    if os.path.exists(opt.input_path):
        img_01 = load_images_as_tensors(opt.input_path).to(device)
    else:
        raise ValueError("Check input path")
    
    # Normalization [-1, 1]
    img_m11 = img_01 * 2.0 - 1.0
    batch_size = img_01.shape[0]
    
    monitor_indices = list(range(opt.monitor_range[0], min(opt.monitor_range[1], batch_size)))

    # Encode
    z = model.encode_first_stage(img_m11)
    z = model.get_first_stage_encoding(z).detach()
    
    z_mean = z.mean(dim=(1, 2, 3), keepdim=True)
    z_var = torch.var(z, dim=(1, 2, 3)).view(-1, 1, 1, 1)
    eps = 1e-7
    z_norm = (z - z_mean) / (torch.sqrt(z_var) + eps)
    
    s_0_real = z_norm / np.sqrt(2.0)
    s_0, latent_shape = latent_to_mimo_streams(s_0_real, t_mimo)
    s_0 = s_0.to(device)

    # Simulation Setup (Fixed SNR for demo or loop)
    # Using a single SNR for demonstration of uncertainty feature
    snr_list = [10] 
    
    t_vec = torch.arange(t_mimo, device=device)
    N_vec = torch.arange(N_pilot, device=device)
    tt, NN = torch.meshgrid(t_vec, N_vec, indexing='ij')
    P = torch.sqrt(torch.tensor(P_power/(N_pilot*t_mimo))) * torch.exp(1j*2*torch.pi*tt*NN/N_pilot)
    P = P.to(device)

    for snr in snr_list:
        print(f"\n=== Running SNR {snr} dB ===")
        noise_variance = t_mimo / (10**(snr/10))
        sigma_n = np.sqrt(noise_variance / 2.0)

        # Channel & Noise
        H_real = torch.randn(batch_size, r_mimo, t_mimo, device=device) * np.sqrt(0.5)
        H_imag = torch.randn(batch_size, r_mimo, t_mimo, device=device) * np.sqrt(0.5)
        H = torch.complex(H_real, H_imag)

        # Pilot Transmission
        V = torch.randn(batch_size, r_mimo, N_pilot, dtype=torch.cfloat, device=device) * np.sqrt(noise_variance/2) # Simple Complex Noise
        S_pilot = torch.matmul(H, P) + V
        
        # LS Estimation
        P_herm = P.mH
        inv_PP = torch.inverse(torch.matmul(P, P_herm))
        H_hat = torch.matmul(S_pilot, torch.matmul(P_herm, inv_PP))
        sigma_e2 = noise_variance / (P_power/t_mimo)

        # Data Transmission
        W = torch.randn(batch_size, r_mimo, s_0.shape[2], dtype=torch.cfloat, device=device) * sigma_n
        Y = torch.matmul(H, s_0) + W

        # MMSE Init
        eff_noise = sigma_e2 + noise_variance
        Sigma_inv = 1.0 / eff_noise
        
        # MMSE Filter
        H_hat_H = H_hat.mH
        Gram = torch.matmul(H_hat_H, H_hat) 
        Reg = eff_noise * torch.eye(t_mimo, device=device).unsqueeze(0)
        inv_mat = torch.inverse(Gram + Reg)
        W_mmse = torch.matmul(inv_mat, H_hat_H) 
        s_mmse = torch.matmul(W_mmse, Y) 
        
        z_init_real = mimo_streams_to_latent(s_mmse, latent_shape)
        z_init_mmse = z_init_real * np.sqrt(2.0)
        
        # Normalize for input
        actual_std = z_init_mmse.std(dim=(1, 2, 3), keepdim=True)
        z_init_normalized = z_init_mmse / (actual_std + 1e-8)
        
        noise_power_factor = torch.matmul(W_mmse, W_mmse.mH).diagonal(dim1=-2, dim2=-1).real.mean()
        post_mmse_noise_var_raw = eff_noise * noise_power_factor
        effective_noise_variance = (post_mmse_noise_var_raw / (actual_std**2).flatten().mean())

        def forward_mapper(z): return latent_to_mimo_streams(z / np.sqrt(2.0), t_mimo)
        def backward_mapper(s, shape): return mimo_streams_to_latent(s, shape) * np.sqrt(2.0)
        
        cond = model.get_learned_conditioning(batch_size * [""])
        
        # Call New Sampler
        samples, H_final, H_hist, burst_loss, main_loss, img_hist, unc_hist = sampler.gcr_uncertainty_sampling(
            S=opt.ddim_steps,
            batch_size=batch_size,
            shape=latent_shape[1:],
            conditioning=cond,
            y=Y,
            H_hat=H_hat,
            Sigma_inv=torch.tensor(Sigma_inv, device=device),
            z_init=z_init_normalized,
            burst_iterations=opt.burst_iterations,
            burst_lr=opt.burst_lr,
            mapper=forward_mapper,
            inv_mapper=backward_mapper,
            initial_noise_variance=effective_noise_variance,
            H_true=H,
            monitor_indices=monitor_indices,
            # Uncertainty Args
            uncertainty_interval=opt.uncertainty_interval,
            num_uncertainty_samples=opt.num_uncertainty_samples,
            zeta=opt.dps_scale
        )
        
        # Decode Final
        z_final = samples * (torch.sqrt(z_var) + eps) + z_mean
        rec = model.decode_first_stage(z_final)
        rec = torch.clamp((rec + 1.0) / 2.0, 0.0, 1.0)
        save_img_individually(rec, f"{opt.outdir}/final_snr{snr}.png")

        # Process Uncertainty & Plot Correlation
        print(f"Processing Uncertainty Plots for SNR {snr}...")
        
        for k, batch_idx in enumerate(monitor_indices):
            
            # Ground Truth Latent for this batch
            z_true_batch = z_norm[batch_idx].detach().cpu() # (C, H, W)
            
            # Plot path
            plot_path = os.path.join(channel_outdir, f"unc_corr_snr{snr}_batch{batch_idx}.png")
            
            plot_uncertainty_correlation(unc_hist, img_hist, z_true_batch.unsqueeze(0), plot_path, batch_idx=k)

    print("Done.")