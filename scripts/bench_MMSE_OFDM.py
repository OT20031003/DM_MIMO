import argparse, os, sys, glob
import torch
import numpy as np
import random
import re
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from torchvision.utils import make_grid
from torchvision import transforms
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler 
from torchvision import utils as vutil
import matplotlib.pyplot as plt
import shutil
import time

# ==========================================
#  Sionna & TensorFlow Imports
# ==========================================
import tensorflow as tf

# Configure GPU Memory Growth for TensorFlow
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
#  Helper Functions (General)
# ==========================================

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
    
    # Clip to [0, 1] before saving
    img = torch.clamp(img, 0.0, 1.0)
    
    for i in range(img.shape[0]):
        vutil.save_image(img[i], os.path.join(dirname, f"{basename}_{i}{ext}"))

def remove_png(path):
    png_files = glob.glob(f'{path}/*.png')
    for file in png_files:
        try:
            os.remove(f"{file}")
        except OSError:
            pass

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
    # Ensure even number of elements for complex pairing
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
#  Main Benchmarking Script (OFDM Version)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_path", type=str, default="input_img")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--nosample_outdir", type=str, default=None)
    parser.add_argument("--sentimgdir", type=str, default="./sentimg")
    
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    
    # OFDM / MIMO Settings
    parser.add_argument("--num_tx", type=int, default=2, help="Number of TX antennas (streams)")
    parser.add_argument("--num_rx", type=int, default=2, help="Number of RX antennas")
    
    opt = parser.parse_args()

    seed_everything(opt.seed)
    
    # ---------------------------------------------------------
    # 1. Sionna OFDM System Configuration
    # ---------------------------------------------------------
    t_mimo = opt.num_tx
    r_mimo = opt.num_rx
    num_streams_per_tx = t_mimo 
    
    # [Config] Resource Grid (Same as MYOFDM.py)
    carrier_frequency = 2.6e9 
    subcarrier_spacing = 30e3 
    fft_size = 76 
    num_guard_carriers = [5, 6] 
    
    # Increase symbols to ensure capacity for 256x256 latents (approx 2048 complex syms)
    num_ofdm_symbols = 24 
    cyclic_prefix_length = 6
    
    # Antenna Arrays (38.901)
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
    
    # CDL Channel (NLOS)
    cdl_model = "C" 
    delay_spread = 300e-9
    direction = "uplink"
    speed = 0.0
    cdl = CDL(cdl_model, delay_spread, carrier_frequency, ut_array, bs_array, direction, min_speed=speed)
    
    # Sionna Layers
    rg_mapper = ResourceGridMapper(rg)
    ls_est = LSChannelEstimator(rg, interpolation_type="nn")
    lmmse_equ = LMMSEEqualizer(rg, sm)
    channel_applier = ApplyOFDMChannel(add_awgn=True)
    
    # ---------------------------------------------------------
    # 2. Output Paths Setup
    # ---------------------------------------------------------
    base_experiment_name = f"MIMO_Benchmark_OFDM/t={t_mimo}_r={r_mimo}"
    
    if opt.outdir is None: opt.outdir = f"outputs/{base_experiment_name}"
    if opt.nosample_outdir is None: opt.nosample_outdir = f"outputs/{base_experiment_name}/nosample"
    
    opt.outdir = os.path.join(opt.outdir, "estimated")
    opt.nosample_outdir = os.path.join(opt.nosample_outdir, "estimated")

    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(opt.sentimgdir, exist_ok=True)
    os.makedirs(opt.nosample_outdir, exist_ok=True)
    
    remove_png(opt.outdir)
    remove_png(opt.nosample_outdir)

    print(f"Experiment outputs will be saved to: {opt.outdir}")

    # ---------------------------------------------------------
    # 3. Load Diffusion Model & Images
    # ---------------------------------------------------------
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

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
    print(f"Batch Size: {batch_size}")

    img_m11 = img_01 * 2.0 - 1.0 
    
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
    
    print(f"Latent Symbols needed: {required_symbols}, Grid Capacity: {total_grid_capacity}")
    
    if required_symbols > total_grid_capacity:
        print("Warning: Latent size exceeds grid capacity. Truncating.")
        s_complex_torch = s_complex_torch[:, :total_grid_capacity]
        pad_len = 0
    else:
        pad_len = total_grid_capacity - required_symbols
        s_complex_torch = pad_to_length(s_complex_torch, total_grid_capacity, dim=1)
    
    # [Batch, 1, Streams, Time]
    s_reshaped = s_complex_torch.view(batch_size, 1, num_streams_per_tx, num_data_symbols_per_stream)
    
    x_data_np = s_reshaped.cpu().numpy()
    x_data_tf = tf.convert_to_tensor(x_data_np, dtype=tf.complex64)
    
    # ---------------------------------------------------------
    # 5. SNR Loop
    # ---------------------------------------------------------
    min_snr = -5
    max_snr = 25
    
    for snr in range(min_snr, max_snr + 1, 3):
        print(f"\n======== SNR = {snr} dB (OFDM-MIMO {t_mimo}x{r_mimo}) ========")
        
        # Scale noise by number of transmit streams
        no = num_streams_per_tx / (10**(snr/10.0))
        
        # --- A. Transmission ---
        x_rg = rg_mapper(x_data_tf) 
        
        cir = cdl(batch_size=batch_size, num_time_steps=rg.num_ofdm_symbols, sampling_frequency=1/rg.ofdm_symbol_duration)
        frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        h_freq = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
        
        y_rg = channel_applier(x_rg, h_freq, no)
        
        # --- B. Reception (LS Estimation + LMMSE) ---
        h_hat, err_var = ls_est(y_rg, no)
        
        # LMMSE returns (Estimated Symbols, Effective Noise Variance)
        x_hat_tf, no_eff_tf = lmmse_equ(y_rg, h_hat, err_var, no) 
        
        # --- C. Convert back to PyTorch ---
        x_hat_np = x_hat_tf.numpy()
        x_hat_torch = torch.from_numpy(x_hat_np).to(device)
        
        s_hat_flat = x_hat_torch.view(batch_size, -1)
        if pad_len > 0:
            s_hat_flat = s_hat_flat[:, :-pad_len]
            
        z_out_real = complex_symbols_to_latent(s_hat_flat, z.shape)
        
        # Restore scaling (Inverse of z_norm calculation)
        z_mmse_scaled = z_out_real * (torch.sqrt(z_var) + eps) + z_mean

        # ---------------------------------------------------------
        # Metric 1: No-Sample Result (Pure LMMSE)
        # ---------------------------------------------------------
        with torch.no_grad():
            rec_nosample = model.decode_first_stage(z_mmse_scaled)
        
        rec_nosample_01 = torch.clamp((rec_nosample + 1.0) / 2.0, 0.0, 1.0)
        save_img_individually(rec_nosample_01, f"{opt.nosample_outdir}/mmse_ofdm_snr{snr}.png")

        # ---------------------------------------------------------
        # Metric 2: Blind Diffusion Sampling (Robust Scaling + Noise Est)
        # ---------------------------------------------------------
        
        # 1. Normalize Input for Sampler (Robust Scaling)
        # z_mmse_scaled contains (Signal * alpha + Noise). 
        # We normalize it to Standard Normal for the LDM.
        actual_std = z_mmse_scaled.std(dim=(1, 2, 3), keepdim=True)
        z_input_for_sampler = z_mmse_scaled / (actual_std + 1e-8)
        
        # 2. Estimate Effective Noise Variance Ratio
        # Sionna provides 'no_eff' which is the post-equalization noise variance.
        # We need the ratio of this noise variance to the total signal variance.
        no_eff_scalar = np.mean(no_eff_tf.numpy()) # Mean effective noise variance per symbol
        
        # Since we scaled z_mmse_scaled by 'actual_std', we must scale the noise variance too.
        # effective_noise_variance = no_eff / (actual_std^2)
        # Note: 'no_eff' is variance of complex symbols. Latent z is real. 
        # Variance of Real part = 1/2 * Variance of Complex. 
        # But 'latent_to_complex' logic involves sqrt(2).
        # Let's approximate using the ratio of variances directly.
        
        actual_var_flat = (actual_std.flatten()) ** 2
        effective_noise_variance = (no_eff_scalar / actual_var_flat).mean()

        print(f"  [Blind Diff] Eff Noise Var (Sionna): {no_eff_scalar:.5f}")
        print(f"  [Blind Diff] Scaled Noise Ratio: {effective_noise_variance:.5f}")

        cond = model.get_learned_conditioning(batch_size * [""])
        
        # 3. Sampling
        samples = sampler.MIMO_decide_starttimestep_ddim_sampling(
            S=opt.ddim_steps,
            batch_size=batch_size,
            shape=z.shape[1:4],
            x_T=z_input_for_sampler,       # Normalized Input
            conditioning=cond,
            noise_variance=effective_noise_variance, # Estimated Noise Level
            starttimestep=None, # Let it decide based on noise_variance
            verbose=False
        )

        # 4. Decode & Save
        z_restored = samples * (torch.sqrt(z_var) + eps) + z_mean
        rec_bench = model.decode_first_stage(z_restored)
        
        rec_bench = torch.clamp((rec_bench + 1.0) / 2.0, 0.0, 1.0)
        save_img_individually(rec_bench, f"{opt.outdir}/bench_ofdm_snr{snr}.png")
        print(f"Saved result for SNR {snr}")

    print("Benchmarking Finished.")