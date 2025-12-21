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

# ==========================================
#  Sionna & TensorFlow Imports
# ==========================================
import tensorflow as tf

# GPU Memory Growth
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
#  Helper Functions
# ==========================================

def plot_channel_evolution(H_true, H_init, H_final, save_path):
    """
    Visualizes channel coefficients (Flattened for OFDM).
    """
    # H shape is [Batch, REs, Rx, Tx]
    h_gt = H_true[0].detach().cpu().numpy().flatten()
    h_ls = H_init[0].detach().cpu().numpy().flatten()
    h_final = H_final[0].detach().cpu().numpy().flatten() if H_final is not None else h_ls

    plt.figure(figsize=(6, 6))
    
    # Downsample for plotting if too many points
    step = max(1, len(h_gt) // 1000)
    
    plt.scatter(h_gt[::step].real, h_gt[::step].imag, c='red', marker='x', s=50, alpha=0.5, label='Ground Truth')
    plt.scatter(h_ls[::step].real, h_ls[::step].imag, c='blue', marker='^', s=40, alpha=0.5, label='Initial Est')
    
    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title("Channel Constellation (OFDM Subsampled)")
    plt.xlabel("Real")
    plt.ylabel("Imag")
    
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
    img = torch.clamp(img, 0.0, 1.0)
    for i in range(img.shape[0]):
        vutil.save_image(img[i], os.path.join(dirname, f"{basename}_{i}{ext}"))

def remove_png(path):
    for file in glob.glob(f'{path}/*.png'):
        try: os.remove(file)
        except: pass

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
    B = z.shape[0]
    z_flat = z.view(B, -1)
    if z_flat.shape[1] % 2 != 0:
        z_flat = torch.cat([z_flat, torch.zeros(B, 1, device=z.device)], dim=1)
    
    s_real, s_imag = torch.chunk(z_flat, 2, dim=1)
    s_complex = torch.complex(s_real, s_imag) / np.sqrt(2.0)
    return s_complex

def complex_symbols_to_latent(s_complex, shape):
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
#  Main Script (Proposed DPS + OFDM)
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # OFDM / MIMO Parameters
    parser.add_argument("--num_tx", type=int, default=2, help="Number of TX antennas (streams)")
    parser.add_argument("--num_rx", type=int, default=2, help="Number of RX antennas")
    
    base_experiment_name = f"MIMO_OFDM_Proposed"
    
    parser.add_argument("--input_path", type=str, default="input_img")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--nosample_outdir", type=str, default=None)
    parser.add_argument("--sentimgdir", type=str, default="./sentimg")
    
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--dps_scale", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    
    opt = parser.parse_args()

    # Seed
    seed_everything(opt.seed)
    
    # Sionna Setup
    t_mimo = opt.num_tx
    r_mimo = opt.num_rx
    num_streams_per_tx = t_mimo 
    
    # OFDM Grid Config
    carrier_frequency = 2.6e9 
    subcarrier_spacing = 30e3 
    fft_size = 76 
    num_guard_carriers = [5, 6] 
    num_ofdm_symbols = 24 
    cyclic_prefix_length = 6
    
    # Antenna & Stream
    ut_array = AntennaArray(num_rows=1, num_cols=int(t_mimo/2), polarization="dual", 
                            polarization_type="cross", antenna_pattern="38.901", 
                            carrier_frequency=carrier_frequency)
    bs_array = AntennaArray(num_rows=1, num_cols=int(r_mimo/2), polarization="dual", 
                            polarization_type="cross", antenna_pattern="38.901", 
                            carrier_frequency=carrier_frequency)
    rx_tx_association = np.ones([1, 1], dtype=int)
    sm = StreamManagement(rx_tx_association, num_streams_per_tx)
    
    # Resource Grid & Layers
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
    
    cdl = CDL("C", 300e-9, carrier_frequency, ut_array, bs_array, "uplink", min_speed=0.0)
    
    rg_mapper = ResourceGridMapper(rg)
    ls_est = LSChannelEstimator(rg, interpolation_type="nn")
    lmmse_equ = LMMSEEqualizer(rg, sm)
    channel_applier = ApplyOFDMChannel(add_awgn=True)
    remove_nulled_scs = RemoveNulledSubcarriers(rg) 
    
    # Directory Setup
    suffix = f"t={t_mimo}_r={r_mimo}"
    if opt.outdir is None: opt.outdir = f"outputs/{base_experiment_name}/{suffix}/estimated"
    if opt.nosample_outdir is None: opt.nosample_outdir = f"outputs/{base_experiment_name}/{suffix}/nosample/estimated"
    channel_outdir = f"outputs/{base_experiment_name}/{suffix}/channel_plots"

    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(opt.nosample_outdir, exist_ok=True)
    os.makedirs(opt.sentimgdir, exist_ok=True)
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
    existing_imgs = glob.glob(os.path.join(opt.sentimgdir, "*.png"))
    if len(existing_imgs) > 0:
        print(f"Loading existing images from {opt.sentimgdir}...")
        img_01 = load_images_as_tensors(opt.sentimgdir).to(device)
    else:
        print(f"Loading images from {opt.input_path}...")
        img_01 = load_images_as_tensors(opt.input_path).to(device)
        save_img_individually(img_01, opt.sentimgdir + "/original.png")

    batch_size = img_01.shape[0]
    print(f"Batch Size: {batch_size}")

    # Latent Encoding
    img_m11 = img_01 * 2.0 - 1.0
    with torch.no_grad():
        z = model.encode_first_stage(img_m11)
        z = model.get_first_stage_encoding(z).detach()
    
    z_mean = z.mean(dim=(1, 2, 3), keepdim=True)
    z_var = torch.var(z, dim=(1, 2, 3)).view(-1, 1, 1, 1)
    eps = 1e-7
    z_norm = (z - z_mean) / (torch.sqrt(z_var) + eps) 

    # Prepare Symbols
    s_complex_torch = latent_to_complex_symbols(z_norm)
    
    num_data_symbols_per_stream = rg.num_data_symbols
    total_grid_capacity = num_data_symbols_per_stream * num_streams_per_tx
    required_symbols = s_complex_torch.shape[1]
    
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
    
    # SNR Loop
    for snr in range(-5, 26, 3):
        print(f"\n======== SNR = {snr} dB (OFDM) ========")
        no = num_streams_per_tx / (10**(snr/10.0)) # scaled by streams

        # 1. Transmission
        x_rg = rg_mapper(x_data_tf)
        cir = cdl(batch_size=batch_size, num_time_steps=rg.num_ofdm_symbols, sampling_frequency=1/rg.ofdm_symbol_duration)
        frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        h_freq = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
        y_rg = channel_applier(x_rg, h_freq, no)
        
        # 2. Reception (LS Est + LMMSE)
        h_hat, err_var = ls_est(y_rg, no)
        x_hat_tf, no_eff_tf = lmmse_equ(y_rg, h_hat, err_var, no)
        
        # 3. Save MMSE Init
        x_hat_torch = torch.from_numpy(x_hat_tf.numpy()).to(device)
        s_hat_flat = x_hat_torch.view(batch_size, -1)
        if pad_len > 0: s_hat_flat = s_hat_flat[:, :-pad_len]
        
        z_init_raw = complex_symbols_to_latent(s_hat_flat, z.shape)
        z_init_mmse = z_init_raw * (torch.sqrt(z_var) + eps) + z_mean
        
        rec_nosample = model.decode_first_stage(z_init_mmse)
        rec_nosample = torch.clamp((rec_nosample + 1.0) / 2.0, 0.0, 1.0)
        save_img_individually(rec_nosample, f"{opt.nosample_outdir}/mmse_ofdm_snr{snr}.png")
        
        # 4. Prepare for DPS (Proposed) with Effective Subcarriers
        # [Corrected Logic] 
        # Apply remove_nulled_scs to both Y and H to get Effective Grid dimensions (1536)
        y_eff_tf = remove_nulled_scs(y_rg)
        h_hat_eff_tf = remove_nulled_scs(h_hat) 
        h_true_eff_tf = remove_nulled_scs(h_freq)
        
        # Convert to numpy
        y_eff_np = y_eff_tf.numpy()
        h_hat_eff_np = h_hat_eff_tf.numpy()
        h_true_eff_np = h_true_eff_tf.numpy()
        
        # --- Reshape H: [B, ..., Rx, ..., Tx, ...] -> [B, REs, Rx, Tx] ---
        H_torch_eff = torch.from_numpy(h_hat_eff_np).to(device)
        # remove_nulled_scs often maintains 1-dims. Reshape safely to [B, Rx, Tx, REs] first
        H_torch_eff = H_torch_eff.reshape(batch_size, r_mimo, t_mimo, -1) 
        H_for_sampler = H_torch_eff.permute(0, 3, 1, 2) # [B, REs, Rx, Tx]
        
        H_true_torch = torch.from_numpy(h_true_eff_np).to(device)
        H_true_torch = H_true_torch.reshape(batch_size, r_mimo, t_mimo, -1)
        H_true_for_plot = H_true_torch.permute(0, 3, 1, 2)
        
        # --- Reshape Y: [B, ..., Rx, ...] -> [B, REs, Rx, 1] ---
        Y_torch_eff = torch.from_numpy(y_eff_np).to(device)
        Y_torch_eff = Y_torch_eff.reshape(batch_size, r_mimo, -1) # [B, Rx, REs]
        Y_for_sampler = Y_torch_eff.permute(0, 2, 1).unsqueeze(-1) # [B, REs, Rx, 1]
        
        # Check size match
        num_eff_REs = H_for_sampler.shape[1] 
        assert H_for_sampler.shape[1] == Y_for_sampler.shape[1], \
            f"Shape Mismatch: H={H_for_sampler.shape}, Y={Y_for_sampler.shape}"

        # Variance calc
        no_eff_scalar = np.mean(no_eff_tf.numpy())
        Sigma_inv_scalar = 1.0 / (no_eff_scalar + 1e-8)
        
        # Initial Z (Normalized)
        actual_std = z_init_mmse.std(dim=(1, 2, 3), keepdim=True)
        z_input_for_sampler = z_init_mmse / (actual_std + 1e-8)
        
        effective_noise_var = (no_eff_scalar / (actual_std.flatten()**2).mean()).item()
        
        # --- Define Mappers for OFDM (Pure PyTorch) ---
        # This replaces the Sionna-based mapper to ensure gradients flow
        def forward_mapper_ofdm(z_in):
            # z_in: [B, C, H, W] -> s_data -> x_eff
            curr_B = z_in.shape[0]
            s = latent_to_complex_symbols(z_in) 
            # We map directly to Effective REs * Streams
            target_len = num_eff_REs * num_streams_per_tx
            s_padded = pad_to_length(s, target_len, dim=1)
            
            # Reshape to [B, Tx, REs]
            s_view = s_padded.view(curr_B, num_streams_per_tx, num_eff_REs)
            
            # Permute to [B, REs, Tx, 1]
            x_flat = s_view.permute(0, 2, 1).unsqueeze(-1)
            
            return x_flat, (curr_B, *z.shape[1:])

        def backward_mapper_ofdm(x_flat, shape):
            # x_flat: [B, REs, Tx, 1] -> [B, Tx, REs] -> flatten -> z
            curr_B = x_flat.shape[0]
            s_view = x_flat.squeeze(-1).permute(0, 2, 1)
            s_flat = s_view.reshape(curr_B, -1)
            return complex_symbols_to_latent(s_flat, shape)

        # Zeta adjustment
        current_zeta = opt.dps_scale
        if snr < 5: current_zeta *= 0.1
        
        # Sampler
        samples, _ = sampler.proposed_dps_sampling(
            S=opt.ddim_steps,
            batch_size=batch_size,
            shape=z.shape[1:],
            conditioning=model.get_learned_conditioning(batch_size * [""]),
            
            y=Y_for_sampler,
            H_hat=H_for_sampler,
            Sigma_inv=torch.tensor(Sigma_inv_scalar, device=device),
            z_init=z_input_for_sampler,
            zeta=current_zeta,
            
            mapper=forward_mapper_ofdm,
            inv_mapper=backward_mapper_ofdm,
            
            initial_noise_variance=effective_noise_var,
            eta=0.0,
            verbose=False
        )
        
        # Plot
        plot_path = os.path.join(channel_outdir, f"channel_plot_snr{snr}.png")
        plot_channel_evolution(H_true_for_plot, H_for_sampler, None, plot_path)
        
        # Decode
        z_restored = samples * (torch.sqrt(z_var) + eps) + z_mean
        rec_proposed = model.decode_first_stage(z_restored)
        rec_proposed = torch.clamp((rec_proposed + 1.0) / 2.0, 0.0, 1.0)
        save_img_individually(rec_proposed, f"{opt.outdir}/proposed_ofdm_snr{snr}.png")
        print(f"Saved result for SNR {snr}")

    print("Experiment Finished.")