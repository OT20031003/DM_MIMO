import argparse, os, sys, glob
import torch
import numpy as np
import random
import re  # 【修正】自然順ソート用にreモジュールを追加
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision import transforms
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from torchvision import utils as vutil
import lpips
import matplotlib.pyplot as plt

# ==========================================
#  Helper Classes & Functions
# ==========================================

class MatrixOperator:
    def __init__(self, tensor):
        self.tensor = tensor

    def __mul__(self, other):
        return torch.matmul(self.tensor, other)

def plot_channel_evolution(H_true, H_init, H_final, save_path):
    """
    チャネル推定の可視化。
    """
    # バッチの先頭[0]を取得し、CPUへ移動・平坦化
    h_gt = H_true[0].detach().cpu().numpy().flatten()
    h_ls = H_init[0].detach().cpu().numpy().flatten()
    h_final = H_final[0].detach().cpu().numpy().flatten()

    plt.figure(figsize=(6, 6))
    
    # 1. Ground Truth (x, Red)
    plt.scatter(h_gt.real, h_gt.imag, c='red', marker='x', s=120, linewidths=2, label='Ground Truth')
    
    # 2. Initial LS Estimate (^, Blue)
    plt.scatter(h_ls.real, h_ls.imag, c='blue', marker='^', s=100, label='Initial LS')
    
    # 3. Final Estimate (o, Green) - ProposedではLSと同じ場所に重なる
    plt.scatter(h_final.real, h_final.imag, c='none', edgecolors='green', marker='o', s=120, linewidths=2, label='Final Estimate')

    # レイアウト調整
    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title("Channel Estimation (Proposed: Fixed)")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    
    # 保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved channel plot to {save_path}")

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
        # print(f"Warning: No images found in {dir_path}")
        return torch.empty(0)

    # 【重要・修正】ファイル名に含まれる数値を考慮して自然順ソートを行う
    # これにより original_2.png が original_10.png より先に来るようになる
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
    
    # 全スクリプトで挙動を揃えるため、明示的にクリップを入れる
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
    
    base_experiment_name = f"MIMO_Proposed_LS/t={t_mimo}_r={r_mimo}"
    
    parser.add_argument("--input_path", type=str, default="input_img")
    parser.add_argument("--outdir", type=str, default=f"outputs/{base_experiment_name}")
    parser.add_argument("--nosample_outdir", type=str, default=f"outputs/{base_experiment_name}/nosample")
    parser.add_argument("--sentimgdir", type=str, default="./sentimg")
    
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--dps_scale", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    
    opt = parser.parse_args()

    # Seed Setting
    seed_everything(opt.seed)

    # Directory Setup
    suffix = "perfect" if Perfect_Estimate else "estimated"
    base_out_path = f"outputs/{base_experiment_name}"

    # Image output
    opt.outdir = os.path.join(opt.outdir, suffix)
    opt.nosample_outdir = os.path.join(opt.nosample_outdir, suffix)
    
    # Channel Plot output (Matched to GCR structure)
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

    # ------------------------------------------------------------------
    # [Modified] Load Images: Check sentimgdir first to preserve order
    # ------------------------------------------------------------------
    # sentimgdir に画像があるか確認
    existing_imgs = glob.glob(os.path.join(opt.sentimgdir, "*.png")) + \
                    glob.glob(os.path.join(opt.sentimgdir, "*.jpg"))
    
    if len(existing_imgs) > 0:
        print(f"Found existing images in {opt.sentimgdir}. Loading from there to preserve order...")
        img = load_images_as_tensors(opt.sentimgdir).to(device)
    else:
        print(f"No existing images in {opt.sentimgdir}. Loading from {opt.input_path}...")
        img = load_images_as_tensors(opt.input_path).to(device)
        # まだ保存されていない場合のみ保存
        save_img_individually(img, opt.sentimgdir + "/original.png")

    if img.shape[0] == 0:
        raise ValueError("No images loaded! Please check input paths.")

    batch_size = img.shape[0]
    # ------------------------------------------------------------------

    # Encode & Normalize
    z = model.encode_first_stage(img)
    z = model.get_first_stage_encoding(z).detach()
    
    z_mean = z.mean(dim=(1, 2, 3), keepdim=True)
    z_var = torch.var(z, dim=(1, 2, 3)).view(-1, 1, 1, 1)
    eps = 1e-7
    z_norm = (z - z_mean) / (torch.sqrt(z_var) + eps)
    
    # ----------------------------------------------------------------
    # 1. Map Latent to MIMO Streams
    # ----------------------------------------------------------------
    s_0_real = z_norm / np.sqrt(2.0)
    s_0, latent_shape = latent_to_mimo_streams(s_0_real, t_mimo)
    s_0 = s_0.to(device)
    
    L_len = s_0.shape[2]
    print(f"MIMO Streams: {t_mimo}x{L_len} complex symbols")

    # ----------------------------------------------------------------
    # 2. Pilot Signal Setup
    # ----------------------------------------------------------------
    t_vec = torch.arange(t_mimo, device=device)
    N_vec = torch.arange(N_pilot, device=device)
    tt, NN = torch.meshgrid(t_vec, N_vec, indexing='ij')
    P = torch.sqrt(torch.tensor(P_power/(N_pilot*t_mimo))) * torch.exp(1j*2*torch.pi*tt*NN/N_pilot)
    P = P.to(device) 

    # Simulation Loop
    for snr in range(-5, 26, 3): 
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
        
        # D. MMSE Initialization
        eff_noise = sigma_e2 + noise_variance
        
        H_hat_H = H_hat.mH
        Gram = torch.matmul(H_hat_H, H_hat) 
        Reg = eff_noise * torch.eye(t_mimo, device=device).unsqueeze(0)
        
        inv_mat = torch.inverse(Gram + Reg)
        W_mmse = torch.matmul(inv_mat, H_hat_H) # (B, t, r)
        
        # Equalization
        s_mmse = torch.matmul(W_mmse, Y) # (B, t, L)
        
        # Save MMSE Result
        z_init_real = mimo_streams_to_latent(s_mmse, latent_shape)
        z_init_mmse = z_init_real * np.sqrt(2.0)
        
        z_nosample = z_init_mmse * (torch.sqrt(z_var) + eps) + z_mean
        rec_nosample = model.decode_first_stage(z_nosample)
        save_img_individually(rec_nosample, f"{opt.nosample_outdir}/mmse_snr{snr}.png")
        
        # E. Prepare for Proposed Method (DPS)
        
        # [Corrected] Noise Variance Normalization
        W_W_H = torch.matmul(W_mmse, W_mmse.mH) # (B, t, t)
        noise_power_factor = W_W_H.diagonal(dim1=-2, dim2=-1).real.mean()
        
        # MMSE後の物理的な残留ノイズ分散
        post_mmse_noise_var_raw = eff_noise * noise_power_factor
        
        # z_initの信号分散で正規化して、Samplerが期待する「相対ノイズ分散」に変換
        actual_std = z_init_mmse.std(dim=(1, 2, 3), keepdim=True)
        actual_var_flat = (actual_std.flatten()) ** 2
        
        # これがbench_MMSE.pyと同じスケールのノイズ分散になる
        effective_noise_variance = (post_mmse_noise_var_raw / actual_var_flat).mean()
        
        # Guidance Variance (Sigma_inv)
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

        # [Match GCR] Adaptive Guidance Scale
        current_zeta = opt.dps_scale
        if snr < 5:
            current_zeta *= 0.1
            print(f"[Info] Low SNR ({snr}dB): Reducing Zeta to {current_zeta:.4f}")
        
        print(f"Starting Proposed Sampling... Steps={opt.ddim_steps}, Zeta={current_zeta}")
        print(f"  > Effective Noise Var: {effective_noise_variance.item():.5f} (Matched with Benchmark)")
        
        # Call Proposed Sampling
        # Returns tuple (img, H_final) to be symmetric with GCR
        samples, H_final_est = sampler.proposed_dps_sampling(
            S=opt.ddim_steps,
            batch_size=batch_size,
            shape=z.shape[1:4], 
            conditioning=cond,
            
            y=Y,                 
            H_hat=H_hat, 
            Sigma_inv=torch.tensor(Sigma_inv, device=device),
            z_init=z_init_normalized, 
            zeta=current_zeta,
            
            mapper=forward_mapper,
            inv_mapper=backward_mapper,
            
            # 修正済みの正規化分散を渡す
            initial_noise_variance=effective_noise_variance,
            
            eta=0.0,
            verbose=False
        )
        
        # --- [Match GCR] Plot Channel Evolution ---
        plot_path = os.path.join(channel_outdir, f"channel_plot_snr{snr}.png")
        plot_channel_evolution(H, H_hat, H_final_est, plot_path)
        # ------------------------------------
        
        # Denormalize & Decode
        z_restored = samples * (torch.sqrt(z_var) + eps) + z_mean
        rec_proposed = model.decode_first_stage(z_restored)
        
        save_img_individually(rec_proposed, f"{opt.outdir}/proposed_snr{snr}.png")
        print(f"Saved result for SNR {snr}")