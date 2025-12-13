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
# DDIMSamplerは同じddim.pyを使用
from ldm.models.diffusion.ddim import DDIMSampler
from torchvision import utils as vutil
import lpips

# ==========================================
#  Helper Functions
# ==========================================

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_images_as_tensors(dir_path, image_size=(256, 256)):
    """
    指定されたディレクトリ内の画像をテンソルとして読み込む
    """
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

    # ファイル名に含まれる数値を考慮して自然順ソートを行う
    image_paths.sort(key=lambda f: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', os.path.basename(f))])

    tensors_list = []
    for t in trange(len(image_paths), desc=f"Loading Images from {dir_path}"):
        path = image_paths[t]
        try:
            img = Image.open(path).convert("RGB")
            tensor_img = transform(img)
            tensors_list.append(tensor_img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    if not tensors_list:
        return torch.empty(0)

    return torch.stack(tensors_list, dim=0)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

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
    png_files = glob.glob(f'{path}/*.png')
    for file in png_files:
        try:
            os.remove(f"{file}")
        except OSError:
            pass

# ==========================================
#  Mappers (Latent <-> MIMO Streams)
# ==========================================
def latent_to_mimo_streams(z_real, t_antennas):
    """
    (Batch, C, H, W) -> (Batch, t, L) Complex
    """
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
    """
    (Batch, t, L) Complex -> (Batch, C, H, W) Real
    """
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
#  Main Script (Benchmark: MMSE + Blind Diffusion)
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # MIMO Parameters
    t_mimo = 2 
    r_mimo = 2 
    N_pilot = 2 
    # python -m scripts.bench_MMSE > output_bench.txt
    P_power = 1.0 
    Perfect_Estimate = False
    
    base_experiment_name = f"MIMO_Benchmark_MMSE/t={t_mimo}_r={r_mimo}"
    
    parser.add_argument("--input_path", type=str, default="input_img", help="input image path")
    parser.add_argument("--outdir", type=str, nargs="?", default=f"outputs/{base_experiment_name}")
    parser.add_argument("--nosample_outdir", type=str, nargs="?", default=f"outputs/{base_experiment_name}/nosample")
    parser.add_argument("--sentimgdir", type=str, nargs='?', default="./sentimg")
    
    parser.add_argument("--ddim_steps", type=int, default=200, help="number of ddim sampling steps")
    parser.add_argument("--scale", type=float, default=5.0, help="unconditional guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    
    opt = parser.parse_args()

    # Seed Setting
    seed_everything(opt.seed)

    # ディレクトリ設定
    suffix = "perfect" if Perfect_Estimate else "estimated"
    opt.outdir = os.path.join(opt.outdir, suffix)
    opt.nosample_outdir = os.path.join(opt.nosample_outdir, suffix)

    print(f"Output Directory: {opt.outdir}")
    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(opt.sentimgdir, exist_ok=True)
    os.makedirs(opt.nosample_outdir, exist_ok=True)
    remove_png(opt.outdir)

    # Load Model
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    # ------------------------------------------------------------------
    # Load Images
    # ------------------------------------------------------------------
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
    # 1. Encode to Latent Space
    # ------------------------------------------------------------------
    # [FIX] Convert [0, 1] -> [-1, 1] to match LDM requirement
    img_m11 = img * 2.0 - 1.0
    z = model.encode_first_stage(img_m11)
    z = model.get_first_stage_encoding(z).detach()
    
    # Normalize Latent
    z_mean = z.mean(dim=(1, 2, 3), keepdim=True)
    z_var_original = torch.var(z, dim=(1, 2, 3)).view(-1, 1, 1, 1)
    eps = 1e-7
    z_norm = (z - z_mean) / (torch.sqrt(z_var_original) + eps)
    
    # 2. Map to MIMO Streams
    s_0_real = z_norm / np.sqrt(2.0) 
    s_0, latent_shape = latent_to_mimo_streams(s_0_real, t_mimo)
    s_0 = s_0.to(device)
    
    L_len = s_0.shape[2]
    print(f"MIMO Streams: {t_mimo}x{L_len} complex symbols")

    # 3. Pilot Signal Setup
    t_vec = torch.arange(t_mimo, device=device)
    N_vec = torch.arange(N_pilot, device=device)
    tt, NN = torch.meshgrid(t_vec, N_vec, indexing='ij')
    P = torch.sqrt(torch.tensor(P_power/(N_pilot*t_mimo))) * torch.exp(1j*2*torch.pi*tt*NN/N_pilot)
    P = P.to(device)

    # Simulation Loop
    min_snr = 0
    max_snr = 15
    
    for snr in range(min_snr, max_snr + 1, 1): 
        print(f"\n======== SNR = {snr} dB (Benchmark: MMSE) ========")
        
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
        
        # D. MMSE Equalization
        eff_noise = sigma_e2 + noise_variance
        
        H_hat_H = H_hat.mH
        Gram = torch.matmul(H_hat_H, H_hat) 
        Reg = eff_noise * torch.eye(t_mimo, device=device).unsqueeze(0)
        
        inv_mat = torch.inverse(Gram + Reg)
        W_mmse = torch.matmul(inv_mat, H_hat_H) 
        
        s_mmse = torch.matmul(W_mmse, Y) 
        
        # E. Reconstruct Latent
        z_init_real = mimo_streams_to_latent(s_mmse, latent_shape)
        z_mmse_scaled = z_init_real * np.sqrt(2.0) # 元のスケールに戻す
        
        # No-Sample Result (単純復号画像の保存)
        z_nosample = z_mmse_scaled * (torch.sqrt(z_var_original) + eps) + z_mean
        rec_nosample = model.decode_first_stage(z_nosample)
        
        # [FIX] Rescale [-1, 1] -> [0, 1] for saving
        rec_nosample = torch.clamp((rec_nosample + 1.0) / 2.0, 0.0, 1.0)
        save_img_individually(rec_nosample, f"{opt.nosample_outdir}/mmse_snr{snr}.png")

        # F. Blind Diffusion Sampling (No DPS, No Guidance)
        # =========================================================
        
        # 1. 入力を標準正規分布に正規化 (Robust Scaling)
        actual_std = z_mmse_scaled.std(dim=(1, 2, 3), keepdim=True)
        z_input_for_sampler = z_mmse_scaled / (actual_std + 1e-8)
        
        # 2. MMSE後の残留ノイズレベルの推定
        W_W_H = torch.matmul(W_mmse, W_mmse.mH)
        noise_power_factor = W_W_H.diagonal(dim1=-2, dim2=-1).real.mean()
        
        post_mmse_noise_var = eff_noise * noise_power_factor
        
        # 3. Samplerへ渡すために、入力分散に対する相対的なノイズ分散へ変換
        actual_var_flat = (actual_std.flatten()) ** 2
        effective_noise_variance = (post_mmse_noise_var / actual_var_flat).mean()
        
        cond = model.get_learned_conditioning(batch_size * [""])
        
        # 4. Sampling
        samples = sampler.MIMO_decide_starttimestep_ddim_sampling(
            S=opt.ddim_steps,
            batch_size=batch_size,
            shape=z.shape[1:4],
            x_T=z_input_for_sampler,       # 正規化済み入力
            conditioning=cond,
            noise_variance=effective_noise_variance, # 計算したノイズレベル
            starttimestep=None,
            verbose=False
        )

        # 5. 復元と保存
        z_restored = samples * (torch.sqrt(z_var_original) + eps) + z_mean
        rec_bench = model.decode_first_stage(z_restored)
        
        # [FIX] Rescale [-1, 1] -> [0, 1] for saving
        rec_bench = torch.clamp((rec_bench + 1.0) / 2.0, 0.0, 1.0)
        save_img_individually(rec_bench, f"{opt.outdir}/bench_snr{snr}.png")
        print(f"Saved result for SNR {snr}")