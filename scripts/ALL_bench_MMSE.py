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
# DDIMSampler uses the same ddim.py
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

def get_image_paths(dir_path):
    """
    指定されたディレクトリ内の画像パスを取得し、自然順ソートして返す
    """
    image_paths = []
    supported_formats = ["*.jpg", "*.jpeg", "*.png"]
    for fmt in supported_formats:
        image_paths.extend(glob.glob(os.path.join(dir_path, fmt)))

    # 自然順ソート (original_2.png < original_10.png)
    image_paths.sort(key=lambda f: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', os.path.basename(f))])
    
    return image_paths

def load_images_from_paths(paths, image_size=(256, 256)):
    """
    指定されたパスのリストから画像を読み込んでテンソルにする
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    tensors_list = []
    # tqdmは各バッチのロード時に表示
    for path in tqdm(paths, desc="Loading Batch"):
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

def save_img_individually(img, path, start_index=0):
    """
    start_index: バッチ処理時の通し番号の開始位置
    """
    if len(img.shape) == 3: img = img.unsqueeze(0)
    
    dirname = os.path.dirname(path)
    basename = os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1]
    os.makedirs(dirname, exist_ok=True)
    
    img = torch.clamp(img, 0.0, 1.0)

    for i in range(img.shape[0]):
        # 通し番号 (global_index) を使用して保存
        global_idx = start_index + i
        vutil.save_image(img[i], os.path.join(dirname, f"{basename}_{global_idx}{ext}"))
    # print(f"Saved images to {dirname}/ ({start_index} ~ {start_index + img.shape[0] - 1})")

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
#  Main Script (Benchmark: MMSE + Blind Diffusion)
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # MIMO Parameters
    t_mimo = 2 
    r_mimo = 2 
    N_pilot = 2 
    P_power = 1.0 
    Perfect_Estimate = False
    
    # Batch Size Configuration
    BATCH_SIZE = 20  # ここでバッチサイズを指定
    
    base_experiment_name = f"MIMO_Benchmark_MMSE/t={t_mimo}_r={r_mimo}"
    
    parser.add_argument("--input_path", type=str, default="input_img", help="input image path")
    parser.add_argument("--outdir", type=str, nargs="?", default=f"outputs/{base_experiment_name}")
    parser.add_argument("--nosample_outdir", type=str, nargs="?", default=f"outputs/{base_experiment_name}/nosample")
    parser.add_argument("--sentimgdir", type=str, nargs='?', default="./sentimg")
    
    parser.add_argument("--ddim_steps", type=int, default=200, help="number of ddim sampling steps")
    parser.add_argument("--scale", type=float, default=5.0, help="unconditional guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    
    opt = parser.parse_args()

    seed_everything(opt.seed)

    suffix = "perfect" if Perfect_Estimate else "estimated"
    opt.outdir = os.path.join(opt.outdir, suffix)
    opt.nosample_outdir = os.path.join(opt.nosample_outdir, suffix)

    print(f"Output Directory: {opt.outdir}")
    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(opt.sentimgdir, exist_ok=True)
    os.makedirs(opt.nosample_outdir, exist_ok=True)
    remove_png(opt.outdir)

    # Load Model (1回だけロード)
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    # ------------------------------------------------------------------
    # Determine Image Source and Paths
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
        is_new_data = True # 今回初めてロードするのでsentimgdirに保存が必要

    all_image_paths = get_image_paths(source_dir)
    total_images = len(all_image_paths)
    
    if total_images == 0:
        raise ValueError(f"No images found in {source_dir}")

    print(f"Total images: {total_images}. Processing in batches of {BATCH_SIZE}...")

    # ==========================================
    #  Batch Processing Loop
    # ==========================================
    for batch_start_idx in range(0, total_images, BATCH_SIZE):
        batch_end_idx = min(batch_start_idx + BATCH_SIZE, total_images)
        current_batch_paths = all_image_paths[batch_start_idx : batch_end_idx]
        
        print(f"\nProcessing Batch: {batch_start_idx} to {batch_end_idx - 1} ({len(current_batch_paths)} images)")
        
        # Load Batch
        img = load_images_from_paths(current_batch_paths).to(device)
        
        # 新規データの場合はオリジナルを保存 (通し番号を使用)
        if is_new_data:
            save_img_individually(img, opt.sentimgdir + "/original.png", start_index=batch_start_idx)

        batch_size = img.shape[0]

        # 1. Encode
        z = model.encode_first_stage(img)
        z = model.get_first_stage_encoding(z).detach()
        
        # Normalize
        z_mean = z.mean(dim=(1, 2, 3), keepdim=True)
        z_var_original = torch.var(z, dim=(1, 2, 3)).view(-1, 1, 1, 1)
        eps = 1e-7
        z_norm = (z - z_mean) / (torch.sqrt(z_var_original) + eps)
        
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

        # Simulation Loop per SNR
        min_snr = -5
        max_snr = 25
        
        for snr in range(min_snr, max_snr + 1, 3): 
            # print(f"  SNR = {snr} dB")
            
            noise_variance = t_mimo / (10**(snr/10))
            sigma_n = np.sqrt(noise_variance / 2.0)

            # A. Channel
            H_real = torch.randn(batch_size, r_mimo, t_mimo, device=device) * np.sqrt(0.5)
            H_imag = torch.randn(batch_size, r_mimo, t_mimo, device=device) * np.sqrt(0.5)
            H = torch.complex(H_real, H_imag)

            # B. Pilot
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

            # C. Data
            W_real = torch.randn(batch_size, r_mimo, L_len, device=device) * sigma_n
            W_imag = torch.randn(batch_size, r_mimo, L_len, device=device) * sigma_n
            W = torch.complex(W_real, W_imag)
            Y = torch.matmul(H, s_0) + W
            
            # D. MMSE
            eff_noise = sigma_e2 + noise_variance
            H_hat_H = H_hat.mH
            Gram = torch.matmul(H_hat_H, H_hat) 
            Reg = eff_noise * torch.eye(t_mimo, device=device).unsqueeze(0)
            inv_mat = torch.inverse(Gram + Reg)
            W_mmse = torch.matmul(inv_mat, H_hat_H) 
            s_mmse = torch.matmul(W_mmse, Y) 
            
            # E. Reconstruct No-Sample
            z_init_real = mimo_streams_to_latent(s_mmse, latent_shape)
            z_mmse_scaled = z_init_real * np.sqrt(2.0)
            
            z_nosample = z_mmse_scaled * (torch.sqrt(z_var_original) + eps) + z_mean
            rec_nosample = model.decode_first_stage(z_nosample)
            
            # バッチインデックスを渡して保存
            save_img_individually(rec_nosample, f"{opt.nosample_outdir}/mmse_snr{snr}.png", start_index=batch_start_idx)

            # F. Blind Diffusion
            actual_std = z_mmse_scaled.std(dim=(1, 2, 3), keepdim=True)
            z_input_for_sampler = z_mmse_scaled / (actual_std + 1e-8)
            
            W_W_H = torch.matmul(W_mmse, W_mmse.mH)
            noise_power_factor = W_W_H.diagonal(dim1=-2, dim2=-1).real.mean()
            post_mmse_noise_var = eff_noise * noise_power_factor
            
            actual_var_flat = (actual_std.flatten()) ** 2
            effective_noise_variance = (post_mmse_noise_var / actual_var_flat).mean()
            
            cond = model.get_learned_conditioning(batch_size * [""])
            
            samples = sampler.MIMO_decide_starttimestep_ddim_sampling(
                S=opt.ddim_steps,
                batch_size=batch_size,
                shape=z.shape[1:4],
                x_T=z_input_for_sampler,
                conditioning=cond,
                noise_variance=effective_noise_variance,
                starttimestep=None,
                verbose=False
            )

            z_restored = samples * (torch.sqrt(z_var_original) + eps) + z_mean
            rec_bench = model.decode_first_stage(z_restored)
            
            # バッチインデックスを渡して保存
            save_img_individually(rec_bench, f"{opt.outdir}/bench_snr{snr}.png", start_index=batch_start_idx)
        
        # バッチ終了ごとにメモリ解放（念のため）
        torch.cuda.empty_cache()

    print("All batches processed.")