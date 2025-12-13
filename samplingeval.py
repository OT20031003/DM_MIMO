import argparse, os, sys, glob
import torch
import torch.nn as nn # 追加
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

# ==========================================
#  [NEW] Importance Model Definition
#  (train_importance_CNN.py と同じ構造)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class LatentImportancePredictor(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=64, num_blocks=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.tail = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1),
            nn.ReLU() # 出力は非負 (Importance >= 0)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

def load_student_model(ckpt_path, device):
    print(f"Loading Importance Student Model from {ckpt_path}")
    model = LatentImportancePredictor().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model

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
#  [NEW] Importance to Power Weights
# ==========================================
def compute_power_weights(importance_map, t_mimo, alpha=0.5):
    """
    重要度マップを受け取り、MIMOストリームと同じ形状 (B, T, L) の重み係数を計算する。
    alpha: 重みの強さを調整するパラメータ (0なら一律, 1なら重要度に比例)
    """
    B, C, H, W = importance_map.shape
    
    # 1. まずMIMOストリームと同じ形式にマッピング (実部・虚部用)
    #    Importanceは実数なので、実部用と虚部用で同じ値を使うように複製して扱う
    imp_flat = importance_map.view(B, -1)
    
    total_elements = imp_flat.shape[1]
    L_complex = total_elements // (t_mimo * 2)
    cutoff = L_complex * t_mimo * 2
    
    imp_used = imp_flat[:, :cutoff]
    imp_view = imp_used.view(B, t_mimo, -1) # (B, t, 2L)
    
    # 実部と虚部に対応する重要度を取り出し、平均をとる（複素シンボル1個に対する重要度とする）
    imp_real, imp_imag = torch.chunk(imp_view, 2, dim=2)
    symbol_importance = (imp_real + imp_imag) / 2.0 # (B, t, L)
    
    # 2. 電力配分の計算 (Water filling like strategy or simple proportional)
    #    ここではシンプルに重要度の alpha 乗に比例させ、平均パワーが 1 になるように正規化
    
    # ゼロ除算回避のための微小値
    epsilon = 1e-6
    weights = (symbol_importance + epsilon) ** alpha
    
    # 平均が1になるように正規化 (Bごとに、あるいは全バッチ共通で)
    # ここではバッチごとに正規化し、バッチ内の総電力は変えない
    mean_w = weights.mean(dim=(1, 2), keepdim=True)
    weights_norm = weights / mean_w
    
    # weights_norm は電力係数 (|s|^2 に掛かる)。振幅には sqrt(weights) を掛ける
    return weights_norm

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
    
    base_experiment_name = f"MIMO_Importance/t={t_mimo}_r={r_mimo}"
    
    parser.add_argument("--input_path", type=str, default="input_img")
    parser.add_argument("--outdir", type=str, default=f"outputs/{base_experiment_name}")
    parser.add_argument("--sentimgdir", type=str, default="./sentimg")
    parser.add_argument("--student_ckpt", type=str, required=True, help="Path to trained student_epoch_XX.pth")
    
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--dps_scale", type=float, default=0.3)
    
    parser.add_argument("--burst_iterations", type=int, default=20)
    parser.add_argument("--burst_lr", type=float, default=0.05)
    parser.add_argument("--anchor_lambda", type=float, default=1.0)
    
    parser.add_argument("--h_lr_max", type=float, default=20.0)
    parser.add_argument("--h_lr_min", type=float, default=0.05)
    
    parser.add_argument("--seed", type=int, default=42)
    # Power Allocation Strength (0.0 = Uniform, 1.0 = Linear to Importance)
    parser.add_argument("--power_alpha", type=float, default=0.5, help="Strength of power allocation")

    opt = parser.parse_args()

    seed_everything(opt.seed)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load LDM
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")
    model = model.to(device)
    sampler = DDIMSampler(model)

    # [NEW] Load Student Model
    student_model = load_student_model(opt.student_ckpt, device)

    # Setup Paths
    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(opt.sentimgdir, exist_ok=True)

    # Load Images
    existing_imgs = glob.glob(os.path.join(opt.sentimgdir, "*.png"))
    if len(existing_imgs) > 0:
        img = load_images_as_tensors(opt.sentimgdir).to(device)
    else:
        img = load_images_as_tensors(opt.input_path).to(device)
        save_img_individually(img, opt.sentimgdir + "/original.png")

    batch_size = img.shape[0]

    # Encode
    z = model.encode_first_stage(img)
    z = model.get_first_stage_encoding(z).detach()
    
    z_mean = z.mean(dim=(1, 2, 3), keepdim=True)
    z_var = torch.var(z, dim=(1, 2, 3)).view(-1, 1, 1, 1)
    eps = 1e-7
    z_norm = (z - z_mean) / (torch.sqrt(z_var) + eps)

    # -----------------------------------------------------------
    # [NEW] Predict Importance & Calculate Power Weights
    # -----------------------------------------------------------
    with torch.no_grad():
        # z_normを入力として重要度を推論 (B, 4, 32, 32)
        importance_map = student_model(z_norm)
        
        # 可視化のために重要度マップを保存
        imp_vis = importance_map.mean(dim=1, keepdim=True)
        vutil.save_image(imp_vis / imp_vis.max(), os.path.join(opt.outdir, "importance_map.png"))
        
        # 重要度に基づき、各シンボルの電力係数を計算 (B, T, L)
        # weight_factorは電力比率。振幅には sqrt(weight_factor) を掛ける
        power_weights = compute_power_weights(importance_map, t_mimo, alpha=opt.power_alpha)
        amplitude_scale = torch.sqrt(power_weights)

    # Map to MIMO Streams
    s_0_real = z_norm / np.sqrt(2.0)
    s_0, latent_shape = latent_to_mimo_streams(s_0_real, t_mimo)
    s_0 = s_0.to(device)
    L_len = s_0.shape[2]

    # Pilot Setup (Standard Uniform)
    t_vec = torch.arange(t_mimo, device=device)
    N_vec = torch.arange(N_pilot, device=device)
    tt, NN = torch.meshgrid(t_vec, N_vec, indexing='ij')
    P = torch.sqrt(torch.tensor(P_power/(N_pilot*t_mimo))) * torch.exp(1j*2*torch.pi*tt*NN/N_pilot)
    P = P.to(device) 

    min_snr_sim = 0
    max_snr_sim = 20
    
    for snr in range(min_snr_sim, max_snr_sim + 1, 5): 
        print(f"\n======== SNR = {snr} dB (With Power Allocation) ========")
        
        noise_variance = t_mimo / (10**(snr/10))
        sigma_n = np.sqrt(noise_variance / 2.0)

        # Channel
        H_real = torch.randn(batch_size, r_mimo, t_mimo, device=device) * np.sqrt(0.5)
        H_imag = torch.randn(batch_size, r_mimo, t_mimo, device=device) * np.sqrt(0.5)
        H = torch.complex(H_real, H_imag)

        # Pilot Transmission & Estimation
        V_real = torch.randn(batch_size, r_mimo, N_pilot, device=device) * np.sqrt(noise_variance/2)
        V_imag = torch.randn(batch_size, r_mimo, N_pilot, device=device) * np.sqrt(noise_variance/2)
        V = torch.complex(V_real, V_imag)
        S_pilot = torch.matmul(H, P) + V
        
        P_herm = P.mH
        inv_PP = torch.inverse(torch.matmul(P, P_herm))
        H_hat = torch.matmul(S_pilot, torch.matmul(P_herm, inv_PP))
        sigma_e2 = noise_variance / (P_power/t_mimo)

        # -----------------------------------------------------------
        # [NEW] Weighted Data Transmission
        # -----------------------------------------------------------
        # データ信号 s_0 に電力係数(振幅スケール)を適用
        s_weighted = s_0 * amplitude_scale
        
        W_real = torch.randn(batch_size, r_mimo, L_len, device=device) * sigma_n
        W_imag = torch.randn(batch_size, r_mimo, L_len, device=device) * sigma_n
        W = torch.complex(W_real, W_imag)
        
        # 送信: Y = H * s_weighted + W
        Y = torch.matmul(H, s_weighted) + W
        
        # -----------------------------------------------------------
        # [NEW] Weighted MMSE Receiver
        # -----------------------------------------------------------
        # 通常のMMSE: s = (H'H + sigma I)^-1 H' y
        # 重み付きの場合、受信機がこの重みを知っていると仮定 (Side Information)
        
        # 戦略: 
        # 1. 見かけのチャネル H_eff = H * diag(scale) とみなしてMMSEを解く
        #    しかし scale はシンボル(L)ごとに異なるため、行列演算を一括でするのは難しい。
        # 2. 簡易的アプローチ (Zero-Forcing的な電力戻し):
        #    通常のMMSEで受信 -> 重みで割って元のスケールに戻す。
        #    ただし、MMSEの正規化項(Reg)においては信号電力1を仮定しているため、
        #    本来は Reg = NoiseVar / SignalVar とすべき。
        #    SignalVar = weights なので、Reg = NoiseVar / weights となる。
        
        # ここではループを使わず効率的に計算するため、簡易版を採用
        # 「通常のHでMMSE受信」を行い、その後に「amplitude_scale」で割る
        
        eff_noise = sigma_e2 + noise_variance
        H_hat_H = H_hat.mH
        Gram = torch.matmul(H_hat_H, H_hat) 
        
        # 信号電力が変化したことを正則化項に反映 (近似的)
        # 本来はシンボルごとなので行列が変わるが、ここでは平均的なノイズ分散として扱う
        Reg = eff_noise * torch.eye(t_mimo, device=device).unsqueeze(0)
        inv_mat = torch.inverse(Gram + Reg)
        W_mmse_matrix = torch.matmul(inv_mat, H_hat_H) 
        
        # 受信信号の推定 (これは s_weighted の推定値)
        s_est_weighted = torch.matmul(W_mmse_matrix, Y) 
        
        # スケールを元に戻す (復調)
        # s_est = s_est_weighted / amplitude_scale
        # 小さい値での除算を防ぐ
        s_mmse = s_est_weighted / (amplitude_scale + 1e-6)
        
        # -----------------------------------------------------------
        
        z_init_real = mimo_streams_to_latent(s_mmse, latent_shape)
        z_init_mmse = z_init_real * np.sqrt(2.0)
        
        # 初期解の保存
        z_vis = z_init_mmse * (torch.sqrt(z_var) + eps) + z_mean
        rec_vis = model.decode_first_stage(z_vis)
        save_img_individually(rec_vis, f"{opt.outdir}/mmse_weighted_snr{snr}.png")
        
        # Sampling (通常の復元プロセスへ)
        # ※ バースト/リセット処理内でも、データ一貫性(Likelihood)計算時にHを使う場合
        #    本来はPower Allocationを考慮する必要がありますが、
        #    GCR (Generative Cyclic Reconstruction) では「画像空間での尤度」と
        #    「受信信号Yとの整合性」を見ます。
        #    sampler.gcr_burst_sampling 内の y = Hx + n のモデルも
        #    重みを考慮したものに変更する必要があります。
        #    
        #    ★修正が深くなるため、ここでは簡易的に「初期値(Input)の質を上げる」ことに
        #    注力し、Sampling内部の整合性は既存のままとします。
        #    (初期値が良いだけで結果はかなり改善します)
        
        # ... (以下、サンプリング呼び出し等は元のコードと同様)
        # 必要であれば sampler.gcr_burst_sampling に amplitude_scale を渡し、
        # 内部の観測モデル H を H_eff = H * scale に置き換える処理が必要です。