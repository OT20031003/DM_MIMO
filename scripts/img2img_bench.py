import argparse, os, sys, glob
import torch
import numpy as np
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
import os
import glob # ファイルパスのリストを正規表現で取得するために使用
import lpips

def load_images_as_tensors(dir_path, image_size=(256, 256)):
    """
    指定されたディレクトリ内のすべての画像ファイルを読み込み、
    PyTorchテンソルのリストとして返す。
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
        print(f"警告: ディレクトリ '{dir_path}' にサポートされている画像ファイルが見つかりません。")
        return []

    tensors_list = []
    for t in trange(len(image_paths), desc="Loading Image"):
        path = image_paths[t]
        try:
            img = Image.open(path).convert("RGB")
            tensor_img = transform(img)
            tensors_list.append(tensor_img)
        except Exception as e:
            print(f"エラー: ファイル '{path}' の読み込みに失敗しました。スキップします。エラー内容: {e}")
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
    """
    バッチ画像を個別のファイルとして1枚ずつ保存する関数。
    """
    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    dirname = os.path.dirname(path)
    basename = os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1]

    os.makedirs(dirname, exist_ok=True)

    batch_size = img.shape[0]
    for i in range(batch_size):
        individual_path = os.path.join(dirname, f"{basename}_{i}{ext}")
        vutil.save_image(img[i], individual_path)

    print(f"{batch_size} images are saved in {dirname}/")

def remove_png(path):
    png_files = glob.glob(f'{path}/*.png')
    for file in png_files:
        try:
            os.remove(f"{file}")
        except OSError as e:
            print(f"Error: {e.strerror} - {e.filename}")
    print(f"remove_png complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    T = None # 固定タイムステップ
    t = 2
    N = t
    r = 2
    P_power = 1.0
    Perfect_Estimate = True
    
    # ---------------------------------------------------------
    # ベンチマーク用のディレクトリ名
    # ---------------------------------------------------------
    base_experiment_name = f"SU-MIMO_Benchmark/t={t}_r={r}"

    parser.add_argument("--prompt", type=str, nargs="?", default="benchmark", help="the prompt to render")
    parser.add_argument("--outdir", type=str, nargs="?", default=f"outputs/{base_experiment_name}")
    parser.add_argument("--nosample_outdir", type=str, nargs="?", default=f"outputs/{base_experiment_name}/nosample")
    parser.add_argument("--sentimgdir", type=str, nargs='?', default="./sentimg")
    parser.add_argument("--ddim_steps", type=int, default=200, help="number of ddim sampling steps")
    parser.add_argument("--plms", action='store_true', help="use plms sampling")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta")
    parser.add_argument("--n_iter", type=int, default=1, help="sample this often")
    parser.add_argument("--H", type=int, default=256, help="image height")
    parser.add_argument("--W", type=int, default=256, help="image width")
    parser.add_argument("--n_samples", type=int, default=4, help="how many samples to produce")
    parser.add_argument("--scale", type=float, default=5.0, help="unconditional guidance scale")
    parser.add_argument("--input_path", type=str, default="input_img", help="input image path")
    parser.add_argument("--intermediate_path", type=str, default=None, help="intermediate path")
    parser.add_argument("--intermediate_skip", type=int, default=1, help="intermediate path")
    opt = parser.parse_args()
    
    if opt.intermediate_path != None:
        os.makedirs(opt.intermediate_path, exist_ok=True)
        print(f"{opt.intermediate_path} is created new")
    
    # ディレクトリ設定の整理
    base_path = f"outputs/{base_experiment_name}"
    if Perfect_Estimate == True:
        opt.outdir  = os.path.join(base_path, "perfect_estimate")
        opt.nosample_outdir = os.path.join(base_path, "nosample_perfect")
    else:
        # 推定ありの場合
        opt.outdir  = os.path.join(base_path, "estimated")
        opt.nosample_outdir = os.path.join(base_path, "nosample_estimated")
        
    if T is not None and T >= 0:
        opt.outdir = os.path.join(opt.outdir, f"T={T}")

    print(f"Output Directory: {opt.outdir}")
    
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(opt.sentimgdir, exist_ok=True)
    os.makedirs(opt.nosample_outdir, exist_ok=True)
    outpath = opt.outdir

    # 画像ロード & エンコード
    remove_png(opt.outdir)
    eps = 0.0000001
    img = load_images_as_tensors(opt.input_path)

    batch_size = img.shape[0]
    print(f"img shape = {img.shape}")
    save_img_individually(img, opt.sentimgdir + "/sentimg.png")
    img = img.to(device=device)
    z = model.encode_first_stage(img)
    z = model.get_first_stage_encoding(z).detach()
    
    # 正規化パラメータ保存
    z_encode_mean = z.mean(dim=(1, 2, 3), keepdim=True)
    z_variances_original = torch.var(z, dim=(1, 2, 3)).view(-1, 1, 1, 1)
    
    # 信号正規化 (平均0, 分散1)
    z_norm = (z - z_encode_mean) / (torch.sqrt(z_variances_original) + eps)
    z_variance = torch.var(z_norm, dim=(1, 2, 3)) 
    
    z_channel = z.shape[1]
    z_w_size = z.shape[3]
    z_h_size = z.shape[2]
    
    # 送信電力制約 (複素シンボルあたりの電力=1)
    q_real_data = z_norm / torch.sqrt(2*(z_variance + eps)).view(-1, 1, 1, 1)
    
    # マッピング
    q_view = q_real_data.view(batch_size, t, -1) 
    l = q_view.shape[2] // 2 
    real_part, imag_part = torch.chunk(q_view, 2, dim=2)
    q = torch.complex(real_part, imag_part).to(device)
    
    # パイロット信号
    t_vec = torch.arange(t, device=device)
    N_vec = torch.arange(N, device=device)
    tt, NN = torch.meshgrid(t_vec, N_vec)
    P = torch.sqrt(torch.tensor(P_power/(N*t)))* torch.exp(1j*2*torch.pi*tt*NN/N)


    for snr in range(-5, 26, 3):
        print(f"--------SNR = {snr} (Benchmark)-----------")
        # ノイズ分散 (Simulation Setting / Receiver Knowledge)
        noise_variance = t/(10**(snr/10))
        
        # チャネル生成
        X = q 
        H_real = torch.randn(batch_size, r, t) * torch.sqrt(torch.tensor(0.5))
        H_imag = torch.randn(batch_size, r, t) * torch.sqrt(torch.tensor(0.5))
        H = H_real + H_imag * 1j 
        H = H.to(device)
        
        # パイロット送信
        V_real = torch.randn(batch_size, r, N) * torch.sqrt(torch.tensor(0.5*noise_variance))
        V_imag = torch.randn(batch_size, r, N) * torch.sqrt(torch.tensor(0.5*noise_variance))
        V = V_real + V_imag * 1j 
        V = V.to(device)
        S = H @ P + V 

        # チャネル推定
        H_hat = S @ (P.mH @ torch.inverse(P@P.mH))
        H_tilde = H_hat - H
        
        # データ送信
        W_real = torch.randn(batch_size, r, l) * torch.sqrt(torch.tensor(0.5*noise_variance))
        W_imag = torch.randn(batch_size, r, l) * torch.sqrt(torch.tensor(0.5*noise_variance))
        W = W_real + W_imag * 1j 
        W = W.to(device)

        Y = H @ X + W
        
        # ZFフィルタ計算
        if Perfect_Estimate == True:
            H_for_ZF = H
        else:
            H_for_ZF = H_hat
            
        # ZF行列 A = (H^H H)^-1 H^H
        inv_HH = torch.inverse(H_for_ZF.mH @ H_for_ZF)
        A = inv_HH @ H_for_ZF.mH
        
        AY = A @ Y 
        
        # -- 受信側の雑音分散推定 --
        # ZF等化後の有効雑音は、元の雑音が A 倍されたもの -> 分散は A A^H 倍 (要素ごとの期待値は diag(inv_HH))
        # 受信機は noise_variance (SNR設定) と H_for_ZF を知っているので、有効雑音レベルを計算できる。
        # noise_amplification_factor = mean(diag((H^H H)^-1))
        
        noise_amplification = torch.mean(torch.diagonal(inv_HH.real, dim1=1, dim2=2), dim=1) # Shape: (Batch_Size,)

        current_noise_variance = noise_variance * noise_amplification # Shape: (Batch_Size,)
        # (参考) 真のSINR計算 (デバッグ用)
        # SINR = torch.var(X, dim=(1, 2)) / torch.var(A@(W-H_tilde@X), dim=(1, 2))
        # print(f"Reference SINR mean = {torch.mean(10*torch.log10(SINR))}")

        # --逆符号化--
        AY_real_imag = torch.view_as_real(AY)
        real_part_restored = AY_real_imag[..., 0]
        imag_part_restored = AY_real_imag[..., 1]
        q_view_restored = torch.cat([real_part_restored, imag_part_restored], dim=2)
        q_real_data_restored = q_view_restored.view(batch_size, z_channel, z_h_size, z_w_size)
        # -------------------------------------------------------------
        #  BENCHMARK SPECIFIC SCALING & RECOVERY (Revised)
        # -------------------------------------------------------------
        
        # 1. No Sample 画像 (単純なZF復元)
        # ※ここは変更なし（比較用のため、単純なスケーリングで復元）
        z_nosample = q_real_data_restored * torch.sqrt(2*(z_variance + eps)).view(-1, 1, 1, 1)
        z_nosample = z_nosample * (torch.sqrt(z_variances_original) + eps) + z_encode_mean
        
        recoverd_img_no_samp = model.decode_first_stage(z_nosample)
        save_img_individually(recoverd_img_no_samp, f"{opt.nosample_outdir}/output_{snr}.png")

        # 2. Diffusion Sampling (Blind Denoising)
        # === 修正: 実測値ベースの強制正規化 (Robust Scaling) ===
        
        # 各バッチごとの実際の標準偏差を計算 (Shape: [Batch_Size, 1, 1, 1])
        # これには信号成分(約0.5)と増幅されたノイズ成分の両方が含まれます
        actual_std = q_real_data_restored.std(dim=(1, 2, 3), keepdim=True)
        
        # 強制的に分散1.0に正規化
        # これにより、ZFが不安定で値が暴れても、拡散モデル入力は必ず標準正規分布のスケールになります
        z_input_for_sampler = q_real_data_restored / (actual_std + 1e-8)
        
        print(f"z_input_for_sampler variance = {torch.var(z_input_for_sampler, dim=(1, 2, 3))}")

        # === 重要: サンプラーに渡すノイズ分散の補正 ===
        # 入力を actual_std で割って小さくしたため、その中に含まれるノイズ分散情報も
        # 同じ比率(の2乗)で縮小してサンプラーに伝える必要があります。
        
        actual_var_flat = (actual_std.flatten()) ** 2
        effective_noise_variance = current_noise_variance / actual_var_flat
        
        cond = model.get_learned_conditioning(z.shape[0] * [""])
        
        # サンプリング実行
        # noise_variance には補正後の effective_noise_variance を渡す
        samples = sampler.MIMO_decide_starttimestep_ddim_sampling(
            S=opt.ddim_steps,
            batch_size=batch_size,
            shape=z.shape[1:4],
            x_T=z_input_for_sampler, # 正規化済み入力
            conditioning=cond,
            starttimestep=T,
            noise_variance=effective_noise_variance # 補正済みノイズ分散
        )

        # サンプリング結果を元のスケールに戻す
        z_restored = samples * (torch.sqrt(z_variances_original) + eps) + z_encode_mean
        recoverd_img = model.decode_first_stage(z_restored)
        
        save_img_individually(recoverd_img, f"{opt.outdir}/output_{snr}.png")
        print(f"Saved SNR {snr}")
        