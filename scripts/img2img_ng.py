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
import glob 
import lpips
import torch.distributions as dist 

# ==========================================
# 修正: 多様な非ガウスノイズ生成用関数
# ==========================================
def get_noise(shape, type='gaussian', device='cuda'):
    """
    指定された分布のノイズを生成し、平均0、分散1に正規化して返す。
    SNR定義を厳密に守るため、理論分散または統計的分散で正規化を行う。
    """
    if type == 'gaussian':
        # 標準正規分布
        return torch.randn(shape, device=device, dtype=torch.float32)
    
    elif type == 'uniform':
        # 一様分布 (分散1に正規化)
        noise = torch.rand(shape, device=device, dtype=torch.float32)
        return (noise - 0.5) * np.sqrt(12)
    
    elif type == 'laplace':
        # ラプラス分布 (分散1に正規化)
        loc = torch.tensor([0.0], device=device, dtype=torch.float32)
        scale = torch.tensor([1.0/np.sqrt(2)], device=device, dtype=torch.float32)
        m = dist.Laplace(loc, scale)
        noise = m.sample(shape).squeeze()
        if noise.dim() > len(shape): noise = noise.view(shape)
        return noise
    
    elif type == 'student_t':
        # 【推定誤差モデル】スチューデントのt分布 (自由度nu=3)
        # 自由度が小さいほど裾が厚くなる(外れ値が増える)。nu=3は分散が定義できる最小に近い値。
        # 分散 = nu / (nu - 2) なので、これで割って分散1にする。
        nu = 3.0
        m = dist.StudentT(df=torch.tensor([nu], device=device))
        noise = m.sample(shape).squeeze()
        if noise.dim() > len(shape): noise = noise.view(shape)
        
        # 正規化: 分散を1にする
        std_dev = np.sqrt(nu / (nu - 2))
        return (noise / std_dev).float()

    elif type == 'impulsive':
        # 【バーストエラーモデル】ベルヌーイ・ガウス混合分布
        # 確率 p で、分散が非常に大きい(K倍)ノイズが発生する。
        # 通信路の「突発的な等化ミス」や「干渉」を模倣。
        
        prob_impulse = 0.1  # 10%の確率でインパルス発生
        k_factor = 10.0     # インパルスは通常ノイズの10倍の標準偏差 (分散は100倍)
        
        # ベースのガウスノイズ
        noise_bg = torch.randn(shape, device=device, dtype=torch.float32)
        # インパルス成分
        noise_impulse = torch.randn(shape, device=device, dtype=torch.float32) * k_factor
        
        # マスク生成 (1ならインパルス)
        mask = torch.bernoulli(torch.full(shape, prob_impulse, device=device)).bool()
        
        # 混合
        noise = torch.where(mask, noise_impulse, noise_bg)
        
        # 正規化計算
        # 理論分散 V = (1-p)*1^2 + p*K^2
        theo_var = (1 - prob_impulse) * 1.0 + prob_impulse * (k_factor ** 2)
        return (noise / np.sqrt(theo_var)).float()

    else:
        raise ValueError(f"Unsupported noise type: {type}")

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

def save_img(img, path):
    if len(img.shape) == 3:
        img.unsqueeze_(0)
    vutil.save_image(img, path, nrow=4)
    print(f"images are saved in {path}")

def save_img_individually(img, path):
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

def caluc_lpips(x,y):
    loss_fn = lpips.LPIPS(net='alex')
    d = loss_fn(x, y)
    return d.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    
    # 修正: ノイズタイプの選択肢を追加
    parser.add_argument(
        "--noise_type",
        type=str,
        default="impulsive",
        choices=["gaussian", "uniform", "laplace", "student_t", "impulsive"],
        help="Type of noise: gaussian, uniform, laplace, student_t (estimation error), impulsive (burst error)"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default=f"outputs/noise_distribution"
    )
    parser.add_argument(
        "--nosample_outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default=f"outputs/noise_distribution"
    )

    parser.add_argument(
        "--sentimgdir",
        type=str, 
        nargs='?',
        help="sent img dir path",
        default="./sentimg"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    
    parser.add_argument(
        "--input_path",
        type=str, 
        default="input_img",
        help="input image path"
    )
    parser.add_argument(
        "--intermediate_path",
        type=str, 
        default=None,
        help="intermediate path"
    )
    parser.add_argument(
        "--intermediate_skip",
        type=int, 
        default=1,
        help="intermediate path"
    )
    opt = parser.parse_args()
    
    # 出力パス設定
    opt.outdir = os.path.join(opt.outdir, opt.noise_type)
    opt.outdir = os.path.join(opt.outdir, "DM")
    opt.nosample_outdir = os.path.join(opt.nosample_outdir, opt.noise_type)
    opt.nosample_outdir = os.path.join(opt.nosample_outdir, "nodiffusion")
    if opt.intermediate_path != None:
        os.makedirs(opt.intermediate_path, exist_ok=True)
        print(f"{opt.intermediate_path} is created new")

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

    prompt = opt.prompt

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    
    # 画像をロード
    remove_png(opt.outdir)
    
    img = load_images_as_tensors(opt.input_path)
    print(f"img shape = {img.shape}")
    save_img_individually(img, opt.sentimgdir + "/sentimg.png")
    img = img.to(device="cuda")
    z = model.encode_first_stage(img)
    
    print(f"encode start = ")
    z = model.get_first_stage_encoding(z).detach()
    print(f"z = {z.shape}, z_max = {z.max()}, z_min = {z.min()}")
    z_variances = torch.var(z, dim=(1, 2, 3))
    z = z / z_variances.view(-1, 1, 1, 1)
    print(f"z_variance = {z_variances}")
    
    z_copy = z.clone()

    for snr in range(-5, 26, 5):
        print(f"--------SNR = {snr}, Noise Type = {opt.noise_type}-----------")
        z = z_copy.clone()
        snrp = pow(10, snr/10) 
        noise_variances = z_variances/snrp 
        scalar_variance = 1 / snrp 
        
        base_noise = get_noise(z.shape, type=opt.noise_type, device="cuda")
        
        if isinstance(noise_variances, torch.Tensor):
            scale_src = noise_variances.clone().detach()
        else:
            scale_src = torch.tensor(noise_variances)
            
        scale = torch.sqrt(scale_src.view(-1, 1, 1, 1).to("cuda"))
        
        noise = base_noise * scale
        
        z = z + noise
        z *= z_variances.view(-1, 1, 1, 1)
        z = z.float() 
        
        recoverd_img_no_samp = model.decode_first_stage(z)
        cond = model.get_learned_conditioning(z.shape[0] * [""])
        print(f"####cond finisihed #####")
        
        samples = sampler.my_ddim_sampling(S=opt.ddim_steps, batch_size=z.shape[0], 
                        shape= z.shape[1:4], x_T=z, noise_variance=scalar_variance,
                        conditioning=cond, intermediate_path=opt.intermediate_path, intermediate_skip=opt.intermediate_skip, snr=snr)
    
        print(f"d = {samples.shape}")
        recoverd_img = model.decode_first_stage(samples)
        print(f"recoverd_img = {recoverd_img.shape}")
        save_img_individually(recoverd_img, f"{opt.outdir}/output_{snr}.png")
        save_img_individually(recoverd_img_no_samp, f"{opt.nosample_outdir}/output_{snr}.png")