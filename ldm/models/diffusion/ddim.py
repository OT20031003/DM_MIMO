import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import os
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
import torchvision.utils as vutil
import torch
def print_debug_stats(tensor, name="Tensor"):
    if tensor is None:
        print(f"[{name}] is None!")
        return
    
    # Detach and move to CPU for safe printing
    t = tensor.detach().cpu().float()
    
    min_val = t.min().item()
    max_val = t.max().item()
    mean_val = t.mean().item()
    std_val = t.std().item()
    norm_val = torch.norm(t).item()
    
    print(f"  [DEBUG] {name:20s} | Shape: {tuple(t.shape)} | Range: [{min_val:.4f}, {max_val:.4f}] | Mean: {mean_val:.4f} | Std: {std_val:.4f} | Norm: {norm_val:.4f}")
# def print_debug_stats(tensor, name="Tensor"):
#     """
#     テンソルの統計情報を表示して、異常値（NaN/Inf/全ゼロ）を検知します。
#     """
#     if tensor is None:
#         print(f"[{name}] is None!")
#         return

#     # GPUにある場合はCPUに移して計算
#     t = tensor.detach().cpu().float()
    
#     min_val = t.min().item()
#     max_val = t.max().item()
#     mean_val = t.mean().item()
#     std_val = t.std().item()
#     has_nan = torch.isnan(t).any().item()
#     has_inf = torch.isinf(t).any().item()
    
#     print(f"--- DEBUG: {name} ---")
#     print(f"    Shape: {t.shape}")
#     print(f"    Range: [{min_val:.5f}, {max_val:.5f}]")
#     print(f"    Mean:  {mean_val:.5f} | Std: {std_val:.5f}")
    
#     if has_nan:
#         print(f"    !!! WARNING: Contains NaN (Not a Number) !!!")
#     if has_inf:
#         print(f"    !!! WARNING: Contains Inf (Infinity) !!!")
#     if min_val == 0.0 and max_val == 0.0:
#         print(f"    !!! WARNING: All values are ZERO !!!")
#     print("-------------------------")
def orthogonal_guidance(pred_noise, known_noise, current_scale):
    """
    予測ノイズから既知ノイズと直交する成分（有色雑音や幻覚）を
    current_scale の割合だけ減算するヘルパー関数。
    """
    n_b = pred_noise.shape[0]
    pred_flat = pred_noise.reshape(n_b, -1)
    known_flat = known_noise.reshape(n_b, -1)

    # 射影係数: alpha = (pred . known) / (known . known)
    dot_pk = torch.sum(pred_flat * known_flat, dim=1, keepdim=True)
    dot_kk = torch.sum(known_flat * known_flat, dim=1, keepdim=True)
    dot_kk = torch.clamp(dot_kk, min=1e-8)
    
    alpha = dot_pk / dot_kk
    
    # 成分分解
    alpha_view = alpha.view(n_b, 1, 1, 1)
    eps_parallel = alpha_view * known_noise       # 既知ノイズと平行な成分（信頼できる）
    eps_orthogonal = pred_noise - eps_parallel    # 直交成分（怪しい有色雑音など）

    # 直交成分を current_scale 分だけ引く
    # scale=0.2 なら、直交成分を 20% 削減して 80% 残す
    eps_modified = eps_parallel + (1.0 - current_scale) * eps_orthogonal
    
    return eps_modified
def extract_colored_noise_smooth(pred_noise, known_noise, removal_rate=0.0):
    """
    有色成分は常に100%使用する。
    白色成分（既知ノイズ平行成分）は removal_rate の割合だけ使用する。
    
    Args:
        removal_rate (float): 0.0 = 白色成分を使わない（画像に残す）
                              1.0 = 白色成分もフルに使う（通常除去）
                              0.0 -> 1.0 に徐々に変化させる
    """
    n_b = pred_noise.shape[0]
    pred_flat = pred_noise.reshape(n_b, -1)
    known_flat = known_noise.reshape(n_b, -1)

    # 1. 平行成分（白色成分）の抽出
    dot_pk = torch.sum(pred_flat * known_flat, dim=1, keepdim=True)
    dot_kk = torch.sum(known_flat * known_flat, dim=1, keepdim=True)
    dot_kk = torch.clamp(dot_kk, min=1e-8)
    
    alpha = dot_pk / dot_kk
    alpha_view = alpha.view(n_b, 1, 1, 1)
    
    eps_parallel = alpha_view * known_noise  # これが「除去候補」の白色成分
    eps_colored = pred_noise - eps_parallel  # これは常に除去する有色成分
    
    # 2. 混合
    # rate=0 なら eps_colored のみ（白色は引かれない＝残る）
    # rate=1 なら eps_colored + eps_parallel = pred_noise (全除去)
    eps_used = eps_colored + (eps_parallel * removal_rate)
    
    return eps_used

# ==========================================
#  Added: Colored Noise Only Extraction
# ==========================================
def extract_colored_noise(pred_noise, known_noise):
    """
    予測ノイズから「白色成分（既知ノイズと平行）」を除去し、
    「有色成分（直交成分）」のみを抽出する。
    """
    n_b = pred_noise.shape[0]
    pred_flat = pred_noise.reshape(n_b, -1)
    known_flat = known_noise.reshape(n_b, -1)

    # 1. 平行成分の係数計算
    dot_pk = torch.sum(pred_flat * known_flat, dim=1, keepdim=True)
    dot_kk = torch.sum(known_flat * known_flat, dim=1, keepdim=True)
    dot_kk = torch.clamp(dot_kk, min=1e-8)
    
    alpha = dot_pk / dot_kk
    alpha_view = alpha.view(n_b, 1, 1, 1)
    
    # 2. 平行成分（白色成分）
    eps_parallel = alpha_view * known_noise
    
    # 3. 直交成分（有色成分）のみを取り出す
    # これをDDIMのノイズとして使うと、画像から有色成分だけが引かれる
    eps_colored = pred_noise - eps_parallel
    
    return eps_colored

# ==========================================
#  Added: Slerp (Spherical Linear Interpolation)
# ==========================================

def slerp_tensor(val, low, high):
    """
    Slerp for tensors.
    val: float or tensor, interpolation factor (0.0 = low (pred), 1.0 = high (known))
    low: tensor, starting vector (e.g., predicted noise)
    high: tensor, target vector (e.g., known noise)
    """
    dims = low.shape
    # Flatten for calculation
    low_flat = low.reshape(low.shape[0], -1)
    high_flat = high.reshape(high.shape[0], -1)

    # Normalize
    low_norm = low_flat / torch.norm(low_flat, dim=1, keepdim=True)
    high_norm = high_flat / torch.norm(high_flat, dim=1, keepdim=True)

    # Dot product (cos omega)
    dot = (low_norm * high_norm).sum(1)
    dot = torch.clamp(dot, -1.0, 1.0)
    
    omega = torch.acos(dot)
    so = torch.sin(omega)

    # Avoid zero division
    mask = so > 1e-6
    
    s1 = torch.zeros_like(dot)
    s2 = torch.zeros_like(dot)
    
    if isinstance(val, float):
        val = torch.tensor(val, device=low.device)
    
    s1[mask] = torch.sin((1.0 - val) * omega[mask]) / so[mask]
    s2[mask] = torch.sin(val * omega[mask]) / so[mask]
    
    # Linear interpolation fallback
    s1[~mask] = 1.0 - val
    s2[~mask] = val
    
    s1 = s1.view(-1, 1)
    s2 = s2.view(-1, 1)
    
    res = s1 * low_flat + s2 * high_flat
    return res.reshape(dims)
# def orthogonal_guidance(pred_noise, known_noise, ortho_scale=0.0):
#     """
#     予測ノイズを既知ノイズに対して射影分解し、直交成分をスケーリングする関数。
    
#     Args:
#         pred_noise (Tensor): U-Netが予測したノイズ (epsilon_theta)
#         known_noise (Tensor): 送信側で付加された既知ノイズ (epsilon_known)
#         ortho_scale (float): 直交成分を「引く（抑制する）」割合。
#                              0.0 = 何もしない（元の予測のまま）
#                              1.0 = 直交成分を完全に除去（既知ノイズ成分のみ残す）
#                              0.5 = 直交成分を半分にする
#     """
#     # バッチごとのドット積計算のためにフラット化 (B, C*H*W)
#     n_b = pred_noise.shape[0]
#     pred_flat = pred_noise.reshape(n_b, -1)
#     known_flat = known_noise.reshape(n_b, -1)

#     # 既知ノイズ方向への射影係数 alpha = (pred . known) / (known . known)
#     dot_pk = torch.sum(pred_flat * known_flat, dim=1, keepdim=True)
#     dot_kk = torch.sum(known_flat * known_flat, dim=1, keepdim=True)
    
#     # ゼロ除算回避
#     dot_kk = torch.clamp(dot_kk, min=1e-8)
    
#     alpha = dot_pk / dot_kk
    
#     # 平行成分 (Parallel component): 既知ノイズと相関する成分
#     # alpha は (B, 1) なので broadcast して元の shape に戻す
#     alpha_view = alpha.view(n_b, 1, 1, 1)
#     eps_parallel = alpha_view * known_noise
    
#     # 直交成分 (Orthogonal component): 予測ノイズのうち既知ノイズで説明できない部分
#     # ここに「有色雑音」や「モデルの幻覚」が含まれる可能性が高い
#     eps_orthogonal = pred_noise - eps_parallel
    
#     # 直交成分を指定された比率で減算（抑制）
#     # ortho_scale が大きいほど、予測ノイズは既知ノイズの形に強制される
#     eps_modified = eps_parallel + (1.0 - ortho_scale) * eps_orthogonal
    
#     return eps_modified

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def forward_diffusion(self,
               S, #ddim_num_steps 200
               batch_size,
               conditioning=None,
               x = None,
               eta=0.,
               verbose=True,
               timestep = 0,
               epsilon = None, 
               **kwargs
               ):
        assert(x != None)
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)

        alpha_t_bar = self.alphas_cumprod[timestep]

        # 1. torch.randn_like を使い、x と同じ形状・デバイスでガウスノイズを生成
        if epsilon == None:
            epsilon = torch.randn_like(x)
        assert(epsilon.shape == x.shape)
        # 2. 係数を変形
        sqrt_alpha_t_bar = torch.sqrt(alpha_t_bar).view(1, 1, 1, 1)
        if timestep == 0:
            #print(f"ddim.py ===========   sqrt_alpha_t_bar = {sqrt_alpha_t_bar}")
            sqrt_alpha_t_bar = torch.full_like(sqrt_alpha_t_bar, 1.0)
            return x
        sqrt_one_minus_alpha_t_bar = torch.sqrt(1.0 - alpha_t_bar).view(1, 1, 1, 1)

        # 3. ノイズを加える
        return sqrt_alpha_t_bar * x + sqrt_one_minus_alpha_t_bar * epsilon


    @torch.no_grad()
    def sample(self,
               S, #ddim_num_steps
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                #impainting用
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        # c: cond
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0



    @torch.no_grad
    def my_ddim_sampling(self,
               S, #ddim_num_steps 200
               batch_size,
               shape,
               noise_sigma,
               noise_sigma_predict,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               intermediate_path = None,
               intermediate_skip = 1,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               snr = None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        print(f"ddim.py alphas_cumprod = {self.alphas_cumprod.shape}") #1000
        C, H, W = shape
        size = (batch_size, C, H, W)

        device = self.model.betas.device
        alpha_bar_u = 1/(1 + noise_sigma_predict)
        print(f"ddim.py, alpha_bar_u = {alpha_bar_u}")
        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        start_timesteps *= 1 # ここでタイムステップをいじくる
        #torch.clamp(start_timesteps, 0, S)

        # alphas
        #indiceからタイムステップ
        print(f"ddim.py start_timesteps after clamp S = {S} : {start_timesteps}")
        results = []
        img = x_T.to(device)
        maxind = start_timesteps.max().item()
        time_range = reversed(range(0, maxind))
        iterator = tqdm(time_range, desc='My Sampling')
        for i, step in enumerate(iterator):
            # 現在のステップが、バッチ内の各画像の開始タイムステップ以下であるかどうかのマスクを作成
            # これにより、まだサンプリングが始まっていない画像をスキップできる
            active_mask = (start_timesteps >= step).view(-1, 1, 1, 1)

            # ts (タイムステップ) はバッチ全体分を用意
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # indexはスケジュール配列を参照するためのもの。現在のループ回数から計算
            # self.ddim_timestepsはmake_scheduleで作成される長さSの配列
            # stepに対応するindexを探す
            time_idx_tensor = torch.where(torch.from_numpy(self.ddim_timesteps).to(device) == step)[0]
            if time_idx_tensor.numel() == 0:
                # もし現在のstepがスケジュールになければスキップ (補間なども可能)
                continue
            index = time_idx_tensor.item()


            # p_sample_ddimをバッチ全体に対して一度だけ呼び出す
            outs = self.p_sample_ddim(img, conditioning, ts, index, # 修正: indexを追加
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning)
            img_prev, pred_x0 = outs

            # active_maskがTrueの画像だけを、計算結果で更新する
            img = torch.where(active_mask, img_prev, img)
            #print(f"step = {step}, index = {index}")

            if intermediate_path != None and index % intermediate_skip == 0:
                #TODO 途中結果を保存

                decoded_img = self.model.decode_first_stage(pred_x0)

                # VAEのデコーダ出力は [-1, 1] の範囲なので、[0, 1] にスケーリングする
                # decoded_img = (decoded_img + 1.0) / 2.0
                # decoded_img = torch.clamp(decoded_img, min=0.0, max=1.0)

                # バッチ内の各画像を個別に保存
                batch_size = decoded_img.shape[0]
                for i in range(batch_size):
                    # ファイル名をステップ番号と画像インデックスで一意に決定
                    # 例: /path/to/intermediate/step_0180_img_00.png
                    target_dir = os.path.join(intermediate_path, str(snr), str(i))

                    # 2. ディレクトリが存在しなければ再帰的に作成 (exist_ok=True)
                    os.makedirs(target_dir, exist_ok=True)

                    # 3. ファイル名を決定 (ステップ番号)
                    #    例: step_0436.png
                    file_name = f"step_{step:04d}.png"

                    # 4. 最終的なファイルパス
                    img_path = os.path.join(target_dir, file_name)

                    # torchvision.utils.save_image を使って保存
                    vutil.save_image(decoded_img[i], img_path)
                #print(f"save figure")
        return img

    @torch.no_grad
    def my_ddim_sampling_knownnoise(self,
               S, #ddim_num_steps 200
               batch_size,
               shape,
               noise_sigma,
               noise_sigma_predict,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               intermediate_path = None,
               intermediate_skip = 1,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               snr = None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        print(f"ddim.py alphas_cumprod = {self.alphas_cumprod.shape}") #1000
        C, H, W = shape
        size = (batch_size, C, H, W)

        device = self.model.betas.device
        alpha_bar_u = 1/(1 + noise_sigma_predict)
        print(f"ddim.py, alpha_bar_u = {alpha_bar_u}")
        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        start_timesteps *= 1 # ここでタイムステップをいじくる
        #torch.clamp(start_timesteps, 0, S)

        # alphas
        #indiceからタイムステップ
        print(f"ddim.py start_timesteps after clamp S = {S} : {start_timesteps}")
        results = []
        img = x_T.to(device)
        maxind = start_timesteps.max().item()
        time_range = reversed(range(0, maxind))
        iterator = tqdm(time_range, desc='My Sampling')
        epsilon_known = torch.rand_like(img) # 既知ノイズ用のノイズを用意
        for i, step in enumerate(iterator):
            # 現在のステップが、バッチ内の各画像の開始タイムステップ以下であるかどうかのマスクを作成
            # これにより、まだサンプリングが始まっていない画像をスキップできる
            active_mask = (start_timesteps >= step).view(-1, 1, 1, 1)

            # ts (タイムステップ) はバッチ全体分を用意
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # indexはスケジュール配列を参照するためのもの。現在のループ回数から計算
            # self.ddim_timestepsはmake_scheduleで作成される長さSの配列
            # stepに対応するindexを探す
            time_idx_tensor = torch.where(torch.from_numpy(self.ddim_timesteps).to(device) == step)[0]
            if time_idx_tensor.numel() == 0:
                # もし現在のstepがスケジュールになければスキップ (補間なども可能)
                continue
            index = time_idx_tensor.item()
            added_timestep = max(0, 300 - step)# stepが増加すると減少
            #
            self.forward_diffusion(S, batch_size, epsilon=epsilon_known, timestep=added_timestep)
            # p_sample_ddimをバッチ全体に対して一度だけ呼び出す
            # A. ノイズ予測 (U-Net)
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(img, ts+added_timestep, conditioning)
            else:
                x_in = torch.cat([img] * 2)
                t_in = torch.cat([ts] * 2)
                c_in = torch.cat([unconditional_conditioning, conditioning])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            # e_tのepsilon_known以外の成分を除外
            e_t = extract_colored_noise_smooth(e_t, epsilon_known, 0.1)
            alphas = self.ddim_alphas
            alphas_prev = self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
            sigmas = self.ddim_sigmas

            a_t = torch.full((batch_size, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((batch_size, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((batch_size, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((batch_size, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(img.shape, device, False) * eta
            img_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            # img_prevからforward processのノイズを消去
            alpha_t_bar  = self.alphas_cumprod[added_timestep]
            sqrt_alpha_t_bar = torch.sqrt(alpha_t_bar).view(1, 1, 1, 1)
            sqrt_one_minus_alpha_t_bar = torch.sqrt(1.0 - alpha_t_bar).view(1, 1, 1, 1)
            img_prev = (img_prev - sqrt_one_minus_alpha_t_bar * epsilon_known ) / sqrt_alpha_t_bar
            

            # active_maskがTrueの画像だけを、計算結果で更新する
            img = torch.where(active_mask, img_prev, img)
            #print(f"step = {step}, index = {index}")

            
        return img

    
    

    @torch.no_grad
    def jointdiffusion_ddim_sampling(self,
               S, #ddim_num_steps 200
               batch_size,
               shape,

               noise_sigma_predict,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               intermediate_path = None,
               intermediate_skip = 1,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               snr = None,
               added_timestep = 0,
               h = torch.tensor(1 + 0j, dtype=torch.complex64),
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        print(f"ddim.py alphas_cumprod = {self.alphas_cumprod.shape}") #1000
        C, H, W = shape
        size = (batch_size, C, H, W)
        alpha_bar_t = self.alphas_cumprod[added_timestep]
        if added_timestep == 0:
            alpha_bar_t = torch.full_like(alpha_bar_t, 1.0)
            print(f"ddim.py alpha_bar_t = {alpha_bar_t}")
        device = self.model.betas.device
        term1 = (torch.sqrt(1 - alpha_bar_t) + torch.sqrt(noise_sigma_predict) / h) / torch.sqrt(alpha_bar_t)
        term1 = term1.abs()
        alpha_bar_u = 1/(term1 * term1 + 1)

        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        start_timesteps *= 1 # ここでタイムステップをいじくる
        #torch.clamp(start_timesteps, 0, S)

        # alphas
        #indiceからタイムステップ
        print(f"ddim.py start_timesteps after clamp S = {S} : {start_timesteps}")
        results = []
        img = x_T.to(device)
        maxind = start_timesteps.max().item()
        time_range = reversed(range(0, maxind))
        iterator = tqdm(time_range, desc='My Sampling')
        for i, step in enumerate(iterator):
            # 現在のステップが、バッチ内の各画像の開始タイムステップ以下であるかどうかのマスクを作成
            # これにより、まだサンプリングが始まっていない画像をスキップできる
            active_mask = (start_timesteps >= step).view(-1, 1, 1, 1)

            # ts (タイムステップ) はバッチ全体分を用意
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # indexはスケジュール配列を参照するためのもの。現在のループ回数から計算
            # self.ddim_timestepsはmake_scheduleで作成される長さSの配列
            # stepに対応するindexを探す
            time_idx_tensor = torch.where(torch.from_numpy(self.ddim_timesteps).to(device) == step)[0]
            if time_idx_tensor.numel() == 0:
                # もし現在のstepがスケジュールになければスキップ (補間なども可能)
                continue
            index = time_idx_tensor.item()


            # p_sample_ddimをバッチ全体に対して一度だけ呼び出す
            outs = self.p_sample_ddim(img, conditioning, ts, index, # 修正: indexを追加
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning)
            img_prev, pred_x0 = outs

            # active_maskがTrueの画像だけを、計算結果で更新する
            img = torch.where(active_mask, img_prev, img)
            #print(f"step = {step}, index = {index}")

            if intermediate_path != None and index % intermediate_skip == 0:
                #TODO 途中結果を保存

                decoded_img = self.model.decode_first_stage(pred_x0)

                # VAEのデコーダ出力は [-1, 1] の範囲なので、[0, 1] にスケーリングする
                # decoded_img = (decoded_img + 1.0) / 2.0
                # decoded_img = torch.clamp(decoded_img, min=0.0, max=1.0)

                # バッチ内の各画像を個別に保存
                batch_size = decoded_img.shape[0]
                for i in range(batch_size):
                    # ファイル名をステップ番号と画像インデックスで一意に決定
                    # 例: /path/to/intermediate/step_0180_img_00.png
                    target_dir = os.path.join(intermediate_path, str(snr), str(i))

                    # 2. ディレクトリが存在しなければ再帰的に作成 (exist_ok=True)
                    os.makedirs(target_dir, exist_ok=True)

                    # 3. ファイル名を決定 (ステップ番号)
                    #    例: step_0436.png
                    file_name = f"step_{step:04d}.png"

                    # 4. 最終的なファイルパス
                    img_path = os.path.join(target_dir, file_name)

                    # torchvision.utils.save_image を使って保存
                    vutil.save_image(decoded_img[i], img_path)
                #print(f"save figure")
        return img

    @torch.no_grad
    def onestep_sampling(self,
               S, #ddim_num_steps 200
               batch_size,
               shape,
               noise_sigma,
               noise_sigma_predict,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,

               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               snr = None,
               starttimestep = None,
               breakstep = 1,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        print(f"ddim.py alphas_cumprod = {self.alphas_cumprod.shape}") #1000
        C, H, W = shape
        size = (batch_size, C, H, W)

        device = self.model.betas.device
        alpha_bar_u = 1/(1 + noise_sigma_predict)
        print(f"ddim.py, alpha_bar_u = {alpha_bar_u}")
        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        start_timesteps *= 1 # ここでタイムステップをいじくる
        #torch.clamp(start_timesteps, 0, S)
        if starttimestep != None:
            start_timesteps = torch.full_like(start_timesteps, starttimestep)
        # alphas
        #indiceからタイムステップ
        print(f"ddim.py start_timesteps after clamp S = {S} : {start_timesteps}")
        results = []
        img = x_T.to(device)
        maxind = start_timesteps.max().item()
        time_range = reversed(range(0, maxind))
        iterator = tqdm(time_range, desc='My Sampling')
        cnt = 0
        for i, step in enumerate(iterator):

            # 現在のステップが、バッチ内の各画像の開始タイムステップ以下であるかどうかのマスクを作成
            # これにより、まだサンプリングが始まっていない画像をスキップできる
            active_mask = (start_timesteps >= step).view(-1, 1, 1, 1)

            # ts (タイムステップ) はバッチ全体分を用意
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # indexはスケジュール配列を参照するためのもの。現在のループ回数から計算
            # self.ddim_timestepsはmake_scheduleで作成される長さSの配列
            # stepに対応するindexを探す
            time_idx_tensor = torch.where(torch.from_numpy(self.ddim_timesteps).to(device) == step)[0]
            if time_idx_tensor.numel() == 0:
                # もし現在のstepがスケジュールになければスキップ (補間なども可能)
                continue
            index = time_idx_tensor.item()


            # p_sample_ddimをバッチ全体に対して一度だけ呼び出す
            outs = self.p_sample_ddim(img, conditioning, ts, index, # 修正: indexを追加
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning)
            img_prev, pred_x0 = outs
            cnt += 1
            # active_maskがTrueの画像だけを、計算結果で更新する
            img = torch.where(active_mask, img_prev, img)
            print(f"ddim.py , onestep sampling complete ,step = {step}, index = {index}, cnt = {cnt}, breakstep = {breakstep}")
            if cnt == breakstep:
                break

        return img

    @torch.no_grad
    def MIMO_decide_starttimestep_ddim_sampling(self,
               S, #ddim_num_steps 200
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               intermediate_path = None,
               intermediate_skip = 1,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               snr = None,
               starttimestep = 200,
               noise_variance = None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        print(f"ddim.py alphas_cumprod = {self.alphas_cumprod.shape}") #1000
        C, H, W = shape
        size = (batch_size, C, H, W)

        device = self.model.betas.device

        #torch.clamp(start_timesteps, 0, S)
        if noise_variance == None:
            noise_variance = 1.0
        alpha_bar_u = 1/(1 + noise_variance)
        print(f"ddim.py, alpha_bar_u = {alpha_bar_u}")
        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        if starttimestep != None and starttimestep>=0:
            start_timesteps = torch.zeros(batch_size, dtype=torch.long).to(device)
            start_timesteps = torch.full_like(start_timesteps, starttimestep).to(device)

        # alphas
        #indiceからタイムステップ
        print(f"ddim.py start_timesteps after clamp S = {S} : {start_timesteps}")
        results = []
        img = x_T.to(device)
        maxind = start_timesteps.max().item()
        time_range = reversed(range(0, maxind))
        iterator = tqdm(time_range, desc='My Sampling')
        for i, step in enumerate(iterator):
            # 現在のステップが、バッチ内の各画像の開始タイムステップ以下であるかどうかのマスクを作成
            # これにより、まだサンプリングが始まっていない画像をスキップできる
            active_mask = (start_timesteps >= step).view(-1, 1, 1, 1).to(device)

            # ts (タイムステップ) はバッチ全体分を用意
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # indexはスケジュール配列を参照するためのもの。現在のループ回数から計算
            # self.ddim_timestepsはmake_scheduleで作成される長さSの配列
            # stepに対応するindexを探す
            time_idx_tensor = torch.where(torch.from_numpy(self.ddim_timesteps).to(device) == step)[0]
            if time_idx_tensor.numel() == 0:
                # もし現在のstepがスケジュールになければスキップ (補間なども可能)
                continue
            index = time_idx_tensor.item()


            # p_sample_ddimをバッチ全体に対して一度だけ呼び出す
            outs = self.p_sample_ddim(img, conditioning, ts, index, # 修正: indexを追加
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning)
            img_prev, pred_x0 = outs

            # active_maskがTrueの画像だけを、計算結果で更新する
            img = torch.where(active_mask, img_prev, img)
            #print(f"step = {step}, index = {index}")

            if intermediate_path != None and index % intermediate_skip == 0:
                #TODO 途中結果を保存

                decoded_img = self.model.decode_first_stage(pred_x0)

                # VAEのデコーダ出力は [-1, 1] の範囲なので、[0, 1] にスケーリングする
                # decoded_img = (decoded_img + 1.0) / 2.0
                # decoded_img = torch.clamp(decoded_img, min=0.0, max=1.0)

                # バッチ内の各画像を個別に保存
                batch_size = decoded_img.shape[0]
                for i in range(batch_size):
                    # ファイル名をステップ番号と画像インデックスで一意に決定
                    # 例: /path/to/intermediate/step_0180_img_00.png
                    target_dir = os.path.join(intermediate_path, str(snr), str(i), str(starttimestep))

                    # 2. ディレクトリが存在しなければ再帰的に作成 (exist_ok=True)
                    os.makedirs(target_dir, exist_ok=True)

                    # 3. ファイル名を決定 (ステップ番号)
                    #    例: step_0436.png
                    file_name = f"step_{step:04d}.png"

                    # 4. 最終的なファイルパス
                    img_path = os.path.join(target_dir, file_name)

                    # torchvision.utils.save_image を使って保存
                    vutil.save_image(decoded_img[i], img_path)
                #print(f"save figure")

        return img


    @torch.no_grad
    def decide_starttimestep_ddim_sampling(self,
               S, #ddim_num_steps 200
               batch_size,
               shape,
               noise_sigma,
               noise_sigma_predict,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               intermediate_path = None,
               intermediate_skip = 1,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               snr = None,
               starttimestep = None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        print(f"ddim.py alphas_cumprod = {self.alphas_cumprod.shape}") #1000
        C, H, W = shape
        size = (batch_size, C, H, W)

        device = self.model.betas.device
        alpha_bar_u = 1/(1 + noise_sigma_predict)
        print(f"ddim.py, alpha_bar_u = {alpha_bar_u}")
        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        start_timesteps *= 1 # ここでタイムステップをいじくる
        #torch.clamp(start_timesteps, 0, S)
        if starttimestep != None:
            start_timesteps = torch.full_like(start_timesteps, starttimestep)

        # alphas
        #indiceからタイムステップ
        print(f"ddim.py start_timesteps after clamp S = {S} : {start_timesteps}")
        results = []
        img = x_T.to(device)
        maxind = start_timesteps.max().item()
        time_range = reversed(range(0, maxind))
        iterator = tqdm(time_range, desc='My Sampling')
        for i, step in enumerate(iterator):
            # 現在のステップが、バッチ内の各画像の開始タイムステップ以下であるかどうかのマスクを作成
            # これにより、まだサンプリングが始まっていない画像をスキップできる
            active_mask = (start_timesteps >= step).view(-1, 1, 1, 1)

            # ts (タイムステップ) はバッチ全体分を用意
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # indexはスケジュール配列を参照するためのもの。現在のループ回数から計算
            # self.ddim_timestepsはmake_scheduleで作成される長さSの配列
            # stepに対応するindexを探す
            time_idx_tensor = torch.where(torch.from_numpy(self.ddim_timesteps).to(device) == step)[0]
            if time_idx_tensor.numel() == 0:
                # もし現在のstepがスケジュールになければスキップ (補間なども可能)
                continue
            index = time_idx_tensor.item()


            # p_sample_ddimをバッチ全体に対して一度だけ呼び出す
            outs = self.p_sample_ddim(img, conditioning, ts, index, # 修正: indexを追加
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning)
            img_prev, pred_x0 = outs

            # active_maskがTrueの画像だけを、計算結果で更新する
            img = torch.where(active_mask, img_prev, img)
            #print(f"step = {step}, index = {index}")

            if intermediate_path != None and index % intermediate_skip == 0:
                #TODO 途中結果を保存

                decoded_img = self.model.decode_first_stage(pred_x0)

                # VAEのデコーダ出力は [-1, 1] の範囲なので、[0, 1] にスケーリングする
                # decoded_img = (decoded_img + 1.0) / 2.0
                # decoded_img = torch.clamp(decoded_img, min=0.0, max=1.0)

                # バッチ内の各画像を個別に保存
                batch_size = decoded_img.shape[0]
                for i in range(batch_size):
                    # ファイル名をステップ番号と画像インデックスで一意に決定
                    # 例: /path/to/intermediate/step_0180_img_00.png
                    target_dir = os.path.join(intermediate_path, str(snr), str(i), str(starttimestep))

                    # 2. ディレクトリが存在しなければ再帰的に作成 (exist_ok=True)
                    os.makedirs(target_dir, exist_ok=True)

                    # 3. ファイル名を決定 (ステップ番号)
                    #    例: step_0436.png
                    file_name = f"step_{step:04d}.png"

                    # 4. 最終的なファイルパス
                    img_path = os.path.join(target_dir, file_name)

                    # torchvision.utils.save_image を使って保存
                    vutil.save_image(decoded_img[i], img_path)
                #print(f"save figure")

        return img

    @torch.no_grad
    def observe_ddim(self,
               S, #ddim_num_steps 200
               batch_size,
               shape,
               noise_sigma,
               noise_sigma_predict,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               intermediate_path = None,
               intermediate_skip = 1,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               snr = None,
               starttimestep = None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        print(f"ddim.py alphas_cumprod = {self.alphas_cumprod.shape}") #1000
        C, H, W = shape
        size = (batch_size, C, H, W)

        device = self.model.betas.device
        alpha_bar_u = 1/(1 + noise_sigma_predict)
        print(f"ddim.py, alpha_bar_u = {alpha_bar_u}")
        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        start_timesteps *= 1 # ここでタイムステップをいじくる
        #torch.clamp(start_timesteps, 0, S)
        if starttimestep != None:
            start_timesteps = torch.full_like(start_timesteps, starttimestep)

        # alphas
        #indiceからタイムステップ
        print(f"ddim.py start_timesteps after clamp S = {S} : {start_timesteps}")
        results = []
        img = x_T.to(device)
        maxind = start_timesteps.max().item()
        time_range = reversed(range(0, maxind))
        iterator = tqdm(time_range, desc='My Sampling')
        past_img = {}
        for i, step in enumerate(iterator):
            # 現在のステップが、バッチ内の各画像の開始タイムステップ以下であるかどうかのマスクを作成
            # これにより、まだサンプリングが始まっていない画像をスキップできる
            active_mask = (start_timesteps >= step).view(-1, 1, 1, 1)

            # ts (タイムステップ) はバッチ全体分を用意
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # indexはスケジュール配列を参照するためのもの。現在のループ回数から計算
            # self.ddim_timestepsはmake_scheduleで作成される長さSの配列
            # stepに対応するindexを探す
            time_idx_tensor = torch.where(torch.from_numpy(self.ddim_timesteps).to(device) == step)[0]
            if time_idx_tensor.numel() == 0:
                # もし現在のstepがスケジュールになければスキップ (補間なども可能)
                continue
            index = time_idx_tensor.item()


            # p_sample_ddimをバッチ全体に対して一度だけ呼び出す
            outs = self.p_sample_ddim(img, conditioning, ts, index, # 修正: indexを追加
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning)
            img_prev, pred_x0 = outs

            # active_maskがTrueの画像だけを、計算結果で更新する
            img = torch.where(active_mask, img_prev, img)
            #print(f"step = {step}, index = {index}")
            past_img[(step, index)] = img


        return past_img
    
    # ----------------------------------------------------------------
    # 修正: Slerpの代わりに Orthogonal Guidance を使用するように変更
    # ----------------------------------------------------------------
    @torch.no_grad()
    def known_noise_guided_ddim_sampling_orthogonal(self,
               S,
               batch_size,
               shape,
               noise_variance, 
               conditioning=None,
               eta=0.,
               x_T=None,
               verbose=True,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               added_timestep=0,
               eps_known=None,
               ortho_guidance_scale=0.5, # 変数名を変更: 直交成分を削減する強さ
               **kwargs
               ):
        
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        device = self.model.betas.device
        
        if not torch.is_tensor(noise_variance):
            noise_variance = torch.tensor(noise_variance, device=device)

        # 開始タイムステップ計算
        alpha_bar_t = self.alphas_cumprod[added_timestep]
        current_noise_level = torch.sqrt(1 - alpha_bar_t) + torch.sqrt(noise_variance)
        alpha_bar_u = 1 / (current_noise_level**2 + 1)
        
        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        start_timesteps = torch.max(start_timesteps, torch.tensor(added_timestep, device=device))
        start_timesteps = start_timesteps.to(device)
        start_timesteps = torch.clamp(start_timesteps, min=0, max=999)

        if verbose:
            print(f"Computed start_timesteps (mean): {start_timesteps.float().mean().item():.2f}")

        maxind = start_timesteps.max().item()
        iterator = tqdm(reversed(range(0, maxind + 1)), desc='Orthogonal Guided DDIM', total=maxind+1)

        img = x_T.clone()
        ddim_timesteps_tensor = torch.from_numpy(self.ddim_timesteps).to(device)

        for i, step in enumerate(iterator):
            # スケジュール判定
            time_idx_tensor = torch.where(ddim_timesteps_tensor == step)[0]
            if time_idx_tensor.numel() == 0: continue
            index = time_idx_tensor.item()

            active_mask = (start_timesteps >= step).view(-1, 1, 1, 1)
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
            
            # ノイズ予測
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                # こちらを使用
                e_t = self.model.apply_model(img, ts, conditioning)
            else:
                x_in = torch.cat([img] * 2)
                t_in = torch.cat([ts] * 2)
                c_in = torch.cat([unconditional_conditioning, conditioning])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            # -----------------------------------------------------------
            # 修正箇所: 既知ノイズに対する直交成分の抑制 (Orthogonal Subtraction)
            # -----------------------------------------------------------
            if eps_known is not None:
                # 徐々に比率を変えるなどのスケジュールもここで可能
                # 例: ノイズが大きい初期段階(stepが大きい時)ほど scale を強くする等
                
                current_scale = ortho_guidance_scale
                
                # 直交成分を抑制して e_t を更新
                e_t = orthogonal_guidance(e_t, eps_known, ortho_scale=current_scale)

            # DDIM更新ステップ
            alphas = self.ddim_alphas
            alphas_prev = self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
            sigmas = self.ddim_sigmas

            a_t = torch.full((batch_size, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((batch_size, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((batch_size, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((batch_size, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(img.shape, device, False) * eta
            img_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

            img = torch.where(active_mask, img_prev, img)

        return img
    @torch.no_grad()
    def orthogonal_projection_guided_ddim_sampling(self,
               S,
               batch_size,
               shape,
               noise_variance, 
               conditioning=None,
               eta=0.,
               x_T=None,
               verbose=True,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               added_timestep=0,
               eps_known=None,
               ortho_guidance_scale=0.2, # 直交成分を削減する強さ (0.0~1.0)
               **kwargs
               ):
        
        # 1. スケジュール作成
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        device = self.model.betas.device
        
        if not torch.is_tensor(noise_variance):
            noise_variance = torch.tensor(noise_variance, device=device)

        # 2. 開始ステップの計算
        alpha_bar_t = self.alphas_cumprod[added_timestep]
        current_noise_level = torch.sqrt(1 - alpha_bar_t) + torch.sqrt(noise_variance)
        alpha_bar_u = 1 / (current_noise_level**2 + 1)
        
        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        start_timesteps = torch.max(start_timesteps, torch.tensor(added_timestep, device=device))
        start_timesteps = start_timesteps.to(device)
        start_timesteps = torch.clamp(start_timesteps, min=0, max=999)

        if verbose:
            print(f"Computed start_timesteps (mean): {start_timesteps.float().mean().item():.2f}")

        # 3. サンプリングループ
        maxind = start_timesteps.max().item()
        iterator = tqdm(reversed(range(0, maxind + 1)), desc='Orthogonal Guided DDIM', total=maxind+1)

        img = x_T.clone()
        ddim_timesteps_tensor = torch.from_numpy(self.ddim_timesteps).to(device)

        for i, step in enumerate(iterator):
            # インデックス取得
            time_idx_tensor = torch.where(ddim_timesteps_tensor == step)[0]
            if time_idx_tensor.numel() == 0: continue
            index = time_idx_tensor.item()

            active_mask = (start_timesteps >= step).view(-1, 1, 1, 1)
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
            
            # A. ノイズ予測 (U-Net)
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(img, ts, conditioning)
            else:
                x_in = torch.cat([img] * 2)
                t_in = torch.cat([ts] * 2)
                c_in = torch.cat([unconditional_conditioning, conditioning])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            # -----------------------------------------------------------
            # B. 直交成分の抑制 (Orthogonal Subtraction)
            # -----------------------------------------------------------
            if eps_known is not None:
                # ユーザーの意図通り、ステップごとに少しずつ直交成分を引く
                e_t = orthogonal_guidance(e_t, eps_known, current_scale=ortho_guidance_scale)

            # C. 状態更新 (Standard DDIM Step)
            alphas = self.ddim_alphas
            alphas_prev = self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
            sigmas = self.ddim_sigmas

            a_t = torch.full((batch_size, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((batch_size, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((batch_size, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((batch_size, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(img.shape, device, False) * eta
            img_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

            img = torch.where(active_mask, img_prev, img)

        return img
    
    @torch.no_grad()
    def projected_noise_guided_ddim_sampling(self,
               S,
               batch_size,
               shape,
               noise_variance, 
               conditioning=None,
               eta=0.,
               x_T=None,
               verbose=True,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               added_timestep=None, # ★重要: ここで指定したステップ(例:400)から強制スタート
               eps_known=None,      # ★重要: 注入した既知ノイズの真値
               **kwargs
               ):
        """
        Fixed-Timestep Injection & Projected Guidance Sampling
        
        有色雑音対策として、強力な既知白色雑音(eps_known)を注入して
        固定のタイムステップ(added_timestep)までジャンプした状態から開始し、
        予測ノイズ内の既知成分を真値に置換しながらデノイズを行う。
        """
        
        # 1. スケジュール作成
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        device = self.model.betas.device
        
        # 2. 開始ステップの確定 (Fixed-Timestep Logic)
        if added_timestep is None:
            # 指定がない場合は安全策として最大ステップを使うが、基本は指定すること
            added_timestep = self.ddim_timesteps[-1]
            
        # DDIMのタイムステップ配列の中で、added_timestep以下で最大のものを探す
        # 例: ddim_timesteps=[0, 5, ..., 400, 405...], added_timestep=400 -> index corresponding to 400
        ddim_timesteps_tensor = torch.from_numpy(self.ddim_timesteps).to(device)
        
        # added_timestep を超えない最大のステップのインデックスを取得
        start_idx = (ddim_timesteps_tensor <= added_timestep).nonzero(as_tuple=True)[0][-1].item()
        actual_start_step = self.ddim_timesteps[start_idx]
        
        if verbose:
            print(f"Projected Guidance: Fixed Start at t={actual_start_step} (idx={start_idx})")

        # 3. イテレータの作成 (actual_start_step から 0 まで遡る)
        # self.ddim_timesteps[:start_idx+1] は [0, ..., actual_start_step]
        # これを flip して [actual_start_step, ..., 0] にする
        subset_timesteps = np.flip(self.ddim_timesteps[:start_idx+1])
        iterator = tqdm(subset_timesteps, desc='Projected Noise Guidance', total=len(subset_timesteps))

        # 初期画像
        img = x_T.clone()

        # 4. サンプリングループ
        for i, step in enumerate(iterator):
            # 現在のステップに対応するDDIMパラメータのインデックスを取得
            index = (ddim_timesteps_tensor == step).nonzero(as_tuple=True)[0].item()
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
            
            # --- A. ノイズ予測 (U-Net) ---
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(img, ts, conditioning)
            else:
                x_in = torch.cat([img] * 2)
                t_in = torch.cat([ts] * 2)
                c_in = torch.cat([unconditional_conditioning, conditioning])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            # --- B. 提案手法: 既知ノイズ成分の射影と置換 (Projection & Replacement) ---
            if eps_known is not None:
                # バッチ計算用に変形
                n_b = e_t.shape[0]
                pred_flat = e_t.reshape(n_b, -1)
                known_flat = eps_known.reshape(n_b, -1)

                # 1. 射影係数 alpha = (pred . known) / (known . known)
                dot_pk = torch.sum(pred_flat * known_flat, dim=1, keepdim=True)
                dot_kk = torch.sum(known_flat * known_flat, dim=1, keepdim=True)
                dot_kk = torch.clamp(dot_kk, min=1e-8)
                
                alpha = dot_pk / dot_kk
                alpha_view = alpha.view(n_b, 1, 1, 1)
                
                # 2. 平行成分（予測された既知ノイズ成分）
                eps_parallel = alpha_view * eps_known
                
                # 3. 直交成分（有色ノイズや信号成分など、モデルが純粋に推定したもの）
                eps_orthogonal = e_t - eps_parallel
                
                # 4. 置換
                # 予測された平行成分を捨て、真の既知ノイズ eps_known に置き換える
                # これにより、注入した大量のノイズが数学的にキャンセルされる
                e_t = eps_orthogonal + eps_known

            # --- C. 状態更新 (Standard DDIM Step) ---
            alphas = self.ddim_alphas
            alphas_prev = self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
            sigmas = self.ddim_sigmas

            # パラメータ取得
            a_t = torch.full((batch_size, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((batch_size, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((batch_size, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((batch_size, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            # x0の予測
            pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()
            
            # xt方向へのポインティング
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            
            # ランダムノイズ項 (eta=0なら決定論的)
            noise = sigma_t * noise_like(img.shape, device, False) * eta
            
            # 次のステップの画像
            img_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

            img = img_prev

        return img

    @torch.no_grad()
    def smooth_transition_ddim_sampling(self,
               S,
               batch_size,
               shape,
               noise_variance, 
               conditioning=None,
               eta=0.,
               x_T=None,
               verbose=True,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               added_timestep=0,
               eps_known=None,
               curve_power=2.0, 
               **kwargs
               ):
        
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        device = self.model.betas.device
        
        if not torch.is_tensor(noise_variance):
            noise_variance = torch.tensor(noise_variance, device=device)

        # 1. 画像ごとの開始ステップ計算 (Batch個別)
        alpha_bar_t = self.alphas_cumprod[added_timestep]
        current_noise_level = torch.sqrt(1 - alpha_bar_t) + torch.sqrt(noise_variance)
        alpha_bar_u = 1 / (current_noise_level**2 + 1)
        
        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        
        # added_timestep より未来から始まらないようにクランプ
        start_timesteps = torch.max(start_timesteps, torch.tensor(added_timestep, device=device))
        start_timesteps = start_timesteps.to(device)
        start_timesteps = torch.clamp(start_timesteps, min=1, max=999) # 0除算防止のためmin=1

        # ループ用最大値
        maxind = start_timesteps.max().item()
        
        iterator = tqdm(reversed(range(0, maxind + 1)), desc='Smooth Transition DDIM', total=maxind+1)

        img = x_T.clone()
        ddim_timesteps_tensor = torch.from_numpy(self.ddim_timesteps).to(device)

        for i, step in enumerate(iterator):
            time_idx_tensor = torch.where(ddim_timesteps_tensor == step)[0]
            if time_idx_tensor.numel() == 0: continue
            index = time_idx_tensor.item()
            
            # まだ開始していない画像を除外するマスク
            active_mask = (start_timesteps >= step).view(-1, 1, 1, 1)
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
            
            # U-Net Forward
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(img, ts, conditioning)
            else:
                x_in = torch.cat([img] * 2)
                t_in = torch.cat([ts] * 2)
                c_in = torch.cat([unconditional_conditioning, conditioning])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            # -----------------------------------------------------------
            # 修正: バッチごとの個別進捗管理
            # -----------------------------------------------------------
            if eps_known is not None:
                # float型に変換して計算
                current_step_tensor = torch.full((batch_size,), step, device=device, dtype=torch.float32)
                start_steps_float = start_timesteps.float()
                
                # 1. 進捗率 (Progress) の計算: 1.0 (開始時) -> 0.0 (終了時)
                # 画像ごとに start_timesteps が異なるため、progress も異なる
                progress = current_step_tensor / start_steps_float
                
                # 念のため 0~1 にクリップ (開始前の画像などが1を超えないように)
                progress = torch.clamp(progress, min=0.0, max=1.0)
                
                # 2. 除去率 (Removal Rate) の計算
                # (1 - progress)^n
                # progress=1.0 (開始時) -> rate=0.0 (白色残す)
                # progress=0.0 (終了時) -> rate=1.0 (白色消す)
                removal_rate = (1.0 - progress) ** curve_power
                
                # 3. ブロードキャスト用に形状変更 (B,) -> (B, 1, 1, 1)
                removal_rate = removal_rate.view(batch_size, 1, 1, 1)
                
                # 4. ヘルパー関数に渡す (Tensorのまま渡してOK)
                e_t = extract_colored_noise_smooth(e_t, eps_known, removal_rate=removal_rate)

            # DDIM Update
            alphas = self.ddim_alphas
            alphas_prev = self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
            sigmas = self.ddim_sigmas

            a_t = torch.full((batch_size, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((batch_size, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((batch_size, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((batch_size, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(img.shape, device, False) * eta
            img_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

            img = torch.where(active_mask, img_prev, img)

        return img
    

    @torch.no_grad()
    def known_noise_guided_ddim_sampling(self,
               S,
               batch_size,
               shape,
               noise_variance, 
               conditioning=None,
               eta=0.,
               x_T=None,
               verbose=True,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               added_timestep=0,
               eps_known=None,
               known_guidance_scale=0.5, # Slerpの混合比率 (0.0: AI予測のみ, 1.0: 既知ノイズのみ)
               **kwargs
               ):
        """
        Slerpを用いた既知ノイズガイドサンプリング
        
        Args:
            known_guidance_scale (float): 0.0 ~ 1.0 の値。
                - 0.3~0.5 程度推奨。
                - 大きすぎるとチャネルノイズ(推定誤差)まで正解として取り込んでしまう。
                - 小さすぎると既知ノイズの恩恵が得られない。
        """
        
        # 1. スケジュールの作成
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        device = self.model.betas.device
        
        if not torch.is_tensor(noise_variance):
            noise_variance = torch.tensor(noise_variance, device=device)

        # 2. 開始タイムステップの計算 (推定)
        alpha_bar_t = self.alphas_cumprod[added_timestep]
        current_noise_level = torch.sqrt(1 - alpha_bar_t) + torch.sqrt(noise_variance)
        alpha_bar_u = 1 / (current_noise_level**2 + 1)
        
        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        
        # 安全策: 送信時刻より未来(小さいt)から始めない
        start_timesteps = torch.max(start_timesteps, torch.tensor(added_timestep, device=device))
        start_timesteps = start_timesteps.to(device)
        start_timesteps = torch.clamp(start_timesteps, min=0, max=999)

        if verbose:
            print(f"Computed start_timesteps (mean): {start_timesteps.float().mean().item():.2f}")

        # 3. メインループ
        maxind = start_timesteps.max().item()
        iterator = tqdm(reversed(range(0, maxind + 1)), desc='Slerp Guided DDIM', total=maxind+1)

        img = x_T.clone()
        ddim_timesteps_tensor = torch.from_numpy(self.ddim_timesteps).to(device)

        for i, step in enumerate(iterator):
            # A. スケジュール判定
            time_idx_tensor = torch.where(ddim_timesteps_tensor == step)[0]
            if time_idx_tensor.numel() == 0: continue
            index = time_idx_tensor.item()

            # B. Active Mask
            active_mask = (start_timesteps >= step).view(-1, 1, 1, 1)

            # C. ノイズ予測 (U-Net Forward)
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
            
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(img, ts, conditioning)
            else:
                x_in = torch.cat([img] * 2)
                t_in = torch.cat([ts] * 2)
                c_in = torch.cat([unconditional_conditioning, conditioning])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            # -----------------------------------------------------------
            # D. Slerpによる補正 (ここを変更)
            # -----------------------------------------------------------
            if eps_known is not None:
                # Slerpで「予測ノイズ」と「既知ノイズ」をブレンドする
                # stepが大きい（ノイズが多い）ときは、既知ノイズを信頼しすぎると危険（推定誤差も大きいため）
                # stepが小さい（仕上げ段階）ときは、既知ノイズに近づけたい
                
                # ここではシンプルに固定比率、または動的比率を適用可能
                # 例: 0.4 程度で予測値を既知ノイズ側に引っ張る
                
                # ブレンド率の決定 (0.0=予測値そのまま, 1.0=既知ノイズそのもの)
                # 安全のため、added_timestep に近づくほど信頼度を上げる手もあるが、
                # まずは固定値で安定させる
                ratio = known_guidance_scale 
                
                # Slerp実行
                e_t = slerp_tensor(ratio, e_t, eps_known)

            # E. 状態更新 (Standard DDIM Step)
            alphas = self.ddim_alphas
            alphas_prev = self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
            sigmas = self.ddim_sigmas

            a_t = torch.full((batch_size, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((batch_size, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((batch_size, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((batch_size, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(img.shape, device, False) * eta
            img_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

            # F. マスク適用
            img = torch.where(active_mask, img_prev, img)

        return img

    # --------------------------------------------------------------------------------
    # (その他のメソッド: forward_diffusion, sample, ddim_sampling 等は変更なしで維持してください)
    # 必要であれば以前のコードからコピペしてください。
    # --------------------------------------------------------------------------------
    
    @torch.no_grad()
    def apply_model(self, x, t, c):
        # 互換性維持のためのラッパー
        return self.model.apply_model(x, t, c)
    @torch.no_grad
    def search_timestep(self,
               S, #ddim_num_steps 200
               batch_size,
               shape,
               noise_sigma,
               noise_sigma_predict,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               intermediate_path = None,
               intermediate_skip = 1,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               snr = None,
               starttimestep = None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        print(f"ddim.py alphas_cumprod = {self.alphas_cumprod.shape}") #1000
        C, H, W = shape
        size = (batch_size, C, H, W)

        device = self.model.betas.device
        alpha_bar_u = 1/(1 + noise_sigma_predict)
        print(f"ddim.py, alpha_bar_u = {alpha_bar_u}")
        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        start_timesteps *= 1 # ここでタイムステップをいじくる
        #torch.clamp(start_timesteps, 0, S)
        if starttimestep != None:
            start_timesteps = torch.full_like(start_timesteps, starttimestep)


        return start_timesteps
    @torch.no_grad()
    def proposed_dps_sampling(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               y=None, H_hat=None, Sigma_inv=None, z_init=None, zeta=1.0,
               mapper=None, inv_mapper=None,
               initial_noise_variance=None, # MMSE後の残留ノイズ分散
               eta=0., verbose=True, unconditional_guidance_scale=1., unconditional_conditioning=None,
               **kwargs
               ):
        
        # --- Helper for Logging ---
        def get_tensor_stats_str(tensor):
            with torch.no_grad():
                if torch.isnan(tensor).any():
                    return "!!! CONTAINS NaN !!!"
                mean = tensor.mean().item()
                std = tensor.std().item()
                max_val = tensor.abs().max().item()
                norm = torch.norm(tensor).item()
                return f"Mean: {mean:.4f} | Std: {std:.4f} | MaxAbs: {max_val:.4f} | Norm: {norm:.4f}"
        # --------------------------

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        device = self.model.betas.device
        
        # ----------------------------------------------------------------
        # Start Timestep Determination (Consistent with GCR)
        # ----------------------------------------------------------------
        if initial_noise_variance is not None:
            if torch.is_tensor(initial_noise_variance):
                est_noise_var = initial_noise_variance.mean().item()
            else:
                est_noise_var = initial_noise_variance
            if verbose:
                print(f"[Proposed] Using provided initial_noise_variance: {est_noise_var:.6f}")
        else:
            avg_precision = Sigma_inv.abs().mean().item()
            est_noise_var = 1.0 / (avg_precision + 1e-8)
            if verbose:
                print(f"[Proposed] Estimating noise variance from Sigma_inv (Fallback): {est_noise_var:.6f}")
        
        target_alpha = 1.0 / (1.0 + est_noise_var)
        diffs = torch.abs(self.alphas_cumprod.to(device) - target_alpha)
        start_t_ddpm = torch.argmin(diffs).item()
        
        ddim_timesteps_tensor = torch.from_numpy(self.ddim_timesteps).to(device)
        abs_diff = torch.abs(ddim_timesteps_tensor - start_t_ddpm)
        start_index = torch.argmin(abs_diff).item()
        
        actual_start_step = self.ddim_timesteps[start_index]
        if verbose:
            print(f"[Proposed] Start: DDIM Step {start_index}/{S} (t={actual_start_step})")

        # 2. Initialization
        z_init = z_init.to(device)
        img = z_init.clone()
        
        # Proposed does NOT update H_hat, but we verify it's a tensor
        if isinstance(H_hat, torch.Tensor):
            current_H = H_hat.clone().detach() # No gradients needed for H
        else:
             # Wrapper class case (MatrixOperator)
             current_H = H_hat 

        if verbose and isinstance(current_H, torch.Tensor):
            print(f"[Step Init] H_hat | {get_tensor_stats_str(current_H)}")

        # 3. Sampling Loop
        timesteps = self.ddim_timesteps[:start_index+1]
        time_range = np.flip(timesteps)
        iterator = tqdm(time_range, desc='Proposed Sampling', total=len(time_range))

        for i, step in enumerate(iterator):
            index = np.where(self.ddim_timesteps == step)[0][0]
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            should_print = (i == 0) or (i % 10 == 0) or (step % 20 == 0)

            # --- A. Gradient Computation ---
            with torch.enable_grad():
                img_in = img.detach().requires_grad_(True)

                # UNet Prediction
                if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                    e_t = self.model.apply_model(img_in, ts, conditioning)
                else:
                    x_in = torch.cat([img_in] * 2)
                    t_in = torch.cat([ts] * 2)
                    c_in = torch.cat([unconditional_conditioning, conditioning])
                    e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                    e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

                # Tweedie Estimation (x0_hat)
                alphas = self.ddim_alphas
                sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
                
                a_t = torch.full((batch_size, 1, 1, 1), alphas[index], device=device)
                sqrt_one_minus_at = torch.full((batch_size, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
                
                pred_z0 = (img_in - sqrt_one_minus_at * e_t) / a_t.sqrt()

                # Physical Mapping & Loss
                s_hat, _ = mapper(pred_z0) 
                
                # Check if H_hat is a Tensor or MatrixOperator
                if isinstance(current_H, torch.Tensor):
                    y_est = torch.matmul(current_H, s_hat)
                else:
                    y_est = current_H * s_hat # Using MatrixOperator's __mul__

                residual = y - y_est
                
                weighted_res = residual * Sigma_inv 
                
                K = residual.shape[1] * residual.shape[2] 
                loss_val = 0.5 * torch.sum(torch.conj(residual) * weighted_res).real / K
                
                # Backward
                guidance_grad = torch.autograd.grad(loss_val, img_in)[0]
                
                # Logging
                current_loss = loss_val.item()
                grad_norm = torch.linalg.norm(guidance_grad.reshape(batch_size, -1), dim=1).mean().item()
                
                iterator.set_postfix(Loss=f"{current_loss:.4f}", z_grad=f"{grad_norm:.4f}")

                if should_print and verbose:
                    score_norm = torch.linalg.norm(e_t.reshape(batch_size, -1), dim=1).mean().item()
                    print(f"[Step {step}] Loss: {current_loss:.5f} | ScoreNorm: {score_norm:.4f} | GradNorm: {grad_norm:.4f}")

            # --- B. Scaling ---
            # Proposed method scaling (zeta decay)
            max_timestep = self.ddim_timesteps[-1]
            decay_factor = step / max_timestep 
            current_zeta = zeta * decay_factor
            
            scaled_grad = guidance_grad * current_zeta

            # Gradient Clipping
            clip_val = 0.1 
            scaled_grad = torch.clamp(scaled_grad, min=-clip_val, max=clip_val)

            # --- C. DDIM Update Step ---
            with torch.no_grad():
                alphas_prev = self.ddim_alphas_prev
                sigmas = self.ddim_sigmas
                a_prev = torch.full((batch_size, 1, 1, 1), alphas_prev[index], device=device)
                sigma_t = torch.full((batch_size, 1, 1, 1), sigmas[index], device=device)
                
                # Standard DDIM Step
                dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
                noise = sigma_t * noise_like(img.shape, device, False) * eta
                img_prev_ddim = a_prev.sqrt() * pred_z0 + dir_xt + noise
                
                # Apply Guided Gradient (Subtraction)
                img = img_prev_ddim - scaled_grad
                
                # Clamp Latents
                img = torch.clamp(img, min=-3.0, max=3.0)

                if torch.isnan(img).any():
                    print(f"!!! NAN DETECTED at Step {step} !!!")
                    break 

        # Return tuple to match GCR signature (img, H_final)
        # For Proposed, H_final is same as initial H_hat
        return img, current_H
    

    
    
    @torch.no_grad()
    def gcr_burst_sampling(self,
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
               monitor_batch_idx=0, # [NEW] 監視するバッチインデックス
               **kwargs
               ):
        """
        Early Burst Calibration & Latent Reset Sampling
        1. Burst Phase: 初期画像推定を用いてチャネルHを事前補正
        2. Latent Reset: 補正されたHでMMSEフィルタを作り直し、z_Tをリセット (ノイズ付加なし)
        3. Main Phase: 通常のGCRサンプリング
        
        [修正] 指定された monitor_batch_idx のLatent履歴(img_history)を返す
        """
        
        # --- Helper: MMSE Solver for Reset ---
        def compute_new_initial_latent(y_batch, H_batch, noise_pwr, target_mean, target_std):
            B_local, r, t = H_batch.shape
            H_herm = H_batch.mH
            Gram = torch.matmul(H_herm, H_batch)
            Reg = noise_pwr * torch.eye(t, device=H_batch.device).unsqueeze(0)
            inv_mat = torch.inverse(Gram + Reg)
            W_mmse = torch.matmul(inv_mat, H_herm)
            
            s_new = torch.matmul(W_mmse, y_batch)
            z_raw = inv_mapper(s_new, (B_local, *shape))
            
            # --- [修正箇所] バッチごとの正規化 ---
            # z_raw の形状は (B, C, H, W) を想定
            
            # 1. バッチ以外の次元を平坦化して計算しやすくする (B, C*H*W)
            z_flat = z_raw.view(B_local, -1)
            
            # 2. バッチごとに平均と標準偏差を計算 (dim=1)
            # keepdim=True にすることで (B, 1) になり、ブロードキャスト計算が可能になる
            batch_mean = z_flat.mean(dim=1, keepdim=True)
            batch_std = z_flat.std(dim=1, keepdim=True)
            
            # 3. 元の形状 (B, C, H, W) に戻せるようにビューを調整
            # ※ target_mean, target_std もスカラーの場合はそのままで良いですが、
            #    もしバッチごとの統計量なら同様に shape を合わせる必要があります。
            
            # reshape用のtupleを作成 (B, 1, 1, 1)
            view_shape = (B_local,) + (1,) * (z_raw.ndim - 1)
            
            batch_mean = batch_mean.view(view_shape)
            batch_std = batch_std.view(view_shape)

            # 4. 正規化と再スケーリング
            z_new = (z_raw - batch_mean) / (batch_std + 1e-8)
            z_new = z_new * target_std + target_mean
            
            return z_new

        # 1. Setup & Schedule
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=False)
        device = self.model.betas.device
        
        if initial_noise_variance is not None:
            est_noise_var = initial_noise_variance.mean().item() if torch.is_tensor(initial_noise_variance) else initial_noise_variance
        else:
            avg_precision = Sigma_inv.abs().mean().item()
            est_noise_var = 1.0 / (avg_precision + 1e-8)
        
        target_alpha = 1.0 / (1.0 + est_noise_var)
        diffs = torch.abs(self.alphas_cumprod.to(device) - target_alpha)
        start_t_ddpm = torch.argmin(diffs).item()
        
        ddim_timesteps_tensor = torch.from_numpy(self.ddim_timesteps).to(device)
        abs_diff = torch.abs(ddim_timesteps_tensor - start_t_ddpm)
        start_index = torch.argmin(abs_diff).item()
        
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
                K_dim = residual.shape[1] * residual.shape[2]
                loss_data = 0.5 * torch.sum(torch.conj(residual) * weighted_res).real / K_dim
                
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

        # [NEW] 指定されたバッチインデックスのみ保存
        img_history = [img[monitor_batch_idx].detach().cpu().clone()]

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
                K_dim = residual.shape[1] * residual.shape[2] 
                loss_val = 0.5 * torch.sum(torch.conj(residual) * weighted_res).real / K_dim
                
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
            current_zeta = zeta 
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
                
                # [NEW] 指定インデックスのみ保存
                img_history.append(img[monitor_batch_idx].detach().cpu().clone())

                if verbose and (i % 1 == 0):
                    logs = {"Loss": f"{loss_val.item():.2f}"}
                    if H_true is not None:
                        logs["H_err"] = f"{torch.norm(current_H - H_true).item():.3f}"
                    iterator.set_postfix(logs)

        # img_history を返り値に追加
        return img, current_H, H_history, burst_loss_history, main_loss_history, img_history