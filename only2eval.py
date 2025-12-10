import numpy as np
import torch
import lpips
import sys
from skimage.metrics import structural_similarity as ssim
from PIL import Image # OpenCV(cv2) の代わりに Pillow をインポート
import warnings # 警告を非表示にするためにインポート

# --- LPIPSモデルのキャッシュ用グローバル変数 ---
_lpips_model = None
_lpips_device = None

def np_to_torch_lpips(img_np):
    """
    NumPy配列(H, W, C) [0, 255] RGB を
    LPIPS用PyTorchテンソル(B, C, H, W) [-1, 1] RGB に変換するヘルパー関数
    
    (PillowはRGBで読み込むため、BGR->RGB変換は不要)
    """
    
    # [0, 255] -> [-1, 1]
    img_np = img_np.astype(np.float32) / 127.5 - 1.0
    
    # HWC -> CHW
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
    
    # CHW -> BCHW
    return img_tensor.unsqueeze(0)


def compute_metric_from_paths(path1, path2, metric='ssim'):
    """
    2つの画像ファイルパスを受け取り、指定された評価指標を計算する関数。
    (画像読み込みに Pillow を使用)

    Args:
        path1 (str): 比較元の画像ファイルパス
        path2 (str): 比較対象の画像ファイルパス
        metric (str): 'ssim', 'mse', 'psnr', 'lpips' のいずれか

    Returns:
        float: 計算された指標の値
    """
    
    # --- 1. 画像の読み込み (Pillowを使用) と検証 ---
    try:
        # Pillowで画像を開き、NumPy配列に変換
        x = np.array(Image.open(path1))
        y = np.array(Image.open(path2))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"画像の読み込みに失敗しました: {e.filename}")
    except Exception as e:
        raise IOError(f"画像の読み込み中にエラーが発生しました: {e}")

    # グレースケール画像をRGBに変換 (もしあれば)
    if x.ndim == 2:
        print(f"Gray to RGB")
        x = np.stack([x] * 3, axis=-1)
    if y.ndim == 2:
        y = np.stack([y] * 3, axis=-1)
        
    if x.shape != y.shape:
        raise ValueError(f"画像サイズ・チャンネルが異なります: {x.shape} vs {y.shape}")

    # --- 2. SSIMの計算 ---
    if metric == 'ssim':
        data_range = float(x.max() - x.min())
        if data_range == 0:
            return 1.0 if np.array_equal(x, y) else 0.0
        
        # channel_axis=-1 は (H, W, C) 形式を示す
        # Pillow (RGB) でも OpenCV (BGR) でもピクセル値の構造的類似性は同じ
        return ssim(x, y, channel_axis=-1, data_range=data_range, win_size=7)

    # --- 3. MSE / PSNR の計算準備 ---
    xd = x.astype(np.float64)
    yd = y.astype(np.float64)
    mse = float(np.mean((xd - yd) ** 2))

    if metric == 'mse':
        return mse
    
    elif metric == 'psnr':
        if mse == 0:
            return np.inf
        max_pixel = 255.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        return float(psnr)
        
    # --- 4. LPIPSの計算 ---
    elif metric == 'lpips':
        global _lpips_model, _lpips_device
        
        if _lpips_model is None:
            # print("初回実行：LPIPSモデル(AlexNet)をロード中...", file=sys.stderr) # ログを非表示化
            
            # --- 警告非表示設定 ---
            # torch.loadのFutureWarningを非表示にする
            warnings.filterwarnings(
                "ignore", 
                message="You are using `torch.load` with `weights_only=False`*", 
                category=FutureWarning
            )
            # --------------------

            _lpips_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # verbose=False を追加して、LPIPSの読み込みログを非表示にする
            _lpips_model = lpips.LPIPS(net='alex', verbose=False).to(_lpips_device)
            _lpips_model.eval()

        # NumPy画像をLPIPS用のTorchテンソルに変換
        tensor_x = np_to_torch_lpips(x).to(_lpips_device)
        tensor_y = np_to_torch_lpips(y).to(_lpips_device)
        
        with torch.no_grad():
            dist = _lpips_model(tensor_x, tensor_y)
            
        return float(dist.item())

    else:
        raise ValueError("Metric must be 'ssim', 'mse', 'psnr', or 'lpips'.")

# --- 実行例 ---
if __name__ == "__main__":
    # --- 必要なライブラリ ---
    # pip install numpy torch lpips scikit-image Pillow
    
    

    # --- 各指標を計算 ---
    path1 = "./intermediate/k=0.0/0/0/400/step_0266.png"
    path2 = "./sentimg/sentimg_0.png"
    path_same = "./sentimg/sentimg_0.png"
    path3 = "./intermediate/k=0.0/0/0/400/step_0016.png"

    try:
        # 1. 元画像 vs ノイズ画像
        print("-" * 30)
        print(f"[{path1}] vs [{path2}]")
        psnr_val = compute_metric_from_paths(path1, path2, metric='psnr')
        print(f"   PSNR: {psnr_val:.4f} dB")
        
        ssim_val = compute_metric_from_paths(path1, path2, metric='ssim')
        print(f"   SSIM: {ssim_val:.4f}")

        lpips_val = compute_metric_from_paths(path1, path2, metric='lpips')
        print(f"   LPIPS: {lpips_val:.4f}")
        print("-" * 30)
        print(f"[{path3}] vs [{path2}]")
        psnr_val = compute_metric_from_paths(path2, path3, metric='psnr')
        print(f"   PSNR: {psnr_val:.4f} dB")
        
        ssim_val = compute_metric_from_paths(path3, path2, metric='ssim')
        print(f"   SSIM: {ssim_val:.4f}")

        lpips_val = compute_metric_from_paths(path3, path2, metric='lpips')
        print(f"   LPIPS: {lpips_val:.4f}")
        print("-" * 30)

        

    except Exception as e:
        print(f"エラーが発生しました: {e}")

"""
filename = ./intermediate/k=0.0/0/0/300/step_0051.png, sentdir = ./sentimg/sentimg_0.png, val = 0.43268436193466187
filename = ./intermediate/k=0.0/0/0/300/step_0271.png, sentdir = ./sentimg/sentimg_0.png, val = 0.4060175120830536
filename = ./intermediate/k=0.0/0/0/300/step_0046.png, sentdir = ./sentimg/sentimg_0.png, val = 0.43575233221054077
filename = ./intermediate/k=0.0/0/0/300/step_0101.png, sentdir = ./sentimg/sentimg_0.png, val = 0.40344691276550293
filename = ./intermediate/k=0.0/0/0/300/step_0116.png, sentdir = ./sentimg/sentimg_0.png, val = 0.39673054218292236
filename = ./intermediate/k=0.0/0/0/300/step_0221.png, sentdir = ./sentimg/sentimg_0.png, val = 0.38992786407470703
filename = ./intermediate/k=0.0/0/0/300/step_0196.png, sentdir = ./sentimg/sentimg_0.png, val = 0.38552290201187134
filename = ./intermediate/k=0.0/0/0/300/step_0141.png, sentdir = ./sentimg/sentimg_0.png, val = 0.3892253041267395
filename = ./intermediate/k=0.0/0/0/300/step_0021.png, sentdir = ./sentimg/sentimg_0.png, val = 0.4559253454208374
filename = ./intermediate/k=0.0/0/0/300/step_0086.png, sentdir = ./sentimg/sentimg_0.png, val = 0.4106801152229309
filename = ./intermediate/k=0.0/0/0/300/step_0281.png, sentdir = ./sentimg/sentimg_0.png, val = 0.4103766679763794
filename = ./intermediate/k=0.0/0/0/300/step_0131.png, sentdir = ./sentimg/sentimg_0.png, val = 0.3912757933139801
filename = ./intermediate/k=0.0/0/0/300/step_0081.png, sentdir = ./sentimg/sentimg_0.png, val = 0.4136209487915039
filename = ./intermediate/k=0.0/0/0/300/step_0011.png, sentdir = ./sentimg/sentimg_0.png, val = 0.46471673250198364
filename = ./intermediate/k=0.0/0/0/300/step_0096.png, sentdir = ./sentimg/sentimg_0.png, val = 0.40564024448394775
filename = ./intermediate/k=0.0/0/0/300/step_0016.png, sentdir = ./sentimg/sentimg_0.png, val = 0.46067550778388977
filename = ./intermediate/k=0.0/0/0/300/step_0091.png, sentdir = ./sentimg/sentimg_0.png, val = 0.4083825945854187
filename = ./intermediate/k=0.0/0/0/300/step_0216.png, sentdir = ./sentimg/sentimg_0.png, val = 0.3889044523239136
filename = ./intermediate/k=0.0/0/0/300/step_0266.png, sentdir = ./sentimg/sentimg_0.png, val = 0.4037543833255768
filename = ./intermediate/k=0.0/0/0/300/step_0236.png, sentdir = ./sentimg/sentimg_0.png, val = 0.3924546241760254
filename = ./intermediate/k=0.0/0/0/300/step_0041.png, sentdir = ./sentimg/sentimg_0.png, val = 0.43963423371315
filename = ./intermediate/k=0.0/0/0/300/step_0286.png, sentdir = ./sentimg/sentimg_0.png, val = 0.4134387671947479
filename = ./intermediate/k=0.0/0/0/300/step_0036.png, sentdir = ./sentimg/sentimg_0.png, val = 0.4433409869670868
filename = ./intermediate/k=0.0/0/0/300/step_0061.png, sentdir = ./sentimg/sentimg_0.png, val = 0.4258459508419037
filename = ./intermediate/k=0.0/0/0/300/step_0126.png, sentdir = ./sentimg/sentimg_0.png, val = 0.39273273944854736

"""