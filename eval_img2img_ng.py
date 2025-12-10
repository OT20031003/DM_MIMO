import os
import argparse
import re
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Imports for LPIPS ---
try:
    import torch
    import lpips
except ImportError:
    print("Warning: 'torch' or 'lpips' libraries not found.")
    print("To use the LPIPS metric, please install them: pip install torch lpips")
    torch = None
    lpips = None
# -------------------------------

def np_to_torch(img_np):
    """
    Converts a NumPy image (H, W, C) in range [0, 255]
    to a PyTorch tensor (N, C, H, W) in range [-1, 1].
    """
    img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img_tensor = (img_tensor / 127.5) - 1.0
    return img_tensor

def compute_metric(x, y, metric='ssim', lpips_model=None, device=None):
    """
    Computes the similarity/error between image pair x, y.
    """
    if metric == 'ssim':
        data_range = float(x.max() - x.min())
        if data_range == 0:
            return 1.0
        return ssim(x, y, channel_axis=-1, data_range=data_range)

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
        
    elif metric == 'lpips':
        if lpips_model is None or device is None:
            raise ValueError("lpips_model and device must be provided for LPIPS metric.")
        tensor_x = np_to_torch(x).to(device)
        tensor_y = np_to_torch(y).to(device)
        with torch.no_grad():
            dist = lpips_model(tensor_x, tensor_y)
        return float(dist.item())
    else:
        raise ValueError("Metric must be 'ssim', 'mse', 'psnr', or 'lpips'.")

def parse_filename_info(filename, is_sent=False):
    """
    img2img_ng.py の出力形式に合わせてファイル名をパースする
    Sent: "sentimg_0.png" -> id=0
    Recv: "output_-5_0.png" -> snr=-5, id=0
    """
    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.split('_')

    try:
        if is_sent:
            # format: sentimg_{id}
            # parts: ['sentimg', '0']
            img_id = parts[-1]
            if not img_id.isdigit(): return None
            return {'id': img_id}
        else:
            # format: output_{snr}_{id}
            # parts example: ['output', '-5', '0'] or ['output', '10', '1']
            if len(parts) < 3: return None
            
            img_id = parts[-1]
            snr_str = parts[-2]
            
            # IDチェック
            if not img_id.isdigit(): return None
            
            # SNRチェック (負の数や小数も許容)
            # 単純なfloat変換チェック
            float(snr_str) 

            return {'id': img_id, 'snr': snr_str}
    except ValueError:
        return None

def calculate_snr_vs_metric(sent_path, received_path, metric='ssim', resize=(256,256), lpips_model=None, device=None):
    dic_sum = {}
    dic_num = {}

    if not os.path.isdir(sent_path):
        print(f"Error: Sent directory not found: {sent_path}")
        return [], []
    if not os.path.isdir(received_path):
        # 存在しないディレクトリはスキップ（警告なしで静かに戻る）
        return [], []

    print(f"Processing: {received_path} ...")

    # 送信画像をプリロードしてIDで辞書化
    sent_images = {}
    for sp in os.listdir(sent_path):
        if not sp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): continue
        
        info = parse_filename_info(sp, is_sent=True)
        if info:
            sent_images[info['id']] = os.path.join(sent_path, sp)

    if not sent_images:
        print("Error: No valid images found in sent directory (expected format: sentimg_{id}.png)")
        return [], []

    # 受信画像を走査
    for rp in os.listdir(received_path):
        if not rp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): continue

        info = parse_filename_info(rp, is_sent=False)
        if not info: continue

        img_id = info['id']
        snr_str = info['snr']

        if img_id in sent_images:
            try:
                sentimg = Image.open(sent_images[img_id]).convert('RGB')
                recimg = Image.open(os.path.join(received_path, rp)).convert('RGB')

                if resize is not None:
                    sentimg = sentimg.resize(resize)
                    recimg = recimg.resize(resize)

                sentarr = np.array(sentimg)
                recarr = np.array(recimg)

                val = compute_metric(sentarr, recarr, metric=metric, lpips_model=lpips_model, device=device)

                dic_sum[snr_str] = dic_sum.get(snr_str, 0.0) + val
                dic_num[snr_str] = dic_num.get(snr_str, 0) + 1
            except Exception as e:
                print(f"Warning: Error processing {rp}: {e}")
                continue

    if not dic_sum:
        print(f"  -> No matched files found in {received_path}")
        return [], []

    xy = []
    for snr_key, total in dic_sum.items():
        try:
            snr_float = float(snr_key)
            count = dic_num[snr_key]
            avg = total / count
            xy.append((snr_float, avg))
        except ValueError:
            continue
    
    # SNRでソート
    xy.sort()
    x_vals = [item[0] for item in xy]
    y_vals = [item[1] for item in xy]
    
    # 簡易統計表示
    if x_vals:
        print(f"  -> Processed SNR range: {min(x_vals)}dB to {max(x_vals)}dB ({len(x_vals)} points)")

    return x_vals, y_vals

def plot_results(results, title_suffix="", output_filename="snr_vs_metric.png"):
    colors = list(mcolors.TABLEAU_COLORS.values())
    colors.extend(['#000000', '#FF00FF', '#808000', '#00FF00', '#000080']) 
    markers = ['o', 'v', 's', '^', 'D', '<', '>', 'p', '*', 'X']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(10, 6))
    
    for i, (x_vals, y_vals, label) in enumerate(results):
        if not x_vals: continue
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        linestyle = linestyles[(i // len(colors)) % len(linestyles)] 
        
        plt.plot(x_vals, y_vals, marker=marker, linestyle=linestyle, label=label, color=color, markersize=6)
    
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel(f"Metric value {title_suffix}", fontsize=12)
    plt.title(f"SNR vs. Metric Comparison {title_suffix}", fontsize=14)
    
    if len(results) > 6:
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize='small')
    else:
        plt.legend()
         
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nPlot saved as '{output_filename}'.")

def main():
    parser = argparse.ArgumentParser(description="SNR vs Metric evaluation for img2img_ng.py structure")
    
    # 親ディレクトリを指定する方式に変更
    parser.add_argument("--root_dir", default="outputs/noise_distribution", 
                        help="Root directory containing noise type folders (e.g. outputs/noise_distribution)")
    
    parser.add_argument("--sent", "-s", default="./sentimg", help="Directory for 'sent' (original) images")
    parser.add_argument("--metric", "-m", choices=["ssim","mse","psnr","lpips","all"], default="ssim", help="Metric to use")
    parser.add_argument("--resize", type=int, nargs=2, metavar=('W','H'), default=(256,256), help="Resize dimensions")

    args = parser.parse_args()

    # 評価したい構造の定義 ["gaussian", "uniform", "laplace", "student_t", "impulsive"]
    noise_types = ["gaussian", "student_t", "impulsive"] 
    methods = ["DM", "nodiffusion"]

    metrics_to_run = ["ssim", "mse", "psnr", "lpips"] if args.metric == "all" else [args.metric]

    lpips_model = None
    device = None
    if "lpips" in metrics_to_run:
        if lpips is None or torch is None:
            print("Error: LPIPS requested but not installed.")
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nInitializing LPIPS model on {device}")
        lpips_model = lpips.LPIPS(net='alex').to(device).eval()

    for metric in metrics_to_run:
        print(f"\n==========================================")
        print(f" PROCESSING METRIC: {metric.upper()} ")
        print(f"==========================================")
        
        metric_results = [] 

        # 自動的にディレクトリ構造を探索
        for n_type in noise_types:
            for method in methods:
                # パス構築: root / noise_type / method
                # 例: outputs/noise_distribution/uniform/DM
                target_path = os.path.join(args.root_dir, n_type, method)
                
                # ラベル生成: "uniform (DM)"
                label = f"{n_type} ({method})"

                if os.path.exists(target_path):
                    x_vals, y_vals = calculate_snr_vs_metric(
                        args.sent, target_path, metric=metric, resize=tuple(args.resize), 
                        lpips_model=lpips_model, device=device
                    )
                    
                    if x_vals: 
                        metric_results.append((x_vals, y_vals, label))
                else:
                    # フォルダが存在しない場合はスキップ（メッセージは出さないか、デバッグ用に出す）
                    pass

        if not metric_results:
            print(f"\nNo valid data found in '{args.root_dir}'. Check directory structure.")
            continue

        outname = f"snr_vs_{metric}_comparison.png"
        plot_results(metric_results, title_suffix=f"({metric.upper()})", output_filename=outname)

    print("\nAll processing complete.")

if __name__ == "__main__":
    main()