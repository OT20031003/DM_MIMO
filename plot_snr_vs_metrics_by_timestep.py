import os
import argparse
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# --- New Imports for LPIPS ---
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
    # From HWC to NCHW
    img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    # Normalize from [0, 255] to [-1, 1]
    img_tensor = (img_tensor / 127.5) - 1.0
    return img_tensor

def compute_metric(x, y, metric='ssim', lpips_model=None, device=None):
    """
    Computes the similarity/error between image pair x, y.
    metric: 'ssim', 'mse', 'psnr', or 'lpips'
    """
    if metric == 'ssim':
        # Assumes RGB images. Specify data_range for stability.
        # Ensure data_range is not zero if images are solid color
        data_range = float(x.max() - x.min())
        if data_range == 0:
            return 1.0 # Images are identical and flat
        return ssim(x, y, channel_axis=-1, data_range=data_range)

    # For MSE and PSNR, convert to float and calculate MSE
    xd = x.astype(np.float64)
    yd = y.astype(np.float64)
    mse = float(np.mean((xd - yd) ** 2))

    if metric == 'mse':
        return mse
    
    elif metric == 'psnr':
        if mse == 0:
            # Images are identical, PSNR is infinite
            return np.inf
        max_pixel = 255.0  # Assuming 8-bit images (0-255)
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        return float(psnr)
        
    elif metric == 'lpips':
        if lpips_model is None or device is None:
            raise ValueError("lpips_model and device must be provided for LPIPS metric.")
        
        # Convert numpy arrays (HWC, 0-255) to torch tensors (NCHW, -1 to 1)
        tensor_x = np_to_torch(x).to(device)
        tensor_y = np_to_torch(y).to(device)
        
        # Calculate LPIPS
        with torch.no_grad(): # No need to track gradients
            dist = lpips_model(tensor_x, tensor_y)
        return float(dist.item())

    else:
        raise ValueError("Metric must be 'ssim', 'mse', 'psnr', or 'lpips'.")

# --- ⭐ 新しいロジック: 階層ディレクトリをスキャンする関数 ---
def calculate_snr_vs_metric_hierarchical(
    sent_path, base_recv_path, target_timestep, 
    metric='ssim', resize=(256,256), lpips_model=None, device=None
):
    """
    Compares images in 'sent_path' against a hierarchical 'base_recv_path'.
    Structure assumed: base_recv_path / SNR / IMG_ID / target_timestep / image_file.png
    """
    dic_sum = {} # Keyed by SNR (float)
    dic_num = {} # Keyed by SNR (float)

    print(f"\n--- Processing: Timestep={target_timestep}, Metric={metric.upper()} ---")

    # 1. Get all 'sent' images and map their IDs
    sent_images = {} # {img_id: full_path}
    if not os.path.isdir(sent_path):
        print(f"Error: Directory not found: {sent_path}")
        return [], []
        
    for sp in os.listdir(sent_path):
        if not sp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        # ファイル名 (拡張子なし) から数字を抽出 (例: 'img_1.png' -> '1')
        img_id = "".join(filter(str.isdigit, os.path.splitext(sp)[0]))
        if not img_id:
            print(f"Warning: Could not extract ID from sent image: {sp}")
            continue
        sent_images[img_id] = os.path.join(sent_path, sp)
    
    if not sent_images:
        print(f"Error: No images found in sent path: {sent_path}")
        return [], []

    # 2. Walk the base_recv_path to find matches
    # Structure: base_recv_path / SNR / IMG_ID / TIMESTEP / image.png
    
    snr_dirs = [d for d in os.listdir(base_recv_path) if os.path.isdir(os.path.join(base_recv_path, d))]

    for snr_str in snr_dirs:
        try:
            # SNRディレクトリ名から数値 (マイナス含む) を抽出 (例: '-1' -> -1.0)
            snr_float = float("".join(filter(lambda c: c.isdigit() or c in '.-', snr_str)))
        except ValueError:
            # print(f"Warning: Skipping non-SNR directory: {snr_str}")
            continue
        
        snr_dir_path = os.path.join(base_recv_path, snr_str)
        img_id_dirs = [d for d in os.listdir(snr_dir_path) if os.path.isdir(os.path.join(snr_dir_path, d))]
        # print(f"snr_dir_path = {snr_dir_path}")
        # print(f"img_id_dirs= {img_id_dirs}")
        for img_id in img_id_dirs:
            if img_id not in sent_images:
                # この img_id (例: '1') が sent_images (例: {'1':...}) に存在しない場合はスキップ
                continue
            
            # We found a matching image ID
            timestep_dir_path = os.path.join(snr_dir_path, img_id, target_timestep)
            
            if os.path.isdir(timestep_dir_path):
                # このディレクトリ内の画像ファイルを探す
                
                image_files = []
                for f in os.listdir(timestep_dir_path):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_files.append(f)

                if not image_files:
                    # print(f"Warning: No image file found in {timestep_dir_path}")
                    continue
                
                # 複数の画像がある場合、ソートして最後の画像 (例: step_0399.png) を使用する
                # 最初のノイズ除去の保存画像を使用
                image_files.sort()
                rec_image_name = image_files[-1] # Take the last one -1 start 0
                rec_image_path = os.path.join(timestep_dir_path, rec_image_name)
                sent_image_path = sent_images[img_id]

                try:
                    sentimg = Image.open(sent_image_path).convert('RGB')
                    recimg = Image.open(rec_image_path).convert('RGB')

                    if resize is not None:
                        sentimg = sentimg.resize(resize)
                        recimg = recimg.resize(resize)

                    sentarr = np.array(sentimg)
                    recarr = np.array(recimg)
                    
                    val = compute_metric(sentarr, recarr, metric=metric, lpips_model=lpips_model, device=device)
                    
                    dic_sum[snr_float] = dic_sum.get(snr_float, 0.0) + val
                    dic_num[snr_float] = dic_num.get(snr_float, 0) + 1
                
                except Exception as e:
                    print(f"Warning: Error processing {rec_image_path}: {e}")
                    continue

    # 3. Compile results
    if not dic_sum:
        print(f"Warning: No matching data found for timestep {target_timestep}.")
        return [], []

    xy = []
    for snr_float, total in dic_sum.items():
        try:
            count = dic_num[snr_float]
            avg = total / count
            xy.append((snr_float, avg))
            print(f"  SNR: {snr_float: >5} dB, Average {metric.upper()}: {avg:.6f} (count={count})")
        except (ValueError, ZeroDivisionError):
            print(f"Warning: Could not process SNR key '{snr_float}'. Skipping.")
            continue

    xy.sort()  # Sort by SNR
    x_vals = [item[0] for item in xy]
    y_vals = [item[1] for item in xy]
    return x_vals, y_vals

def plot_results(results, title_suffix="", output_filename="snr_vs_metric.png"):
    """
    Plots the results.
    results: list of tuples (x_vals, y_vals, label)
    """
    plt.figure(figsize=(10,6))
    for x_vals, y_vals, label in results:
        if not x_vals:
            continue
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', label=label)
    
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel(f"Average {title_suffix.strip('()')} Value", fontsize=12)
    plt.title(f"SNR vs. Metric Comparison {title_suffix}", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"\nPlot saved as '{output_filename}'.")

def main():
    parser = argparse.ArgumentParser(description="SNR vs Metric comparison for hierarchical directories")
    parser.add_argument("--sent", "-s", default="./sentimg", help="Directory for 'sent' (original) images")
    
    # --- ⭐ 変更点: 比較対象ディレクトリの指定方法 ---
    parser.add_argument("--base_recv", "-r", default="./intermediate/k=0.0", 
                        help="Base directory for 'received' images (e.g., ./intermediate/k=0.0)")
    
    parser.add_argument("--metric", "-m", choices=["ssim","mse","psnr","lpips","all"], default="ssim", 
                        help="Metric to use (ssim, mse, psnr, lpips, or all)")
    parser.add_argument("--resize", type=int, nargs=2, metavar=('W','H'), default=(256,256), 
                        help="Resize dimensions for comparison (W H)")
    args = parser.parse_args()

    # --- ⭐ 変更点: タイムステップディレクトリを自動検出 ---
    # Structure: base_recv / SNR / IMG_ID / TIMESTEP
    
    timestep_dirs = set()
    if not os.path.isdir(args.base_recv):
        print(f"Error: Base recv directory not found: {args.base_recv}")
        return

    print(f"Scanning {args.base_recv} for timestep directories...")
    try:
        for snr_dir in os.listdir(args.base_recv):
            snr_path = os.path.join(args.base_recv, snr_dir)
            if not os.path.isdir(snr_path): continue
            
            for img_id_dir in os.listdir(snr_path):
                img_id_path = os.path.join(snr_path, img_id_dir)
                if not os.path.isdir(img_id_path): continue
                
                for timestep_dir in os.listdir(img_id_path):
                    timestep_path = os.path.join(img_id_path, timestep_dir)
                    if os.path.isdir(timestep_path):
                        timestep_dirs.add(timestep_dir)
    except Exception as e:
        print(f"Error scanning directory structure: {e}")
        print("Please ensure --base_recv points to the directory *above* the SNR folders (e.g., '.../k=0.0').")
        return

    if not timestep_dirs:
        print(f"Error: No timestep subdirectories (e.g., '400', '500') found in the structure under {args.base_recv}.")
        return
    
    # タイムステップをソート (例: '400', '500', '1000')
    sorted_timesteps = sorted(list(timestep_dirs), key=lambda x: int("".join(filter(str.isdigit, x)) or 0))
    print(f"Found timesteps: {sorted_timesteps}")

        
    # --- Metric selection (same as eval4) ---
    metrics_to_run = []
    if args.metric == "all":
        metrics_to_run = ["ssim", "mse", "psnr", "lpips"]
    else:
        metrics_to_run = [args.metric]

    # --- LPIPS init (same as eval4) ---
    lpips_model = None
    device = None
    if "lpips" in metrics_to_run:
        if lpips is None or torch is None:
            print("Error: LPIPS metric requested, but 'torch' or 'lpips' libraries are not installed.")
            print("Please run: pip install torch lpips")
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nInitializing LPIPS model (AlexNet) on device: {device}")
        lpips_model = lpips.LPIPS(net='alex').to(device).eval()


    # --- ⭐ 変更点: メトリックごと、タイムステップごとにループ ---
    
    for metric in metrics_to_run:
        # このメトリック用のプロット結果を格納
        metric_plot_results = [] 
        
        for timestep in sorted_timesteps:
            # 新しい階層スキャン関数を呼び出す
            x_vals, y_vals = calculate_snr_vs_metric_hierarchical(
                args.sent, args.base_recv, timestep, 
                metric=metric, resize=tuple(args.resize), 
                lpips_model=lpips_model, device=device
            )
            
            # ラベル (例: "T=400 - SSIM")
            label = f"T={timestep} - {metric.upper()}"
            metric_plot_results.append((x_vals, y_vals, label))

        if not metric_plot_results:
            print(f"No data to plot for metric {metric}.")
            continue

        # メトリックごとにプロットを保存
        outname = f"snr_vs_{metric}_comparison_plt_snr_vs_metrics_by_timestep.png"
        plot_results(metric_plot_results, title_suffix=f"({metric.upper()})", output_filename=outname)

if __name__ == "__main__":
    main()