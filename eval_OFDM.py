import argparse
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# --- Imports for LPIPS ---
try:
    import torch
    import lpips
except ImportError:
    print("Warning: 'torch' or 'lpips' libraries not found. LPIPS metric will fail.")
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
    Computes the similarity/error between image pair x (GT), y (Rec).
    x, y: numpy arrays [H, W, 3] usually.
    """
    # Ensure shapes match (resize Rec to GT)
    if x.shape != y.shape:
        y_img = Image.fromarray(y)
        y_img = y_img.resize((x.shape[1], x.shape[0]), Image.BICUBIC)
        y = np.array(y_img)

    if metric == 'ssim':
        data_range = float(x.max() - x.min())
        if data_range == 0: data_range = 255.0
        win_size = min(x.shape[0], x.shape[1], 7)
        if win_size % 2 == 0: win_size -= 1
        return ssim(x, y, channel_axis=-1, data_range=data_range, win_size=win_size)

    xd = x.astype(np.float64)
    yd = y.astype(np.float64)
    mse = float(np.mean((xd - yd) ** 2))

    if metric == 'mse':
        return mse
    
    elif metric == 'psnr':
        if mse == 0: return 100.0
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
        raise ValueError(f"Unknown metric: {metric}")

def parse_filename_info(filename, is_sent=False):
    """
    Parses filenames.
    Sent images: "original_0.png" -> id=0
    Rec images: "method_snr10_0.png" or "method_snr-5_15.png" -> snr=10/-5, id=0/15
    """
    name_no_ext = os.path.splitext(filename)[0]
    
    try:
        if is_sent:
            # Match "original_X" or "image_X" -> extract X
            match = re.search(r'_(\d+)$', name_no_ext)
            if match:
                return {'id': match.group(1)}
            return None
        else:
            # Match "...snr(NUMBER)_(ID)"
            # Handles negative snr like "snr-5_0"
            match = re.search(r'snr(-?\d+)_(\d+)$', name_no_ext)
            if match:
                return {'snr': match.group(1), 'id': match.group(2)}
            return None
    except ValueError:
        return None

def calculate_snr_vs_metric(sent_path, received_path, metric='ssim', resize=(256,256), lpips_model=None, device=None):
    """
    Scans directories, matches files by ID, groups by SNR, and averages metrics.
    """
    dic_sum = {} # {snr_str: sum_metric}
    dic_num = {} # {snr_str: count}

    # Check directory existence
    if not os.path.isdir(received_path):
        print(f"  [Error] Directory not found: {received_path}")
        return [], []
    
    # Auto-detect 'estimated' subdirectory if user passed parent folder
    if not glob.glob(os.path.join(received_path, "*.png")):
        sub_est = os.path.join(received_path, "estimated")
        if os.path.isdir(sub_est) and glob.glob(os.path.join(sub_est, "*.png")):
            print(f"  [Info] No PNGs in root, checking subdirectory: {sub_est}")
            received_path = sub_est

    print(f"  Processing: {received_path} ...")

    # 1. Map Sent Images {id: path}
    sent_images = {}
    if os.path.isdir(sent_path):
        for sp in os.listdir(sent_path):
            if not sp.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            info = parse_filename_info(sp, is_sent=True)
            if info:
                sent_images[info['id']] = os.path.join(sent_path, sp)
    else:
        print(f"    [Error] Sent directory not found: {sent_path}")
        return [], []

    # 2. Iterate Received Images
    files = os.listdir(received_path)
    valid_files = 0
    
    for rp in files:
        if not rp.lower().endswith(('.png', '.jpg', '.jpeg')): continue

        info = parse_filename_info(rp, is_sent=False)
        if not info: continue

        img_id = info['id']
        snr_str = info['snr']

        if img_id in sent_images:
            try:
                # Load
                sentimg = Image.open(sent_images[img_id]).convert('RGB')
                recimg = Image.open(os.path.join(received_path, rp)).convert('RGB')

                # Resize if needed
                if resize is not None:
                    if sentimg.size != resize: sentimg = sentimg.resize(resize, Image.BICUBIC)
                    if recimg.size != resize: recimg = recimg.resize(resize, Image.BICUBIC)

                sentarr = np.array(sentimg)
                recarr = np.array(recimg)

                # Compute Metric
                val = compute_metric(sentarr, recarr, metric=metric, lpips_model=lpips_model, device=device)

                dic_sum[snr_str] = dic_sum.get(snr_str, 0.0) + val
                dic_num[snr_str] = dic_num.get(snr_str, 0) + 1
                valid_files += 1
            except Exception as e:
                print(f"    Warning processing {rp}: {e}")
                continue

    if valid_files == 0:
        print(f"    -> No matched/valid files found in {received_path}.")
        return [], []

    # 3. Aggregate
    xy = []
    for snr_key, total in dic_sum.items():
        try:
            snr_float = float(snr_key)
            count = dic_num[snr_key]
            avg = total / count
            xy.append((snr_float, avg))
        except ValueError:
            continue
    
    # Sort by SNR
    xy.sort(key=lambda x: x[0])
    
    x_vals = [item[0] for item in xy]
    y_vals = [item[1] for item in xy]
    
    print(f"    -> Processed {valid_files} images across {len(xy)} SNR points.")
    return x_vals, y_vals

def get_style(method_key):
    """
    Returns (color, marker, label_suffix) based on method type.
    """
    if "burst" in method_key:
        return 'green', '*', ' (Burst/GCR)'
    elif "proposed" in method_key:
        return 'red', 'o', ' (Proposed DPS)'
    elif "bench" in method_key:
        return 'blue', 's', ' (MMSE+Diff)'
    elif "linear" in method_key:
        return 'black', 'x', ' (Linear MMSE)'
    else:
        return 'gray', '.', ''

def plot_combined_results(results, metric_name, out_filename):
    """
    results: list of (x_vals, y_vals, legend_label, style_key)
    """
    plt.figure(figsize=(10, 7))
    
    for x_vals, y_vals, label, style_key in results:
        if not x_vals: continue
        color, marker, _ = get_style(style_key)
        plt.plot(x_vals, y_vals, marker=marker, linestyle='-', label=label, color=color, markersize=8, linewidth=2)
    
    plt.xlabel("SNR (dB)", fontsize=14)
    plt.ylabel(f"{metric_name.upper()}", fontsize=14)
    plt.title(f"MIMO-OFDM Evaluation - {metric_name.upper()}", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(out_filename, dpi=300)
    print(f"\n[Plot Saved] {out_filename}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate MIMO Methods by Output Directory")
    
    # --- Input Directories (Flexible) ---
    parser.add_argument("--dir_burst", type=str, default=None, help="Path to Burst/GCR result images")
    parser.add_argument("--dir_proposed", type=str, default=None, help="Path to Proposed DPS result images")
    parser.add_argument("--dir_bench", type=str, default=None, help="Path to Benchmark (Diff) result images")
    parser.add_argument("--dir_linear", type=str, default=None, help="Path to Linear (NoSample) result images")
    
    # --- General Settings ---
    parser.add_argument("--sent", "-s", default="./sentimg", help="Directory containing original images (original_X.png)")
    parser.add_argument("--metric", "-m", choices=["ssim","mse","psnr","lpips","all"], default="lpips", help="Metric to use")
    parser.add_argument("--resize", type=int, default=256, help="Image resize dimension (square)")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output plot filename")
    
    args = parser.parse_args()

    # Define the evaluation tasks based on provided arguments
    # Tuple format: (Path, Legend Label, Style Key)
    tasks = []
    
    if args.dir_burst:
        tasks.append((args.dir_burst, "Burst GCR", "burst"))
    if args.dir_proposed:
        tasks.append((args.dir_proposed, "Proposed DPS", "proposed"))
    if args.dir_bench:
        tasks.append((args.dir_bench, "MMSE + Diffusion", "bench"))
    if args.dir_linear:
        tasks.append((args.dir_linear, "Linear MMSE", "linear"))

    if not tasks:
        print("Error: No input directories provided.")
        print("Usage Example:")
        print("  python eval_dps_gcr.py --dir_burst outputs/BurstExp/estimated --dir_bench outputs/BenchExp/estimated")
        return

    # Prepare Metrics
    metrics_to_run = ["ssim", "psnr", "lpips"] if args.metric == "all" else [args.metric]

    # Initialize LPIPS
    lpips_model = None
    device = None
    if "lpips" in metrics_to_run:
        if lpips is None or torch is None:
            print("Error: LPIPS requested but library missing.")
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nInitializing LPIPS model on {device}...")
        lpips_model = lpips.LPIPS(net='alex').to(device).eval()

    # --- Main Evaluation Loop ---
    for metric in metrics_to_run:
        print(f"\n==========================================")
        print(f" EVALUATING METRIC: {metric.upper()} ")
        print(f"==========================================")
        
        plot_data = []
        
        for path, label, style_key in tasks:
            x, y = calculate_snr_vs_metric(
                args.sent, path, 
                metric=metric, 
                resize=(args.resize, args.resize),
                lpips_model=lpips_model, device=device
            )
            
            if x:
                plot_data.append((x, y, label, style_key))
        
        if plot_data:
            out_name = f"eval_comparison_{metric}{args.suffix}.png"
            plot_combined_results(plot_data, metric, out_name)
        else:
            print("No valid data found to plot.")

if __name__ == "__main__":
    main()

# 全手法の結果を LPIPS で比較
"""
python eval_OFDM.py --sent ./sentimg --dir_burst outputs/MIMO_OFDM_Burst/OFDM_t=2_r=2_steps=200_burst=1000_blr=5e-05/estimated --dir_proposed outputs/MIMO_OFDM_Proposed/t=2_r=2/estimated --dir_bench outputs/MIMO_Benchmark_OFDM/t=2_r=2/estimated --dir_linear outputs/MIMO_Benchmark_OFDM/t=2_r=2/nosample/estimated --metric all

"""
