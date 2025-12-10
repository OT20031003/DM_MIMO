import argparse, os, sys, glob
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
    # Ensure shapes match
    if x.shape != y.shape:
        # Simple resize to match x (sent image)
        y_img = Image.fromarray(y)
        y_img = y_img.resize((x.shape[1], x.shape[0]))
        y = np.array(y_img)

    if metric == 'ssim':
        data_range = float(x.max() - x.min())
        if data_range == 0: data_range = 255.0
        # Win_size must be smaller than image side
        win_size = min(x.shape[0], x.shape[1], 7)
        if win_size % 2 == 0: win_size -= 1
        return ssim(x, y, channel_axis=-1, data_range=data_range, win_size=win_size)

    xd = x.astype(np.float64)
    yd = y.astype(np.float64)
    mse = float(np.mean((xd - yd) ** 2))

    if metric == 'mse':
        return mse
    
    elif metric == 'psnr':
        if mse == 0:
            return 100.0 # Perfect match
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
    Parses filenames based on outputs.
    Expected format for outputs: "{method}_snr{value}_{id}.png"
    e.g., bench_snr-5_0.png, burst_reset_snr10_1.png
    """
    name_no_ext = os.path.splitext(filename)[0]
    
    try:
        if is_sent:
            # Matches "original_0", "image_15", etc. -> extracts ID
            match = re.search(r'_(\d+)$', name_no_ext)
            if match:
                return {'id': match.group(1)}
            return None
        else:
            # Regex to find "snr" followed by a number (possibly negative), then "_" and ID
            match = re.search(r'snr(-?\d+)_(\d+)$', name_no_ext)
            if match:
                return {'snr': match.group(1), 'id': match.group(2)}
            return None
    except ValueError:
        return None

def calculate_snr_vs_metric(sent_path, received_path, metric='ssim', resize=(256,256), lpips_model=None, device=None):
    dic_sum = {}
    dic_num = {}

    if not os.path.isdir(received_path):
        return [], []

    print(f"  Processing: {received_path} ...")

    # 1. Load Sent Images Map
    sent_images = {}
    if os.path.isdir(sent_path):
        for sp in os.listdir(sent_path):
            if not sp.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            info = parse_filename_info(sp, is_sent=True)
            if info:
                sent_images[info['id']] = os.path.join(sent_path, sp)
    
    if not sent_images:
        print("    Error: No valid images found in sent directory (expected format: original_{id}.png)")
        return [], []

    # 2. Iterate Received Images
    valid_files = 0
    for rp in os.listdir(received_path):
        if not rp.lower().endswith(('.png', '.jpg', '.jpeg')): continue

        info = parse_filename_info(rp, is_sent=False)
        if not info: continue

        img_id = info['id']
        snr_str = info['snr']

        if img_id in sent_images:
            try:
                # Load and Resize
                sentimg = Image.open(sent_images[img_id]).convert('RGB')
                recimg = Image.open(os.path.join(received_path, rp)).convert('RGB')

                if resize is not None:
                    sentimg = sentimg.resize(resize)
                    recimg = recimg.resize(resize)

                sentarr = np.array(sentimg)
                recarr = np.array(recimg)

                # Compute
                val = compute_metric(sentarr, recarr, metric=metric, lpips_model=lpips_model, device=device)

                dic_sum[snr_str] = dic_sum.get(snr_str, 0.0) + val
                dic_num[snr_str] = dic_num.get(snr_str, 0) + 1
                valid_files += 1
            except Exception as e:
                print(f"    Warning processing {rp}: {e}")
                continue

    if valid_files == 0:
        print(f"    -> No matched files found (Check filename format '...snrX_Y.png').")
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
    
    xy.sort() # Sort by SNR
    x_vals = [item[0] for item in xy]
    y_vals = [item[1] for item in xy]
    
    return x_vals, y_vals

def get_style(method_key, mode_key):
    """
    Returns (color, linestyle, marker) based on method and mode.
    """
    # 1. Determine Color by Method
    if "burst_reset" in method_key:
        color = 'green'  
        marker = '*'     
    elif "proposed" in method_key:
        color = 'red'
        marker = 'o' 
    elif "mmse_bench" in method_key:
        color = 'blue'
        marker = 's' 
    elif "mmse_linear" in method_key:
        color = 'black'
        marker = 'x'
    else:
        color = 'gray'
        marker = '.'

    # 2. Determine Line Style by Mode
    if mode_key == "perfect":
        linestyle = '-'  # Solid for Perfect
    elif mode_key == "estimated":
        linestyle = '--' # Dashed for Estimated
    else:
        linestyle = '-.'

    return color, linestyle, marker

def plot_results(results, metric_name, t, r):
    """
    results: list of tuples (x_vals, y_vals, label, method_key, mode_key)
    """
    plt.figure(figsize=(10, 6))
    
    for x_vals, y_vals, label, method_key, mode_key in results:
        if not x_vals: continue
        
        c, l, m = get_style(method_key, mode_key)
        
        plt.plot(x_vals, y_vals, marker=m, linestyle=l, label=label, color=c, markersize=8, linewidth=2)
    
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel(f"{metric_name.upper()}", fontsize=12)
    plt.title(f"MIMO ({t}x{r}) Evaluation - {metric_name.upper()}", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Filename includes all plotted info implicitly
    out_filename = f"eval_mimo_t{t}_r{r}_{metric_name}.png"
    plt.tight_layout()
    plt.savefig(out_filename, bbox_inches='tight')
    print(f"\n[Plot Saved] {out_filename}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate MIMO Methods (Linear, Bench, Proposed, Burst)")
    
    # Configuration
    parser.add_argument("--t", type=int, default=2, help="Transmit antennas")
    parser.add_argument("--r", type=int, default=2, help="Receive antennas")
    
    # Selection Arguments
    parser.add_argument("--modes", nargs='+', default=["estimated", "perfect"], 
                        choices=["estimated", "perfect"], 
                        help="CSI Estimation modes to include (space separated)")
    
    # Targets corresponding strictly to the provided files:
    # 1. burst_reset -> mimo_dps_burst_reset.py (GCR Anchor/Burst)
    # 2. proposed    -> mimo_dps_proposed.py (Standard DPS)
    # 3. mmse_bench  -> bench_MMSE.py (Diffusion)
    # 4. mmse_linear -> bench_MMSE.py (No Sample)
    parser.add_argument("--targets", nargs='+', 
                        default=["burst_reset", "proposed", "mmse_bench", "mmse_linear"], 
                        choices=["burst_reset", "proposed", "mmse_bench", "mmse_linear"],
                        help="Methods to plot")

    parser.add_argument("--sent", "-s", default="./sentimg", help="Directory containing original images (original_X.png)")
    parser.add_argument("--metric", "-m", choices=["ssim","mse","psnr","lpips","all"], default="lpips", help="Metric to use")
    parser.add_argument("--resize", type=int, default=256, help="Image resize dimension (square)")
    # python eval_dps_gcr.py -m all --modes estimated
    args = parser.parse_args()

    # ==========================================
    # Define Paths (Based on provided script outputs)
    # ==========================================
    
    # Path from bench_MMSE.py
    base_benchmark = f"outputs/MIMO_Benchmark_MMSE/t={args.t}_r={args.r}"
    
    # Path from mimo_dps_proposed.py
    base_proposed = f"outputs/MIMO_Proposed_LS/t={args.t}_r={args.r}"
    
    # Path from mimo_dps_burst_reset.py
    base_burst = f"outputs/MIMO_Burst_Reset/t={args.t}_r={args.r}"
    
    # Construct the list of paths to evaluate
    eval_targets = []

    for mode in args.modes:
        for target in args.targets:
            if target == "burst_reset":
                path = os.path.join(base_burst, mode)
                label = f"Proposed (Burst+GCR) [{mode}]"
                eval_targets.append((path, label, "burst_reset", mode))

            elif target == "proposed":
                path = os.path.join(base_proposed, mode)
                label = f"Proposed (DPS) [{mode}]"
                eval_targets.append((path, label, "proposed", mode))
            
            elif target == "mmse_bench":
                path = os.path.join(base_benchmark, mode)
                label = f"MMSE + Blind Diffusion [{mode}]"
                eval_targets.append((path, label, "mmse_bench", mode))
            
            elif target == "mmse_linear":
                # Linear results are in a 'nosample' subdirectory
                path = os.path.join(base_benchmark, "nosample", mode)
                label = f"Linear MMSE (No Diffusion) [{mode}]"
                eval_targets.append((path, label, "mmse_linear", mode))

    metrics_to_run = ["ssim", "psnr", "lpips"] if args.metric == "all" else [args.metric]

    # Initialize LPIPS if needed
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
        print(f" EVALUATING METRIC: {metric.upper()} ")
        print(f"==========================================")
        
        plot_data = []
        
        for path, label, method_key, mode_key in eval_targets:
            if os.path.exists(path):
                x, y = calculate_snr_vs_metric(
                    args.sent, path, 
                    metric=metric, 
                    resize=(args.resize, args.resize),
                    lpips_model=lpips_model, device=device
                )
                if x:
                    plot_data.append((x, y, label, method_key, mode_key))
            else:
                print(f"  [Skipping] Path not found: {path}")

        if plot_data:
            plot_results(plot_data, metric, args.t, args.r)
        else:
            print("No valid data found to plot. Please check directory paths and file naming.")

if __name__ == "__main__":
    main()