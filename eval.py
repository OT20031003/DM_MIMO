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
    img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img_tensor = (img_tensor / 127.5) - 1.0
    return img_tensor

def compute_metric(x, y, metric='ssim', lpips_model=None, device=None):
    if x.shape != y.shape:
        y_img = Image.fromarray(y)
        y_img = y_img.resize((x.shape[1], x.shape[0]))
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
        return float(10 * np.log10((max_pixel ** 2) / mse))
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
    name_no_ext = os.path.splitext(filename)[0]
    try:
        if is_sent:
            match = re.search(r'_(\d+)$', name_no_ext)
            return {'id': match.group(1)} if match else None
        else:
            match = re.search(r'snr(-?\d+)_(\d+)$', name_no_ext)
            return {'snr': match.group(1), 'id': match.group(2)} if match else None
    except ValueError:
        return None

def calculate_snr_vs_metric(sent_path, received_path, metric='ssim', resize=(256,256), lpips_model=None, device=None):
    dic_sum = {}
    dic_num = {}

    if not os.path.isdir(received_path):
        return [], []

    print(f"  Processing directory: {received_path}")

    sent_images = {}
    if os.path.isdir(sent_path):
        for sp in os.listdir(sent_path):
            if not sp.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            info = parse_filename_info(sp, is_sent=True)
            if info:
                sent_images[info['id']] = os.path.join(sent_path, sp)
    
    if not sent_images:
        print("    Error: No valid images found in sent directory.")
        return [], []

    valid_files = 0
    for rp in os.listdir(received_path):
        if not rp.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        info = parse_filename_info(rp, is_sent=False)
        if not info: continue

        img_id = info['id']
        snr_str = info['snr']

        if img_id in sent_images:
            try:
                sentimg = Image.open(sent_images[img_id]).convert('RGB')
                recimg = Image.open(os.path.join(received_path, rp)).convert('RGB')
                if resize:
                    sentimg = sentimg.resize(resize)
                    recimg = recimg.resize(resize)
                val = compute_metric(np.array(sentimg), np.array(recimg), metric=metric, lpips_model=lpips_model, device=device)
                dic_sum[snr_str] = dic_sum.get(snr_str, 0.0) + val
                dic_num[snr_str] = dic_num.get(snr_str, 0) + 1
                valid_files += 1
            except Exception as e:
                print(f"    Warning: {e}")
                continue

    if valid_files == 0:
        print(f"    -> No matched files found.")
        return [], []

    xy = []
    for snr_key, total in dic_sum.items():
        try:
            xy.append((float(snr_key), total / dic_num[snr_key]))
        except ValueError: continue
    
    xy.sort()
    return [x[0] for x in xy], [x[1] for x in xy]

def get_style(method_key, mode_key):
    # Color & Marker
    if "burst_nosample" in method_key:
        color = 'tab:olive'
        marker = 'd' # Diamond for Burst NoSample
    elif "burst_reset" in method_key:
        color = 'tab:green'
        marker = '*'     
    elif "proposed" in method_key:
        color = 'tab:red'
        marker = 'o' 
    elif "mmse_bench" in method_key:
        color = 'tab:blue'
        marker = 's' 
    elif "mmse_linear" in method_key:
        color = 'black'
        marker = 'x'
    else:
        color = 'gray'
        marker = '.'

    # Line Style
    if mode_key == "perfect":
        linestyle = '-'
    elif mode_key == "estimated":
        linestyle = '--'
    else:
        linestyle = '-.' # Unknown/Mixed

    return color, linestyle, marker

def plot_results(results, metric_name, t, r, args=None):
    plt.figure(figsize=(12, 8))
    for x_vals, y_vals, label, method_key, mode_key in results:
        if not x_vals: continue
        c, l, m = get_style(method_key, mode_key)
        plt.plot(x_vals, y_vals, marker=m, linestyle=l, label=label, color=c, markersize=8, linewidth=2)
    
    plt.xlabel("SNR (dB)", fontsize=14)
    plt.ylabel(f"{metric_name.upper()}", fontsize=14)
    plt.title(f"MIMO ({t}x{r}) Evaluation - {metric_name.upper()}", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
    
    exp_suffix = "_custom" if args.exp_name else ""
    out_filename = f"eval_mimo_t{t}_r{r}_{metric_name}{exp_suffix}.png"
    plt.tight_layout()
    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    print(f"\n[Plot Saved] {out_filename}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate MIMO Methods")
    parser.add_argument("--exp_name", type=str, default=None, 
                        help="Directly specify the experiment folder path (can be deep path like .../estimated)")
    
    # Params for default/bench path construction
    parser.add_argument("--t", type=int, default=2)
    parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--burst_iterations", type=int, default=20)
    parser.add_argument("--burst_lr", type=float, default=0.05)
    parser.add_argument("--anchor_lambda", type=float, default=1.0)
    parser.add_argument("--dps_scale", type=float, default=0.3)
    
    parser.add_argument("--modes", nargs='+', default=["estimated", "perfect"], choices=["estimated", "perfect"])
    parser.add_argument("--targets", nargs='+', default=["burst_reset", "proposed", "mmse_bench", "mmse_linear"], 
                        choices=["burst_reset", "proposed", "mmse_bench", "mmse_linear"])

    parser.add_argument("--sent", "-s", default="./sentimg")
    parser.add_argument("--metric", "-m", choices=["ssim","mse","psnr","lpips","all"], default="lpips")
    parser.add_argument("--resize", type=int, default=256)
    args = parser.parse_args()

    # ==========================================
    # Logic to resolve Burst/Custom paths
    # ==========================================
    burst_paths_to_eval = [] # List of tuples: (path, label, method_key, mode_key)

    if args.exp_name:
        # 1. Normalize Path
        if os.path.exists(args.exp_name):
            custom_path = args.exp_name
        elif os.path.exists(os.path.join("outputs", args.exp_name)):
            custom_path = os.path.join("outputs", args.exp_name)
        else:
            print(f"Warning: Custom path '{args.exp_name}' not found.")
            custom_path = None

        if custom_path:
            # 2. Check if Leaf (contains images directly)
            png_files = glob.glob(os.path.join(custom_path, "*_*.png"))
            is_leaf = len(png_files) > 0

            if is_leaf:
                print(f"Target '{custom_path}' recognized as a leaf directory.")
                # Infer metadata from path string
                lower_p = custom_path.lower()
                
                # Mode Detection
                if "perfect" in lower_p: 
                    detected_mode = "perfect"
                elif "estimated" in lower_p: 
                    detected_mode = "estimated"
                else: 
                    detected_mode = "estimated" # default
                
                # Type Detection (Sample vs NoSample)
                is_nosample = "nosample" in lower_p
                
                # Key & Label Construction
                if is_nosample:
                    m_key = "burst_nosample"
                    label_tag = "Burst(NoSample)"
                else:
                    m_key = "burst_reset"
                    label_tag = "Burst(Custom)"

                label = f"{label_tag} [{detected_mode}]\n{os.path.basename(custom_path)}"
                burst_paths_to_eval.append((custom_path, label, m_key, detected_mode))
            
            else:
                # Treated as Base Directory
                print(f"Target '{custom_path}' recognized as a base directory.")
                for mode in args.modes:
                    path = os.path.join(custom_path, mode)
                    label = f"Burst(Custom) [{mode}]\n{os.path.basename(custom_path)}"
                    # Assume normal burst if not specified otherwise
                    burst_paths_to_eval.append((path, label, "burst_reset", mode))
    else:
        # Standard Parameter-based Construction
        burst_param_str = (f"t={args.t}_r={args.r}_steps={args.ddim_steps}_"
                           f"burst={args.burst_iterations}_blr={args.burst_lr}_"
                           f"lam={args.anchor_lambda}_zeta={args.dps_scale}")
        base_burst = f"outputs/MIMO_Burst_Reset/{burst_param_str}"
        print(f"Using Auto-Generated Burst Directory: {base_burst}")
        
        for mode in args.modes:
            path = os.path.join(base_burst, mode)
            label = f"Burst+GCR [{mode}]\n(iter={args.burst_iterations})"
            burst_paths_to_eval.append((path, label, "burst_reset", mode))

    # Standard Paths for Benchmarks
    base_proposed = f"outputs/MIMO_Proposed_LS/t={args.t}_r={args.r}"
    base_benchmark = f"outputs/MIMO_Benchmark_MMSE/t={args.t}_r={args.r}"

    # Construct Final Evaluation List
    eval_targets = []
    
    for target in args.targets:
        if target == "burst_reset":
            # Add all resolved burst paths
            for item in burst_paths_to_eval:
                eval_targets.append(item)

        elif target == "proposed":
            for mode in args.modes:
                path = os.path.join(base_proposed, mode)
                label = f"Proposed DPS [{mode}]"
                eval_targets.append((path, label, "proposed", mode))
        
        elif target == "mmse_bench":
            for mode in args.modes:
                path = os.path.join(base_benchmark, mode)
                label = f"MMSE+Diff [{mode}]"
                eval_targets.append((path, label, "mmse_bench", mode))
        
        elif target == "mmse_linear":
            for mode in args.modes:
                path = os.path.join(base_benchmark, "nosample", mode)
                label = f"Linear MMSE [{mode}]"
                eval_targets.append((path, label, "mmse_linear", mode))

    # Execution Loop
    metrics_to_run = ["ssim", "psnr", "lpips"] if args.metric == "all" else [args.metric]
    lpips_model = None
    device = None
    if "lpips" in metrics_to_run:
        if lpips is None or torch is None:
            print("Error: LPIPS requested but not installed.")
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing LPIPS model on {device}...")
        lpips_model = lpips.LPIPS(net='alex').to(device).eval()

    for metric in metrics_to_run:
        print(f"\n=== EVALUATING METRIC: {metric.upper()} ===")
        plot_data = []
        for path, label, method_key, mode_key in eval_targets:
            if os.path.exists(path):
                x, y = calculate_snr_vs_metric(
                    args.sent, path, metric=metric, 
                    resize=(args.resize, args.resize), lpips_model=lpips_model, device=device
                )
                if x: plot_data.append((x, y, label, method_key, mode_key))
            else:
                # Only warn if it's not an optional path (some modes might not exist)
                pass # print(f"  [Skipping] Path not found: {path}")

        if plot_data:
            plot_results(plot_data, metric, args.t, args.r, args)
        else:
            print("No valid data found to plot.")

if __name__ == "__main__":
    main()