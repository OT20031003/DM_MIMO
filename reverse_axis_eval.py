import os
import argparse
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import re # ファイル名からステップ数を抽出するために使用
import matplotlib.colors as mcolors # ⭐ 変更点: インポートを追加

# --- LPIPS Imports (変更なし) ---
try:
    import torch
    import lpips
except ImportError:
    print("Warning: 'torch' or 'lpips' libraries not found.")
    print("To use the LPIPS metric, please install them: pip install torch lpips")
    torch = None
    lpips = None
# -------------------------------

# --- np_to_torch, compute_metric (変更なし) ---
def np_to_torch(img_np):
    img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img_tensor = (img_tensor / 127.5) - 1.0
    return img_tensor

def compute_metric(x, y, metric='ssim', lpips_model=None, device=None):
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

def main():
    parser = argparse.ArgumentParser(description="SNR vs Metric (Multiple pairs)")
    parser.add_argument("--sent", "-s", default="./sentimg", help="Directory for 'sent' (original) images")
    
    parser.add_argument("--base_recv", "-r", default="./intermediate/k2=0.0", 
                        help="Base directory for 'received' images (e.g., ./intermediate/k=0.0)")
    
    parser.add_argument("--metric", "-m", choices=["ssim","mse","psnr","lpips"], default="ssim", 
                        help="Metric to use (ssim, mse, psnr, lpips)")
    
    parser.add_argument("--resize", type=int, nargs=2, metavar=('W','H'), default=(256,256), 
                        help="Resize dimensions for comparison (W H)")
    
    parser.add_argument("--snr",type=float, nargs='+', required=True, help="List of channel SNRs (e.g., -5 0 5)")
    parser.add_argument("--timestep", "-t",type=int, nargs='+', required=True, help="List of all timesteps (e.g., 300 500 1000)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output plot filename. If None, defaults to comparison_[metric]_[...].png")

    
    args = parser.parse_args()

    if len(args.snr) != len(args.timestep):
        print(f"Error: The number of --snr values ({len(args.snr)}) must match the number of --timestep values ({len(args.timestep)}).")
        return
        
    snr_ts_pairs = list(zip(args.snr, args.timestep))

    lpips_model = None
    device = None
    if "lpips" == args.metric:
        if lpips is None or torch is None:
            print("Error: LPIPS metric requested, but 'torch' or 'lpips' libraries are not installed.")
            print("Please run: pip install torch lpips")
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nInitializing LPIPS model (AlexNet) on device: {device}")
        lpips_model = lpips.LPIPS(net='alex').to(device).eval()
    
    sent_dir = {}
    print(f"\nScanning 'sent' directory: {args.sent}")
    for sd in os.listdir(args.sent):
        # ファイル名が "img_001.png" のような形式であることを想定
        ss = sd.split(".")
        if not ss: continue
        ss = ss[0].split("_")
        if len(ss) < 2: continue
        sent_dir[ss[1]] = os.path.join(args.sent, sd) # '001': './sentimg/img_001.png'
    
    plt.figure(figsize=(10, 7)) 

    # --- ⭐ 変更点: 色、マーカー、線スタイルのリストを定義 ---
    colors = list(mcolors.TABLEAU_COLORS.values())
    colors.extend(['#000000', '#FF00FF', '#808000', '#00FF00', '#000080']) 
    markers = ['o', 'v', 's', '^', 'D', '<', '>', 'p', '*', 'X']
    linestyles = ['-', '--', '-.', ':']
    # ---------------------------------------------------

    # (SNR, Timestep) のペアでループ (⭐ 変更点: enumerate を追加)
    for i, (snr, timestep) in enumerate(snr_ts_pairs):
        print(f"\nProcessing: SNR = {snr}, TotalTimestep = {timestep}")
        
        obj_dir = {}
        for k, v in sent_dir.items():
            obj_dir[k] = []
        
        print(f"Scanning {args.base_recv} for matching directories...")
        
        for snr_dir in os.listdir(args.base_recv):
            snr_path = os.path.join(args.base_recv, snr_dir)
            if not os.path.isdir(snr_path):
                continue
            
            try:
                # ディレクトリ名 (snr_dir) と指定したSNR (snr) が一致するか確認
                if snr != float(snr_dir):
                    continue
            except ValueError:
                print(f"Skipping non-numeric directory: {snr_dir}")
                continue

            # ./intermediate/k=0.0/5.0/ などを走査
            for img_id in os.listdir(snr_path):
                img_id_path = os.path.join(snr_path, img_id)
                # img_id ('001'など) が sent_dir に存在するか確認
                if not os.path.isdir(img_id_path) or img_id not in obj_dir:
                    continue
                    
                # ./intermediate/k=0.0/5.0/001/ などを走査
                for timestep_dir in os.listdir(img_id_path):
                    if not os.path.isdir(os.path.join(img_id_path, timestep_dir)):
                        continue
                        
                    try:
                        # ディレクトリ名 (timestep_dir) が指定した timestep と一致するか確認
                        if int(timestep_dir) != timestep:
                            continue
                    except ValueError:
                        print(f"Skipping non-numeric directory: {timestep_dir}")
                        continue
                        
                    # マッチした場合、パスを追加
                    obj_dir[img_id].append(os.path.join(img_id_path, timestep_dir))

        # ステップごとに計算
        ans = {} # {step:metric_sum}
        ans_num = {} 
        for k, v in obj_dir.items():
            for d in v: # d = './intermediate/k=0.0/5.0/001/500'
                if not os.path.isdir(d):
                    continue
                for file_name in os.listdir(d):
                    # ファイル名が ..._500.png, ..._490.png などの形式であることを想定
                    step_match = re.match(r".*_(\d+)\.(png|jpg|jpeg|bmp|webp)$", file_name, re.IGNORECASE)
                    if not step_match:
                        continue 
                        
                    step = int(step_match.group(1)) # 500, 490 など
                    
                    file_path = os.path.join(d, file_name)
                    
                    try:
                        sentimg = Image.open(sent_dir[k]).convert('RGB').resize(args.resize)
                        recimg = Image.open(file_path).convert('RGB').resize(args.resize)

                        sentarr = np.array(sentimg)
                        recarr = np.array(recimg)

                        val = compute_metric(sentarr, recarr, metric=args.metric,lpips_model=lpips_model, device=device)
                        
                        ans[step] = ans.get(step, 0.0) + val
                        ans_num[step] = ans_num.get(step, 0) + 1
                    except Exception as e:
                        print(f"Warning: Error processing {file_path}: {e}")
                        continue

        if not ans:
            print(f"Warning: No valid data found for SNR={snr}, Timestep={timestep}. Skipping this pair.")
            continue
            
        ave = {}
        for k, v in ans.items():
            ave[k] = v / ans_num[k]
        
        # 降順 (reverse=True) にソート (元のTime Step t が T -> 0 になるように)
        ave = dict(sorted(ave.items(), key=lambda item: item[0], reverse=True))
        
        X = []
        Y = []
        for k, v in ave.items():
            # X軸: サンプリング回数 = TotalTimestep - 現在の Time Step
            sampling_steps_taken = timestep - int(k)
            X.append(sampling_steps_taken) # 0, 10, ..., 500 のように昇順になる
            
            Y.append(float(v))
            
        print(f"Data points for this pair (Sampling Steps, Metric): {list(zip(X, Y))}")
        
        # --- ⭐ 変更点: スタイルを循環させて適用 ---
        label = f"TotalTimestep = {timestep}, SNR = {snr}"
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        linestyle = linestyles[(i // len(colors)) % len(linestyles)] 

        plt.plot(X, Y, label=label, marker=marker, linestyle=linestyle, color=color, markersize=5)
        # ----------------------------------------

    # --- グラフの最終調整と保存 (ループの外) ---
    
    plt.xlabel("Number of Sampling Steps", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylabel(f"{args.metric.upper()}", fontsize=12)
    plt.title(f"Number of Sampling Steps vs {args.metric.upper()}", fontsize=14)
    
    # --- ⭐ 変更点: 凡例が多い場合(6個以上)はグラフの外側に表示 ---
    if len(snr_ts_pairs) > 6:
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize='small')
    else:
        plt.legend()
    # ---------------------------------------------------
    
    # レイアウトを自動調整
    plt.tight_layout()
    
    # 出力ファイル名を決定
    if args.output:
        output_filename = args.output
    else:
        # ファイル名が衝突しないよう、ユニークにする
        snr_str = "_".join(map(str, sorted(list(set(args.snr)))))
        ts_str = "_".join(map(str, sorted(list(set(args.timestep)))))
        output_filename = f"comparison_{args.metric}_snr{snr_str}_ts{ts_str}.png"
        
    # --- ⭐ 変更点: bbox_inches='tight' を追加 ---
    # これにより、グラフの外側に描画した凡例も画像に収まる
    plt.savefig(output_filename, bbox_inches='tight')
    # ------------------------------------------
    
    print(f"\nCombined graph saved to: {output_filename}")

if __name__ == "__main__":
    main()

"""
python reverse_axis_eval.py --snr 3 3 3 3 3 3 -t 200 250 255 300 350 400 -m lpips
"""