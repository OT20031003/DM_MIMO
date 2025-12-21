#!/bin/bash

# エラーが起きても止まらずに次を実行したい場合は set -e を外してください
# set -e 

# ログ用のディレクトリ作成
mkdir -p execution_logs

# 仮想環境の有効化 (環境に合わせてパスや名前を変更してください)
# 例: source ~/anaconda3/etc/profile.d/conda.sh
# conda activate ldm 

echo "==========================================="
echo "Starting Batch Execution at $(date)"
echo "==========================================="
# --- 4. Importance Burst Reset ---
echo "[1/5] scripts.ALL_mimo_dps_burst_reset --anchor_lambda 0.0 ..."
python -m scripts.ALL_mimo_dps_burst_reset --anchor_lambda 0.0 > execution_logs/log_burst.txt 2>&1
echo "Finished Burst Reset at $(date)"

# --- 1. MMSE Benchmark ---
echo "[2/5] Running bench_MMSE_OFDMpy..."
python -m scripts.ALL_bench_MMSE > execution_logs/log_mmse.txt 2>&1
echo "Finished MMSE Benchmark at $(date)"

# --- 2. Proposed DPS ---
echo "[3/5] Running mimo_dps_proposed_OFDM .py..."
python -m scripts.ALL_mimo_dps_proposed > execution_logs/log_proposed.txt 2>&1
echo "Finished Proposed DPS at $(date)"

# --- 3. Burst Reset ---
echo "[4/5] Running mimo_dps_burst_OFDM.py..."
python -m scripts.ALL_mimo_dps_burst_reset > execution_logs/log_burst.txt 2>&1
echo "Finished Burst Reset at $(date)"

echo "[5/5] Running eval.py..."
python eval.py --exp_name "outputs/MIMO_Burst_Reset/t=2_r=2_steps=200_burst=20_blr=0.05_lam=1.0_zeta=0.3/estimated" --targets burst_reset proposed mmse_bench mmse_linear --metric all
echo "Finished Burst Reset at $(date)"
echo "==========================================="
echo "All jobs finished at $(date)"
echo "==========================================="

#nohup ./run_all.sh > global_log.txt 2>&1 &

#tail -f global_log.txt
# Burstのフォルダを直接指定しつつ、他手法もターゲットに含める例
#python eval.py --exp_name "outputs/MIMO_Burst_Reset/t=2_r=2_steps=200_burst=20_blr=0.05_lam=1.0_zeta=0.3/estimated" --targets burst_reset proposed mmse_bench mmse_linear --metric lpips