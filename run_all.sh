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

# --- 1. MMSE Benchmark ---
echo "[1/3] Running bench_MMSE.py..."
python -m scripts.bench_MMSE > execution_logs/log_mmse.txt 2>&1
echo "Finished MMSE Benchmark at $(date)"

# --- 2. Proposed DPS ---
echo "[2/3] Running mimo_dps_proposed.py..."
python -m scripts.mimo_dps_proposed > execution_logs/log_proposed.txt 2>&1
echo "Finished Proposed DPS at $(date)"

# --- 3. Burst Reset ---
echo "[3/3] Running mimo_dps_burst.py..."
python -m scripts.mimo_dps_burst_reset > execution_logs/log_burst.txt 2>&1
echo "Finished Burst Reset at $(date)"
# # --- 4. Importance Burst Reset ---
# echo "[3/3] Running mimo_importance.py..."
# python -m scripts.mimo_importance > execution_logs/log_importance.txt 2>&1
# echo "Finished Burst Reset at $(date)"

echo "==========================================="
echo "All jobs finished at $(date)"
echo "==========================================="