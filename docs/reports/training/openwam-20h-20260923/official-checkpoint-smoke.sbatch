#!/usr/bin/env bash
#SBATCH --job-name=openwam-yam-off-save
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=420G
#SBATCH --time=02:00:00
#SBATCH --output=/home/wuyan/lyj/openwam-runs/control/yam-20h-official-checkpoint-smoke-%j.log
set -euo pipefail
code=/home/wuyan/lyj/openwam-install-7c5861e/code
python=/home/wuyan/.conda/envs/vla-openwam/bin/python
output=/home/wuyan/lyj/openwam-runs/yam-20h-official-checkpoint-smoke-20260923-${SLURM_JOB_ID}
cd "$code"
export PATH="/home/wuyan/.conda/envs/vla-openwam/bin:$PATH"
export CUDA_HOME=/home/wuyan/.conda/envs/vla-openwam
export LD_LIBRARY_PATH="$CUDA_HOME/lib/python3.12/site-packages/nvidia/curand/lib:$CUDA_HOME/lib/python3.12/site-packages/torch/lib:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
printf 'job=%s host=%s time=%s commit=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)" "$(git rev-parse HEAD)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
"$python" adapters/openwam/train.py \
  --config configs/native/openwam-yam-20h-official-checkpoint-smoke.json \
  --output "$output"
