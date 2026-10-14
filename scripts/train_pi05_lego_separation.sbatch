#!/usr/bin/env bash
# Confirmed recipe v1; submit only with the user's training authorization.
#SBATCH --job-name=pi05-lego-separation-r1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=08:00:00
set -euo pipefail
code=${LEGO_SEPARATION_CODE_DIR:?Set the immutable Gitea checkout}
commit=${LEGO_SEPARATION_CODE_COMMIT:?Set its approved full commit}
[[ "$commit" =~ ^[a-f0-9]{40}$ && "$(git -C "$code" rev-parse HEAD)" == "$commit" ]]
git -C "$code" diff --quiet HEAD -- scripts src packages configs
cd "$code"
ulimit -n 65535
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
unset JAX_PLATFORMS
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
exec /home/wuyan/.conda/envs/condapi-yam/bin/python scripts/pi05_lego_separation.py train "$@"
