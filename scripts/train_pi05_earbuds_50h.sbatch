#!/usr/bin/env bash
# Submit only after explicit user authorization to start training.
#SBATCH --job-name=pi05-earbuds-50h
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=6-00:00:00
#SBATCH --output=pi05-earbuds-50h-%j.log
set -euo pipefail
code=${EARBUDS_CODE_DIR:?Set the immutable Gitea-sourced checkout directory}
commit=${EARBUDS_CODE_COMMIT:?Set its full approved Git commit SHA}
[[ "$commit" =~ ^[a-f0-9]{40}$ && "$(git -C "$code" rev-parse HEAD)" == "$commit" ]]
git -C "$code" diff --quiet HEAD -- scripts src packages configs docs/reports/training/earbuds-pi05-50h-20261012
cd "$code"
ulimit -n 65535
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
unset JAX_PLATFORMS
exec /home/wuyan/.conda/envs/condapi-yam/bin/python scripts/pi05_earbuds_50h.py train "$@"
