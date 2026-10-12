#!/usr/bin/env bash
# CPU data conversion/norm only. This job never starts or submits training.
#SBATCH --job-name=earbuds-data-50h
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=earbuds-data-50h-%j.log
set -euo pipefail
code=${EARBUDS_CODE_DIR:?Set the immutable Gitea-sourced checkout directory}
commit=${EARBUDS_CODE_COMMIT:?Set its full approved Git commit SHA}
[[ "$commit" =~ ^[a-f0-9]{40}$ && "$(git -C "$code" rev-parse HEAD)" == "$commit" ]]
git -C "$code" diff --quiet HEAD -- scripts src packages configs docs/reports/training/earbuds-pi05-50h-20261012
cd "$code"
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
exec nice -n 19 /home/wuyan/.conda/envs/condapi-yam/bin/python scripts/pi05_earbuds_50h.py prepare-data "$@"
