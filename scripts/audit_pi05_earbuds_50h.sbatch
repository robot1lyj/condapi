#!/usr/bin/env bash
# CPU-only source audit; writes new selection reports and never alters raw files.
#SBATCH --job-name=earbuds-audit-50h
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
set -euo pipefail
code=${EARBUDS_CODE_DIR:?Set an immutable Gitea checkout}
commit=${EARBUDS_CODE_COMMIT:?Set the expected full commit}
[[ "$(git -C "$code" rev-parse HEAD)" == "$commit" ]]
git -C "$code" diff --quiet HEAD -- scripts
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$code"
exec /home/wuyan/.conda/envs/condapi-yam/bin/python scripts/prepare_yam_clean_subset.py \
  --source /home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks/earbuds_charging_case \
  --output /home/wuyan/lyj/YAM/training-assets/earbuds_clean_selection_20261013 \
  --contract "$code/docs/reports/training/earbuds-pi05-50h-20261012/contract.json" --workers 32
