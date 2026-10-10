#!/usr/bin/env bash
#SBATCH --job-name=xr1-lego-50h
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=440G
#SBATCH --time=6-00:00:00
#SBATCH --output=/home/wuyan/lyj/xiaomi-robotics-1/runs/control/lego-50h-20261010/train-%j.log
set -Eeuo pipefail
mode=${1:-start}
[[ "$mode" == start || "$mode" == resume ]] || { echo 'Expected start or resume'; exit 2; }
code=${XR1_CODE_DIR:?Set XR1_CODE_DIR to the pinned server checkout}
revision=${XR1_CODE_REVISION:?Set XR1_CODE_REVISION to its full commit SHA}
output=${XR1_RUN_OUTPUT:-/home/wuyan/lyj/xiaomi-robotics-1/runs/lego-50h-eef-v1-20261010}
[[ "$(git -C "$code" rev-parse HEAD)" == "$revision" ]] || { echo 'Code revision differs'; exit 2; }
[[ -z "$(git -C "$code" status --porcelain --untracked-files=no)" ]] || { echo 'Tracked source was modified'; exit 2; }
export PATH=/home/wuyan/.conda/envs/xr1-posttrain/bin:$PATH
# Reuse only the validated CUDA SDK read-only; Python/model dependencies are
# loaded exclusively from xr1-posttrain. The system CUDA 13.2 is incompatible.
export CUDA_HOME=${XR1_CUDA_HOME:-/home/wuyan/.conda/envs/vla-openwam}
export HF_HOME=/home/wuyan/lyj/xiaomi-robotics-1/hf-home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST=8.9 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export PYTHONFAULTHANDLER=1
ulimit -n 65535
cd "$code"
printf 'XR1_START time=%s job=%s mode=%s revision=%s output=%s\n' "$(date -Is)" "$SLURM_JOB_ID" "$mode" "$revision" "$output"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
args=(--recipe "$code/configs/native/xr1-yam-lego-50h.json" --output "$output")
if [[ "$mode" == resume ]]; then args+=(--resume); fi
python -m adapters.xr1.train "${args[@]}"
