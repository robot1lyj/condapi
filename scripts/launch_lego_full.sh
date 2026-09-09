#!/usr/bin/env bash
set -euo pipefail
# Run from an immutable Gitea-sourced checkout on the login node, inside tmux.
# Usage: bash scripts/launch_lego_full.sh JOB_ID [--resume]
job_id=${1:?Pass a currently valid four-GPU allocation ID}
shift
[[ "$job_id" =~ ^[0-9]+$ ]] || exit 2
code_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
run_name=${LEGO_RUN_NAME:-lego_full_b64}
[[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$ ]] || exit 2
control_dir=/home/wuyan/lyj/YAM/training-runs/control/$run_name
mkdir -p "$control_dir"
exec 9>"$control_dir/launch.lock"
flock -n 9 || { echo 'An existing launcher holds the run lock'; exit 1; }
export PYTHONPATH="$code_dir/src:$code_dir/packages/openpi-client/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
cd "$code_dir"
exec srun --jobid="$job_id" --overlap -N1 -n1 -c32 --gres=gpu:4 --time=3-00:00:00 \
    /home/wuyan/.conda/envs/condapi-yam/bin/python scripts/train_lego_full.py "$@" --run-name "$run_name" \
    >>"$control_dir/train.log" 2>&1
