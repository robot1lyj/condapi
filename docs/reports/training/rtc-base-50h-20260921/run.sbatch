#!/usr/bin/env bash
#SBATCH --job-name=pi05-rtc-50h
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=6-00:00:00
#SBATCH --output=/home/wuyan/lyj/YAM/training-runs/control/lego_pi05_rtc_base_50h_20260921/slurm-%j.log
set -euo pipefail
mode=${1:-start}
stop_step=${2:-170000}
[[ "$stop_step" =~ ^[0-9]+$ ]] && (( stop_step > 0 && stop_step <= 338000 )) || { echo "stop step must be 1..338000"; exit 2; }
[[ "$mode" == start || "$mode" == --resume ]] || { echo 'Expected start or --resume'; exit 2; }
code=/home/wuyan/lyj/YAM/code-snapshots/rtc-50h-20260921
control=/home/wuyan/lyj/YAM/training-runs/control/lego_pi05_rtc_base_50h_20260921
assets=/home/wuyan/lyj/YAM/training-assets/lego_rtc_50h_20260921
python=/home/wuyan/.conda/envs/condapi-yam/bin/python
exec 9>"$control/launch.lock"
flock -n 9 || { echo 'Run already held by another launcher'; exit 1; }
trap 'code=$?; printf "%s exit=%s job=%s\n" "$(date -Is)" "$code" "$SLURM_JOB_ID" > "$control/exit-$SLURM_JOB_ID.txt"' EXIT
(cd "$assets" && sha256sum --check selection.sha256)
ulimit -n 65535
cd "$code"
[[ "$(sha256sum scripts/train_lego_full.py | cut -d " " -f1)" == bfe1dd3347ee2cb2d392468a10cd0e87fe5f5f3a8e20f05343eb104e55f82afd ]]
export PYTHONPATH="$code/src:$code/scripts:$code/packages/openpi-client/src"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1 XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export LEGO_RUN_NAME=lego_pi05_rtc_base_50h_20260921
export LEGO_BATCH_SIZE=32 LEGO_NUM_WORKERS=2 LEGO_PEAK_LR=1e-5 LEGO_DECAY_LR=1e-6
export LEGO_SAVE_INTERVAL=1000 LEGO_KEEP_PERIOD=5000
printf 'START job=%s mode=%s host=%s time=%s\n' "$SLURM_JOB_ID" "$mode" "$(hostname)" "$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
if [[ ! -f "$assets/norm/yam/norm_stats.json" ]]; then
    [[ "$mode" == start ]] || { echo 'Resume requires original completed norm'; exit 1; }
    JAX_PLATFORMS=cpu "$python" scripts/compute_yam_norm_stats.py \
        --dataset /home/wuyan/lyj/YAM/YAM_data/processed/lego_lerobot_v1_20260907/train \
        --train-episodes-file "$assets/episodes.json" --output "$assets/norm/yam" \
        > "$control/norm-$SLURM_JOB_ID.log" 2>&1
fi
args=(--run-name "$LEGO_RUN_NAME" --steps "$stop_step" --warmup-steps 1000 --decay-steps 338000
    --init-params /home/wuyan/.cache/openpi/openpi-assets/checkpoints/pi05_base/params
    --assets-dir "$assets/norm" --train-episodes-file "$assets/episodes.json" --rtc-training-max-delay 10)
if [[ "$mode" == --resume ]]; then args+=(--resume); fi
printf 'TRAIN_START %s\n' "$(date -Is)"
"$python" scripts/train_lego_full.py "${args[@]}" >> "$control/train.log" 2>&1
