#!/usr/bin/env bash
#SBATCH --job-name=pi05-earbuds-50h
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=6-00:00:00
#SBATCH --output=/home/wuyan/lyj/YAM/training-runs/control/earbuds_pi05_rtc_base_50h_clean_v2_excl_pantheon_retry4_20261022/diagnostic-%j.log
set -euo pipefail
DIAG_CODE=${DIAG_CODE:?Set the Gitea diagnostic checkout}
DIAG_COMMIT=${DIAG_COMMIT:?Set the exact diagnostic commit}
[[ "$(git -C "$DIAG_CODE" rev-parse HEAD)" == "$DIAG_COMMIT" ]]
git -C "$DIAG_CODE" diff --quiet HEAD -- scripts/collect_training_incident.py scripts/run_training_with_evidence.py
CODE=/home/wuyan/lyj/YAM/code-snapshots/earbuds50h-clean-v2-20261013
EXPECTED_COMMIT=b1446eefdf8c3cd347689bb41cc77fa6af251aaa
ASSETS=/home/wuyan/lyj/YAM/training-assets/earbuds_rtc_50h_clean_v2_excl_pantheon_20261021
DATASET=/home/wuyan/lyj/YAM/YAM_data/processed/earbuds_lerobot_50h_clean_v2_20261013/train
INIT=/home/wuyan/lyj/YAM/training-runs/pi05_yam/earbuds_pi05_rtc_base_50h_20261012/114000/params
RUN=earbuds_pi05_rtc_base_50h_clean_v2_excl_pantheon_retry4_20261022
CTRL=/home/wuyan/lyj/YAM/training-runs/control/$RUN
[[ "$(git -C "$CODE" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]]
git -C "$CODE" diff --quiet HEAD -- scripts src packages configs docs/reports/training/earbuds-pi05-50h-clean-20261013
test -s "$CTRL/initial_config.json"
test -s "$CTRL/training_contract.json"
test -s "$ASSETS/episodes.json"
test -s "$ASSETS/norm/yam/norm_stats.json"
test "$(sha256sum "$ASSETS/norm/yam/norm_stats.json" | awk '{print $1}')" = 05f65ada286965f1c5c5b32e62bf3fcdafb7d41d2e22d6683e0c05efc443c5b2
jobs=$(squeue --noheader --user "$(id -un)" --name pi05-earbuds-50h --format '%i' | sed '/^$/d')
for job in $jobs; do [[ "$job" == "$SLURM_JOB_ID" ]] || { echo "another earbuds training job is active: $job" >&2; exit 73; }; done
ulimit -n 65535
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1
export PYTHONPATH="$CODE/src:$CODE/scripts:$CODE/packages/openpi-client/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export LEGO_BATCH_SIZE=32 LEGO_NUM_WORKERS=2 LEGO_SAVE_INTERVAL=1000 LEGO_KEEP_PERIOD=5000
export LEGO_PEAK_LR=1e-5 LEGO_DECAY_LR=1e-6
unset JAX_PLATFORMS
cd "$CODE"
INCIDENT_DIR="$CTRL/incidents/job-$SLURM_JOB_ID"
INCIDENT_START=$(date '+%Y-%m-%dT%H:%M:%S')
# Reserve ample capacity for four-device GPU core dumps; no deletion/rotation.
/home/wuyan/.conda/envs/condapi-yam/bin/python - "$CTRL" <<'SPACE'
import shutil, sys
free = shutil.disk_usage(sys.argv[1]).free
print(f"DIAGNOSTIC_FREE_BYTES={free}", flush=True)
if free < 256 * 1024**3:
    raise SystemExit("Insufficient free space for diagnostic core dumps")
SPACE
printf '%s recovery_launcher job=%s run=%s mode=resume stop_step=86000\n' "$(date -Is)" "$SLURM_JOB_ID" "$RUN" >> "$CTRL/recovery.log"
exec flock -n "$CTRL/launch.lock" /home/wuyan/.conda/envs/condapi-yam/bin/python \
  "$DIAG_CODE/scripts/run_training_with_evidence.py" \
  --output "$INCIDENT_DIR" --since "$INCIDENT_START" \
  --metadata-python /home/wuyan/.conda/envs/condapi-yam/bin/python \
  --source-file "$CTRL/training_contract.json" \
  --source-file "$CTRL/initial_config.json" \
  --source-file "$CTRL/resume_config.json" \
  --source-file "$DIAG_CODE/docs/reports/training/pi05-crash-investigation-20261022/launch/diagnostic-resume.sbatch" \
  -- /home/wuyan/.conda/envs/condapi-yam/bin/python scripts/train_lego_full.py \
  --resume \
  --run-name "$RUN" \
  --steps 86000 \
  --init-params "$INIT" \
  --train-episodes-file "$ASSETS/episodes.json" \
  --rtc-training-max-delay 10 \
  --assets-dir "$ASSETS/norm" \
  --dataset "$DATASET" \
  --warmup-steps 1000 \
  --decay-steps 338000
