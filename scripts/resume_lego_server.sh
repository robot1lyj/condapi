#!/usr/bin/env bash
# Server-only one-command resume. Select only an environment with an installation gate.
set -euo pipefail
session=lego-convert-v1
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Session already exists: $session; inspect its log instead of starting a duplicate"
  exit 0
fi
snapshot=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
shared_prefix=/home/wuyan/.conda/envs/condapi-yam
scratch_prefix=/tmp/condapi-yam-smoke.CzQI6oAj/env
if [[ -n ${YAM_ENV_PREFIX:-} ]]; then
  prefix=$YAM_ENV_PREFIX
elif [[ -x "$shared_prefix/bin/python" ]] && grep -qx INSTALL_COMPLETE /home/wuyan/lyj/YAM/env-transfer/install.log; then
  prefix=$shared_prefix
elif [[ -x "$scratch_prefix/bin/python" ]] && grep -qx INSTALL_COMPLETE /tmp/condapi-yam-smoke.CzQI6oAj/install.log; then
  prefix=$scratch_prefix
else
  echo 'No verified environment available; restore one and set YAM_ENV_PREFIX. Checkpoints remain on shared storage.' >&2
  exit 1
fi
[[ -x "$prefix/bin/python" ]]
printf -v command '%q ' taskset -c 1 nice -n 19 timeout 48h bash "$snapshot/scripts/run_yam_conversion.sh" \
  "$prefix" "$snapshot" /home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks/lego_sorting \
  /home/wuyan/lyj/YAM/YAM_data/processed/lego_lerobot_v1_20260907 \
  /home/wuyan/lyj/YAM/env-transfer/lego-clean-20260907/contract.json --resume
printf -v log_path '%q' "$snapshot/resume.log"
tmux new-session -d -s "$session" "$command >> $log_path 2>&1"
echo "RESUME_STARTED environment=$prefix log=$snapshot/resume.log"
