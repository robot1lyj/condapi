#!/bin/bash
# User's revised scope: download/checksums ONLY; never convert, test or serve.
set -euo pipefail
root=/home/wuyan-lyj/thor/pi
stage=$root/probes/rtc-50h-240000-20261013-r1
destination=$root/checkpoints/lego-pi05-rtc-base-50h-20260921/240000
source_host=wuyan@10.18.31.234
source_dir=/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_50h_20260921/240000
contract=/home/wuyan/lyj/YAM/training-runs/control/lego_pi05_rtc_base_50h_20260921/training_contract.json
# Reuse the already-running checksum scan. Its parent was SIGSTOP'ed before
# any transfers or CPU/GPU stages, and must never be resumed.
while [[ -d /proc/2709014 ]]; do
  hash_state=$(ps -o stat= -p 2709014 || true)
  [[ "$hash_state" == Z* ]] && break
  sleep 2
done
for required in params/_METADATA assets/yam/norm_stats.json _CHECKPOINT_METADATA; do
  awk -v path="$required" '$2 == path {found=1} END {exit !found}' "$stage/source.sha256.part"
done
cp "$stage/source.sha256.part" "$stage/source-download-only.sha256.part"
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$source_host" "sha256sum '$contract'" | awk '{print $1 "  training_contract.json"}' >> "$stage/source-download-only.sha256.part"
mv "$stage/source-download-only.sha256.part" "$stage/source.sha256"
# Terminate only our frozen orchestration shell, not a data-transfer process.
python3 - <<'PY'
from pathlib import Path
import os, signal
pid = 2709011
proc = Path('/proc') / str(pid)
if proc.exists():
    args = (proc / 'cmdline').read_bytes().decode().strip('\0').split('\0')
    expected = '/home/wuyan-lyj/thor/pi/probes/rtc-50h-240000-20261013-r1/rtc50h240000-prepare.sh'
    if len(args) < 2 or args[0].split('/')[-1] != 'bash' or args[1] != expected:
        raise RuntimeError('Refuse to signal a reused/unexpected process')
    os.kill(pid, signal.SIGKILL)
PY
for shard in {0..5}; do
  awk -v shard="$shard" '($2 != "training_contract.json") && ((NR-1)%6 == shard) {print $2}' "$stage/source.sha256" > "$stage/download-only-$shard.files"
done
pids=()
for shard in {0..5}; do
  rsync -a --partial --timeout=120 --files-from="$stage/download-only-$shard.files" -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ServerAliveInterval=30' "$source_host:$source_dir/" "$destination/" > "$stage/download-only-$shard.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]]
rsync -a --partial --timeout=120 -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes' "$source_host:$contract" "$destination/training_contract.json"
(cd "$destination" && sha256sum -c "$stage/source.sha256")
date --iso-8601=seconds
echo DOWNLOADED_ONLY_50H_240000_WAIT_FOR_USER_AUTHORIZATION
