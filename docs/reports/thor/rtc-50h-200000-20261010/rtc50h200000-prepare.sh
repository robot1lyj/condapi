#!/bin/bash
# Preparation only. Never stops, replaces or deletes the online 230000 model.
set -euo pipefail
root=/home/wuyan-lyj/thor/pi
stage=$root/probes/rtc-50h-200000-20261010-r1
cp_root=$root/checkpoints/lego-pi05-rtc-base-50h-20260921
source_host=wuyan@10.18.31.234
source_dir=/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_50h_20260921/200000
contract=/home/wuyan/lyj/YAM/training-runs/control/lego_pi05_rtc_base_50h_20260921/training_contract.json
image=openpi-pi:thor-trained-rtc-candidate-20260916
export JAX_PLATFORMS=cpu
if [[ ${1:-} == --download ]]; then
  mkdir -p "$cp_root/200000"
  if ! grep -q '  training_contract.json$' "$stage/source.sha256" 2>/dev/null; then
    ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$source_host" "cd '$source_dir' && test -f params/_METADATA && test -f assets/yam/norm_stats.json && find params assets -type f -print0 | sort -z | xargs -0 sha256sum; sha256sum '$source_dir/_CHECKPOINT_METADATA'" | sed "s|  $source_dir/|  |" > "$stage/source.sha256"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$source_host" "sha256sum '$contract'" | awk '{print $1 "  training_contract.json"}' >> "$stage/source.sha256"
  fi
  # OCDBT shard counts vary by checkpoint; check required entries, not an old count.
  for required in params/_METADATA assets/yam/norm_stats.json _CHECKPOINT_METADATA training_contract.json; do
    awk -v path="$required" '$2 == path {found=1} END {exit !found}' "$stage/source.sha256"
  done
  # Six independent large OCDBT files; stripe the small metadata alongside them.
  for shard in {0..5}; do
    awk -v shard="$shard" '($2 != "training_contract.json") && ((NR-1)%6 == shard) {print $2}' "$stage/source.sha256" > "$stage/download-$shard.files"
  done
  pids=()
  for shard in {0..5}; do
    rsync -a --partial --timeout=120 --files-from="$stage/download-$shard.files" -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ServerAliveInterval=30' "$source_host:$source_dir/" "$cp_root/200000/" > "$stage/download-$shard.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" == 0 ]]
  rsync -a --partial --timeout=120 -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes' "$source_host:$contract" "$cp_root/200000/training_contract.json"
  (cd "$cp_root/200000" && sha256sum -c "$stage/source.sha256")
  date --iso-8601=seconds
  echo DOWNLOADED_50H_200000
fi
test -f "$cp_root/200000/training_contract.json"
(cd "$cp_root/200000" && sha256sum -c "$stage/source.sha256")
# Bounded CPU conversion; no CUDA runtime and no online-container mutation.
docker run --rm --name pi05-rtc-50h200000-cpu-prepare --cpus 8 --memory 80g --network host --ipc host -e JAX_PLATFORMS=cpu -e OMP_NUM_THREADS=8 -v "$stage:/bench:ro" -v "$cp_root:/checkpoints" -v "$root/artifacts:/artifacts" --entrypoint /bin/bash "$image" -c 'python /bench/audit_checkpoint_finite.py --checkpoint /checkpoints/200000 --output /artifacts/rtc-50h-200000-audit-20261010-r1.json && python -c '\''import json; assert json.load(open("/artifacts/rtc-50h-200000-audit-20261010-r1.json"))["nonfinite_elements"] == 0'\'' && python /bench/prepare_rtc_checkpoint.py --checkpoint /checkpoints/200000 --training-contract /checkpoints/200000/training_contract.json --output /checkpoints/200000-pytorch-fp32-r1'
docker run --rm --cpus 8 --memory 32g --network host --ipc host -e JAX_PLATFORMS=cpu -e OMP_NUM_THREADS=8 -v "$stage:/bench:ro" -v "$cp_root:/checkpoints:ro" -v "$root/test-data:/data" --entrypoint python "$image" /bench/prepare_rtc_cases.py --source-suite /data/pi05-replay-v1 --source-parquet /data/ABC-130k-two-tasks/lego_sorting/train/data/source-file-000.parquet --checkpoint /checkpoints/200000 --output /data/pi05-rtc-50h-200000-replay-20261010-r1
date --iso-8601=seconds
echo PREPARED_50H_200000_ONLINE_230000_UNCHANGED
