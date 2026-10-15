#!/bin/bash
# Authorized download only. No conversion, GPU use, service switch or cleanup.
set -euo pipefail
root=/home/wuyan-lyj/thor/pi
stage=$root/probes/rtc-separation-r1-10000-20261015-r1
destination=$root/checkpoints/lego-pi05-rtc-separation-r1-10k-20261014/10000
source_host=wuyan@10.18.31.234
source_dir=/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_separation_r1_10k_20261014/10000
control=/home/wuyan/lyj/YAM/training-runs/control/lego_pi05_rtc_separation_r1_10k_20261014
ssh_opts=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
mkdir -p "$stage" "$destination"
exec 9>"$stage/download.lock"
flock -n 9
date --iso-8601=seconds
echo SOURCE_CHECKSUM_SCAN
ssh "${ssh_opts[@]}" "$source_host" "cd '$source_dir' && test -f _CHECKPOINT_METADATA && test -f params/_METADATA && test -f assets/yam/norm_stats.json && find params assets -type f -print0 | sort -z | xargs -0 sha256sum && sha256sum _CHECKPOINT_METADATA" > "$stage/source.sha256.part"
for required in params/_METADATA assets/yam/norm_stats.json _CHECKPOINT_METADATA; do
  awk -v path="$required" '$2 == path {found=1} END {exit !found}' "$stage/source.sha256.part"
done
for name in training_contract.json initial_config.json; do
  ssh "${ssh_opts[@]}" "$source_host" "cd '$control' && sha256sum '$name'" >> "$stage/source.sha256.part"
done
mv "$stage/source.sha256.part" "$stage/source.sha256"
for shard in {0..5}; do
  awk -v shard="$shard" '$2 != "training_contract.json" && $2 != "initial_config.json" {if ((count++ % 6) == shard) print $2}' "$stage/source.sha256" > "$stage/shard-$shard.files"
done
echo DOWNLOAD_STARTED
pids=()
for shard in {0..5}; do
  rsync -a --partial --stats --timeout=120 --files-from="$stage/shard-$shard.files" -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6' "$source_host:$source_dir/" "$destination/" > "$stage/shard-$shard.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]]
for name in training_contract.json initial_config.json; do
  rsync -a --partial --timeout=120 -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15' "$source_host:$control/$name" "$destination/$name"
done
echo VERIFY_DOWNLOADED_FILES
(cd "$destination" && sha256sum -c "$stage/source.sha256")
python3 - "$destination" "$stage" <<'PY'
import datetime, hashlib, json, pathlib, sys
root, stage = map(pathlib.Path, sys.argv[1:])
contract = json.loads((root/'training_contract.json').read_text())
config = json.loads((root/'initial_config.json').read_text())
metadata = json.loads((root/'_CHECKPOINT_METADATA').read_text())
parent = '/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_50h_20260921/240000/params'
assert contract['parent_params'] == parent == config['weight_loader']['params_path']
assert contract['model']['rtc_training_max_delay'] == 10
assert contract['model']['action_horizon'] == 50
assert metadata['commit_timestamp_nsecs'] > 0
norm_sha = hashlib.sha256((root/'assets/yam/norm_stats.json').read_bytes()).hexdigest()
assert norm_sha == contract['norm_sha256']
files = list(p for p in root.rglob('*') if p.is_file())
receipt = {'status': 'downloaded_sha_verified_not_converted', 'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'parent_checkpoint': 240000, 'adaptation_step': 10000, 'rtc_training_max_delay': 10, 'norm_sha256': norm_sha, 'files': len(files), 'bytes': sum(p.stat().st_size for p in files), 'destination': str(root)}
(stage/'download-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(receipt, ensure_ascii=False))
PY
date --iso-8601=seconds
echo DOWNLOADED_ONLY_240000_PLUS_10000_NO_SERVICE_CHANGE
