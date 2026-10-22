#!/bin/bash
# Frozen handoff: latest complete earbuds checkpoint observed on 2026-10-22.
set -euo pipefail
root=/home/wuyan-lyj/thor/pi
stage=$root/probes/rtc-earbuds-114000-20261022-r1
destination=$root/checkpoints/earbuds-pi05-rtc-base-50h-20261012/114000
source_host=wuyan@10.18.31.234
source_dir=/home/wuyan/lyj/YAM/training-runs/pi05_yam/earbuds_pi05_rtc_base_50h_20261012/114000
control=/home/wuyan/lyj/YAM/training-runs/control/earbuds_pi05_rtc_base_50h_20261012
ssh_opts=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
mkdir -p "$stage" "$destination"
exec 9>"$stage/download.lock"
flock -n 9
date --iso-8601=seconds
ssh "${ssh_opts[@]}" "$source_host" "cd '$source_dir' && test -f _CHECKPOINT_METADATA && test -f params/_METADATA && test -f assets/yam/norm_stats.json && find params assets -type f -print0 | sort -z | xargs -0 sha256sum && sha256sum _CHECKPOINT_METADATA" > "$stage/source.sha256.part"
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
(cd "$destination" && sha256sum -c "$stage/source.sha256")
python3 - "$destination" "$stage" <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import sys

root, stage = map(Path, sys.argv[1:])
contract = json.loads((root / 'training_contract.json').read_text())
config = json.loads((root / 'initial_config.json').read_text())
metadata = json.loads((root / '_CHECKPOINT_METADATA').read_text())
assert config['exp_name'] == 'earbuds_pi05_rtc_base_50h_20261012'
assert contract['model']['rtc_training_max_delay'] == 10
assert contract['model']['action_horizon'] == 50
assert contract['model']['pi05'] is True
assert metadata['commit_timestamp_nsecs'] > 0
norm_data = (root / 'assets/yam/norm_stats.json').read_bytes()
norm_sha = hashlib.sha256(norm_data).hexdigest()
# Orbax may omit exactly one final newline when saving training norm assets.
assert norm_sha == contract['norm_sha256'] or (
    not norm_data.endswith(b'\n')
    and hashlib.sha256(norm_data + b'\n').hexdigest() == contract['norm_sha256'])
files = [p for p in root.rglob('*') if p.is_file()]
receipt = dict(status='downloaded_sha_verified_not_converted',
               observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               checkpoint_step=114000, rtc_training_max_delay=10,
               norm_sha256=norm_sha, files=len(files),
               bytes=sum(p.stat().st_size for p in files), destination=str(root))
(stage / 'download-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
PY
date --iso-8601=seconds
echo DOWNLOADED_EARBUDS_114000
