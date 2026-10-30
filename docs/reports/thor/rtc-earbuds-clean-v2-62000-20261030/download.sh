#!/bin/bash
# Frozen download-only handoff. No conversion, inference or training commands.
set -euo pipefail
root=/home/wuyan-lyj/thor/pi
stage=$root/probes/rtc-earbuds-clean-v2-62000-20261030-r1
destination=$root/checkpoints/earbuds-pi05-rtc-clean-v2-excl-pantheon-retry4-20261022/62000
source_host=wuyan@10.18.31.234
run=earbuds_pi05_rtc_base_50h_clean_v2_excl_pantheon_retry4_20261022
source_dir=/home/wuyan/lyj/YAM/training-runs/pi05_yam/$run/62000
control=/home/wuyan/lyj/YAM/training-runs/control/$run
ssh_opts=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
mkdir -p "$stage" "$destination"
exec 9>"$stage/download.lock"
flock -n 9
test ! -f "$stage/download-receipt.json"
date --iso-8601=seconds
ssh "${ssh_opts[@]}" "$source_host" "cd '$source_dir' && test -f _CHECKPOINT_METADATA && test -f params/_METADATA && test -f assets/yam/norm_stats.json && find params assets -type f -print0 | sort -z | xargs -0 sha256sum && sha256sum _CHECKPOINT_METADATA" > "$stage/source.sha256.part"
for name in training_contract.json initial_config.json; do
  ssh "${ssh_opts[@]}" "$source_host" "cd '$control' && sha256sum '$name'" >> "$stage/source.sha256.part"
done
mv "$stage/source.sha256.part" "$stage/source.sha256"
for shard in {0..5}; do
  awk -v shard="$shard" '$2 != "training_contract.json" && $2 != "initial_config.json" {if ((count++ % 6) == shard) print $2}' "$stage/source.sha256" > "$stage/shard-$shard.files"
done
echo DOWNLOAD_STARTED_ONLY_EARBUDS_CLEAN_V2_62000
pids=()
for shard in {0..5}; do
  rsync -a --partial --stats --timeout=120 --files-from="$stage/shard-$shard.files" -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4' "$source_host:$source_dir/" "$destination/" > "$stage/shard-$shard.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]]
for name in training_contract.json initial_config.json; do
  rsync -a --partial --timeout=120 -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15' "$source_host:$control/$name" "$destination/$name"
done
(cd "$destination" && sha256sum -c "$stage/source.sha256")
python3 - "$destination" "$stage" "$run" <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import sys

root, stage = map(Path, sys.argv[1:3])
contract = json.loads((root / 'training_contract.json').read_text())
config = json.loads((root / 'initial_config.json').read_text())
metadata = json.loads((root / '_CHECKPOINT_METADATA').read_text())
if config['exp_name'] != sys.argv[3]:
    raise ValueError('Wrong training run')
if contract['model']['rtc_training_max_delay'] != 10 or contract['model']['action_horizon'] != 50:
    raise ValueError('Unexpected RTC contract')
if contract['model']['pi05'] is not True or metadata['commit_timestamp_nsecs'] <= 0:
    raise ValueError('Incomplete checkpoint or unexpected model')
data = (root / 'assets/yam/norm_stats.json').read_bytes()
norm_sha = hashlib.sha256(data).hexdigest()
if norm_sha != contract['norm_sha256'] and not (
    not data.endswith(b'\n') and hashlib.sha256(data + b'\n').hexdigest() == contract['norm_sha256']
):
    raise ValueError('Checkpoint norm does not match its own training contract')
files = [p for p in root.rglob('*') if p.is_file()]
receipt = dict(status='downloaded_sha_verified_not_converted',
               observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               checkpoint_step=62000, source_run=sys.argv[3],
               initial_params_path=config['weight_loader']['params_path'],
               rtc_training_max_delay=10, norm_stats_sha256=norm_sha,
               training_norm_sha256=contract['norm_sha256'], files=len(files),
               bytes=sum(p.stat().st_size for p in files), destination=str(root),
               not_downloaded='train_state/optimizer; no conversion/inference/training')
(stage / 'download-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
PY
date --iso-8601=seconds
echo DOWNLOADED_ONLY_EARBUDS_CLEAN_V2_62000
