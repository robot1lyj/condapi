#!/bin/bash
# Resume ONLY final small files and verification after forwarded SSH timeout.
set -euo pipefail
root=/home/wuyan-lyj/thor/pi
stage=$root/probes/rtc-separation-r1-7000-20261015-r1
destination=$root/checkpoints/lego-pi05-rtc-separation-r1-10k-20261014/7000
source_host=wuyan@10.18.31.234
control=/home/wuyan/lyj/YAM/training-runs/control/lego_pi05_rtc_separation_r1_10k_20261014
# Wait for the original writer to finish naturally; never stop a live transfer.
exec 9>"$stage/download.lock"
flock -w 180 9
date --iso-8601=seconds
for name in training_contract.json initial_config.json; do
  rsync -a --partial --timeout=60 -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3' "$source_host:$control/$name" "$destination/$name"
done
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
receipt = {'status': 'downloaded_sha_verified_not_converted', 'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'parent_checkpoint': 240000, 'adaptation_step': 7000, 'rtc_training_max_delay': 10, 'norm_sha256': norm_sha, 'files': len(files), 'bytes': sum(p.stat().st_size for p in files), 'destination': str(root)}
(stage/'download-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(receipt, ensure_ascii=False))
PY
date --iso-8601=seconds
echo DOWNLOADED_ONLY_240000_PLUS_7000_NO_SERVICE_CHANGE
