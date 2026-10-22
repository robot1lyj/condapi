"""Verify this frozen download, allowing only the known Orbax newline difference."""

import datetime
import hashlib
import json
from pathlib import Path

from rtc_norm_identity import checkpoint_norm_identity

root = Path('/home/wuyan-lyj/thor/pi/checkpoints/earbuds-pi05-rtc-base-50h-20261012/114000')
stage = Path('/home/wuyan-lyj/thor/pi/probes/rtc-earbuds-114000-20261022-r1')
entries = []
for line in (stage / 'source.sha256').read_text().splitlines():
    expected, relative = line.split(maxsplit=1)
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Checksum path outside checkpoint')
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise ValueError(f'Incomplete or mismatched download: {relative}')
    entries.append(path)
for relative in ('_CHECKPOINT_METADATA', 'params/_METADATA', 'assets/yam/norm_stats.json',
                 'training_contract.json', 'initial_config.json'):
    if (root / relative) not in entries:
        raise ValueError(f'Missing required download member: {relative}')
contract = json.loads((root / 'training_contract.json').read_text())
config = json.loads((root / 'initial_config.json').read_text())
metadata = json.loads((root / '_CHECKPOINT_METADATA').read_text())
assert metadata['commit_timestamp_nsecs'] > 0
assert config['exp_name'] == 'earbuds_pi05_rtc_base_50h_20261012'
assert contract['model']['rtc_training_max_delay'] == 10
assert contract['model']['action_horizon'] == 50
assert contract['model']['pi05'] is True
identity = checkpoint_norm_identity(root / 'assets/yam/norm_stats.json', contract['norm_sha256'])
receipt = dict(status='downloaded_sha_verified_not_converted', checkpoint_step=114000,
               observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               rtc_training_max_delay=10, norm_identity=identity,
               norm_stats_sha256=identity['checkpoint_norm_sha256'],
               files=len(entries), bytes=sum(p.stat().st_size for p in entries), destination=str(root))
(stage / 'download-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
