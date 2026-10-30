"""Bind unchanged held-out earbud observations/absolute prefixes to the new norm."""

import json
from pathlib import Path
import shutil

from prepare_checkpoint_suite import digest, prepare
from rtc_norm_identity import checkpoint_norm_identity

ROOT = Path('/home/wuyan-lyj/thor/pi')
source = ROOT / 'test-data/pi05-rtc-earbuds-114000-replay-20261022-r1'
output = ROOT / 'test-data/pi05-rtc-earbuds-clean-v2-62000-replay-20261030-r1'
checkpoint = ROOT / 'checkpoints/earbuds-pi05-rtc-clean-v2-excl-pantheon-retry4-20261022/62000'
contract = json.loads((checkpoint / 'training_contract.json').read_text())
suite = json.loads((source / 'suite.json').read_text())
if suite['source_kind'] != 'real_yam_recording':
    raise ValueError('Real replay required')
if suite['source_root'] != str(Path(contract['repo_id']).parent / 'val'):
    raise ValueError('Held-out dataset changed; collect fresh observations instead')
if contract['model']['rtc_training_max_delay'] != 10:
    raise ValueError('Unexpected RTC training contract')
norm = checkpoint / 'assets/yam/norm_stats.json'
identity = checkpoint_norm_identity(norm, contract['norm_sha256'])
prepare(source, checkpoint, output)
cases = json.loads((source / 'cases.json').read_text())
if {case['delay_steps'] for case in cases['cases']} != {0, 1, 10} or len(cases['cases']) != 9:
    raise ValueError('Expected nine real d=0/1/10 cases')
for case in cases['cases']:
    prefix = (source / case['committed_actions']).resolve()
    if not prefix.is_relative_to(source) or prefix.name != case['committed_actions']:
        raise ValueError('Prefix path escaped real replay directory')
    if digest(prefix) != case['committed_actions_sha256']:
        raise ValueError('Original real prefix changed')
    shutil.copy2(prefix, output / prefix.name)
    provenance = json.loads((output / case['provenance']).read_text())
    if provenance['norm_stats_sha256'] != digest(norm):
        raise ValueError('Norm identity mismatch')
    if provenance['sample_sha256'] != digest(output / case['sample']):
        raise ValueError('Observation changed')
cases.update(norm_stats_sha256=digest(norm), norm_identity=identity,
             checkpoint_metadata_sha256=digest(checkpoint / 'params/_METADATA'),
             parent_cases_sha256=digest(source / 'cases.json'))
(output / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
print(json.dumps(dict(status='rebound_same_real_val_arrays_new_checkpoint_norm',
                      output=str(output), cases=9, norm_identity=identity)), flush=True)
