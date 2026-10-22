"""Rebind unchanged real earbud observations/prefixes to checkpoint 114000."""

import json
from pathlib import Path
import shutil

from prepare_checkpoint_suite import digest, prepare
from rtc_norm_identity import checkpoint_norm_identity

root = Path('/home/wuyan-lyj/thor/pi')
source = root / 'test-data/pi05-rtc-earbuds-112000-replay-20261021-r1'
output = root / 'test-data/pi05-rtc-earbuds-114000-replay-20261022-r1'
checkpoint = root / 'checkpoints/earbuds-pi05-rtc-base-50h-20261012/114000'
contract = json.loads((checkpoint / 'training_contract.json').read_text())
suite = json.loads((source / 'suite.json').read_text())
assert suite['source_kind'] == 'real_yam_recording'
assert suite['source_root'] == str(Path(contract['repo_id']).parent / 'val')
assert contract['model']['rtc_training_max_delay'] == 10
norm = checkpoint / 'assets/yam/norm_stats.json'
identity = checkpoint_norm_identity(norm, contract['norm_sha256'])
prepare(source, checkpoint, output)
cases = json.loads((source / 'cases.json').read_text())
assert {case['delay_steps'] for case in cases['cases']} == {0, 1, 10}
assert len(cases['cases']) == 9
for case in cases['cases']:
    prefix = (source / case['committed_actions']).resolve()
    if not prefix.is_relative_to(source) or prefix.name != case['committed_actions']:
        raise ValueError('Prefix path escaped real replay directory')
    if digest(prefix) != case['committed_actions_sha256']:
        raise ValueError('Original real prefix changed')
    shutil.copy2(prefix, output / prefix.name)
    provenance = json.loads((output / case['provenance']).read_text())
    assert provenance['norm_stats_sha256'] == digest(norm)
    assert provenance['sample_sha256'] == digest(output / case['sample'])
cases.update(norm_stats_sha256=digest(norm), norm_identity=identity,
             checkpoint_metadata_sha256=digest(checkpoint / 'params/_METADATA'),
             parent_cases_sha256=digest(source / 'cases.json'))
(output / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
print(json.dumps(dict(status='rebound_real_replay_no_arrays_modified', output=str(output), cases=9)), flush=True)
