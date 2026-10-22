"""User-authorized removal of only Thor's retired earbuds 112000 model assets.

Run only after 114000 is online and its real protocol smoke has passed. Small
historical receipts/replay/JAX reference outputs and shared Pi images stay intact.
"""

import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/home/wuyan-lyj/thor/pi')
STAGE = ROOT / 'probes/rtc-earbuds-114000-20261022-r1'
OLD_ID = 'dc2e00ad1e2581769e2b8aad832f9b1fc75aad122f32a1c60a665fecaa8d5221'
OLD_NAME = 'pi05-rtc-earbuds112000-preserved-20261022'
OLD_WEIGHTS = '1751a7252fcd8d45027c791f36576f1c43b1783c4b752a0354b738eadec2b979'
TARGETS = (
    ROOT / 'checkpoints/earbuds-pi05-rtc-base-50h-20261012/112000',
    ROOT / 'checkpoints/earbuds-pi05-rtc-base-50h-20261012/112000-pytorch-fp32-r1',
    ROOT / 'artifacts/rtc-earbuds-112000-onnx-fp32-cache80-7step-20261021-r1',
    ROOT / 'artifacts/rtc-earbuds-112000-trt-tf32-cache80-7step-20261021-r1',
)


def inspect(name):
    return json.loads(subprocess.check_output(['docker', 'inspect', name], text=True))[0]


def main():
    device = Path('/proc/device-tree/model')
    if os.geteuid() != 0 or not device.is_file() or 'Thor' not in device.read_text():
        raise RuntimeError('This cleanup must run as root on the verified Thor host')
    receipt_path = STAGE / 'cleanup-112000-receipt.json'
    if receipt_path.exists():
        raise FileExistsError('Do not repeat the frozen cleanup')
    manifest = json.loads((ROOT / 'checkpoints/earbuds-pi05-rtc-base-50h-20261012'
                           '/114000-pytorch-fp32-r1/rtc_manifest.json').read_text())
    smoke = json.loads((STAGE / 'earbuds114000-online-smoke.json').read_text())
    assert smoke['status'] == 'passed'
    assert smoke['metadata']['checkpoint_weights_sha256'] == manifest['model_weights_sha256'] != OLD_WEIGHTS
    assert smoke['metadata']['rtc_mode'] == 'trained'
    assert smoke['metadata']['rtc_prefix_norm'] == 'quantile'
    assert len(smoke['cases']) == 9 and all(row['output_finite'] and row['prefix_exact'] for row in smoke['cases'])
    current = inspect('pi05-rtc-infer')
    assert current['State']['Running'] and current['Id'] != OLD_ID
    assert '/artifacts/rtc-earbuds-114000-trt-tf32-cache80-7step-20261022-r1' in current['Config']['Cmd']
    assert any(mount['Source'] == str(ROOT / 'checkpoints/earbuds-pi05-rtc-base-50h-20261012'
                                           '/114000-pytorch-fp32-r1') for mount in current['Mounts'])
    assert inspect(OLD_NAME)['Id'] == OLD_ID and not inspect(OLD_NAME)['State']['Running']
    assert subprocess.run(['systemctl', 'is-active', '--quiet',
                           'thor-pi-maxn-earbuds112000-20261021.service']).returncode != 0
    subprocess.run(['curl', '-fsS', '--max-time', '3', 'http://192.168.250.1:8000/healthz'], check=True)
    rows = []
    for path in TARGETS:
        if not path.is_dir() or path.is_symlink() or path.resolve() != path:
            raise ValueError(f'Unexpected deletion target: {path}')
        assert path != ROOT and path.is_relative_to(ROOT)
        files = [member for member in path.rglob('*') if member.is_file() and not member.is_symlink()]
        rows.append(dict(path=str(path), file_bytes=sum(member.stat().st_size for member in files),
                         allocated_bytes=sum(member.stat().st_blocks * 512 for member in files)))
    receipt = dict(status='validated_not_yet_deleted',
                   authorized_scope='2026-10-22 user: remove Thor earbuds 112000 inference, replace by 114000',
                   observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   new_weights_sha256=manifest['model_weights_sha256'], old_container_id=OLD_ID,
                   targets=rows, filesystem_free_before=shutil.disk_usage(ROOT).free,
                   deletion_is_permanent=True,
                   retained='server originals untouched; shared images, DAgger models, small replay/reference/receipts retained')
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    subprocess.run(['docker', 'rm', OLD_ID], check=True)
    for path in TARGETS:
        shutil.rmtree(path)
        assert not path.exists()
    receipt.update(status='deleted_only_retired_thor_earbuds112000',
                   filesystem_free_after=shutil.disk_usage(ROOT).free,
                   completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
