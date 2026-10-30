"""Frozen, user-authorized Thor cleanup: DAgger 7000/10000 and earbuds 114000.

Only model payloads and three verified stopped containers are removed. Keep
norms, contracts, reports, replay/reference data, images and other checkpoints.
"""

import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/home/wuyan-lyj/thor/pi')
RECEIPT = ROOT / 'probes/rtc-50h-240000-20261014-r1/cleanup-retired-20261030.json'
MODELS = (
    ('lego-pi05-rtc-separation-r1-10k-20261014', '7000',
     'rtc-separation-r1-7000', '20261015-r2',
     'pi05-rtc-separation7000-preserved-20261015',
     'a99a3856b66e7a0815f1499007525422fcbcfb16afca107444fcefa787b737d3'),
    ('lego-pi05-rtc-separation-r1-10k-20261014', '10000',
     'rtc-separation-r1-10000', '20261015-r2',
     'pi05-rtc-separation10000-preserved-20261021',
     '908cf1c2b7d8a99fe3410dd84f8d01a4cdf1314b0036287e5db8899af99a4cd0'),
    ('earbuds-pi05-rtc-base-50h-20261012', '114000',
     'rtc-earbuds-114000', '20261022-r1',
     'pi05-rtc-earbuds114000-preserved-20261022',
     '6de9d18bb2385d3e5f15aeadb285aab4759967b1ec9b1fe37141b83486363528'),
)


def inspect(name):
    return json.loads(subprocess.check_output(['docker', 'inspect', name], text=True))[0]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_receipt(value):
    RECEIPT.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def protected_paths():
    paths = []
    for step, date in [('200000', '20261010-r1'), ('230000', '20261010-r1'),
                       ('240000', '20261014-r1')]:
        base = ROOT / 'checkpoints/lego-pi05-rtc-base-50h-20260921'
        paths.extend([base / step / 'params',
                      base / (step + '-pytorch-fp32-r1') / 'model.safetensors',
                      ROOT / f'artifacts/rtc-50h-{step}-onnx-fp32-cache80-7step-{date}/sampler.onnx.data',
                      ROOT / f'artifacts/rtc-50h-{step}-trt-tf32-cache80-7step-{date}/sampler.engine'])
    paths.extend([ROOT / 'checkpoints/pi05_base/params',
                  ROOT / 'checkpoints/pi05_base_pytorch_fp32_v1/model.safetensors'])
    return paths


def main():
    if os.geteuid() != 0 or 'Thor' not in Path('/proc/device-tree/model').read_text():
        raise RuntimeError('Run as root on Thor only')
    if RECEIPT.exists():
        raise FileExistsError('Frozen cleanup already attempted; inspect its receipt')
    keep = protected_paths()
    for path in keep:
        if not path.exists():
            raise FileNotFoundError(f'Protected asset missing before cleanup: {path}')
    current = inspect('pi05-rtc-infer')
    if '/artifacts/rtc-50h-240000-trt-tf32-cache80-7step-20261014-r1' not in current['Config']['Cmd']:
        raise RuntimeError('Unexpected current Pi model')
    running_names = subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True).splitlines()
    running = [inspect(name) for name in running_names]
    targets, containers = [], []
    for run, step, prefix, date, name, identifier in MODELS:
        container = inspect(name)
        if container['Id'] != identifier or container['State']['Running']:
            raise RuntimeError(f'Container identity/state changed: {name}')
        checkpoint = ROOT / 'checkpoints' / run
        converted = checkpoint / (step + '-pytorch-fp32-r1')
        if not any(m['Source'] == str(converted) for m in container['Mounts']):
            raise RuntimeError(f'Unexpected checkpoint mount: {name}')
        for active in running:
            if prefix in json.dumps(active['Config']['Cmd']) or any(
                m['Source'] in (str(checkpoint / step), str(converted)) for m in active['Mounts']
            ):
                raise RuntimeError(f'Target used by active container: {active["Name"]}')
        containers.append(dict(name=name, id=identifier))
        onnx = ROOT / 'artifacts' / f'{prefix}-onnx-fp32-cache80-7step-{date}'
        engine = ROOT / 'artifacts' / f'{prefix}-trt-tf32-cache80-7step-{date}'
        for path in [checkpoint / step / 'params', converted / 'model.safetensors',
                     onnx / 'sampler.onnx', onnx / 'sampler.onnx.data', engine / 'sampler.engine']:
            if not path.exists() or path.is_symlink() or path.resolve() != path:
                raise ValueError(f'Unexpected target: {path}')
            if not path.is_relative_to(ROOT) or path in keep or path == ROOT:
                raise ValueError(f'Unsafe target: {path}')
            members = list(path.rglob('*')) if path.is_dir() else [path]
            if any(f.is_symlink() for f in members):
                raise ValueError(f'Symlink inside target: {path}')
            files = [f for f in members if f.is_file()]
            targets.append(dict(path=str(path), files=len(files),
                                bytes=sum(f.stat().st_size for f in files),
                                allocated_bytes=sum(f.stat().st_blocks * 512 for f in files)))
    receipt = dict(status='validated_not_deleted', observed_at=now(), targets=targets,
                   containers=containers, permanent=True,
                   authorized_scope='2026-10-30: delete Thor DAgger 7000/10000 and earbuds 114000',
                   available_before=shutil.disk_usage(ROOT).free, completed=[],
                   retained='20w/23w/24w/base; norms/contracts/reports/replays/JAX references/images; server untouched')
    write_receipt(receipt)
    try:
        for container in containers:
            subprocess.run(['docker', 'rm', container['id']], check=True)
            receipt['completed'].append(container['name'])
            write_receipt(receipt)
        for row in targets:
            path = Path(row['path'])
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            receipt['completed'].append(str(path))
            write_receipt(receipt)
        if not all(path.exists() for path in keep):
            raise RuntimeError('Protected asset verification failed')
        if any(Path(row['path']).exists() for row in targets):
            raise RuntimeError('Target remains after deletion')
        receipt.update(status='deleted_verified', completed_at=now(),
                       available_after=shutil.disk_usage(ROOT).free,
                       deleted_allocated_bytes=sum(r['allocated_bytes'] for r in targets))
    except Exception as error:
        receipt.update(status='failed_partial_inspect_completed', error=str(error))
        raise
    finally:
        write_receipt(receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
