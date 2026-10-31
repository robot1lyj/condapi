"""Frozen user-authorized project cleanup; never prune or touch protected jobs.

Remove four exact project roots and dedicated containers/images. Separately
remove Pi-derived stopped containers except Lego 240000 and earbuds 62000.
Pi model files, shared images, dubbing, perception and Flora remain untouched.
"""

import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess

HOME_ROOT = Path('/home/wuyan-lyj')
ROOTS = (
    HOME_ROOT / 'thor/rl-training', HOME_ROOT / 'comfyui-h3',
    HOME_ROOT / 'thor/xr1', HOME_ROOT / 'thor/openwam',
)
STAGE = HOME_ROOT / 'thor/pi/probes/storage-cleanup-20261031'
RECEIPT = STAGE / 'receipt.json'
FAMILIES = ('comfyui-h3:', 'xiaomi-xr1:', 'openwam:', 'pld-rl:', 'pld-rl-train:')
KEEP_PI = 'd80c40497577a3028a4ca7bc16fd49c861c0c3893937df62f291e5887b5ef523'


def command(*args):
    return subprocess.check_output(args, text=True)


def inspect_all():
    ids = command('docker', 'ps', '-aq').splitlines()
    return json.loads(command('docker', 'inspect', *ids)) if ids else []


def under(path, root):
    return path == str(root) or path.startswith(str(root) + '/')


def touches(c):
    return any(under(m['Source'], root) for m in c['Mounts'] for root in ROOTS)


def snapshot(c):
    return dict(name=c['Name'].lstrip('/'), id=c['Id'], running=c['State']['Running'],
                started=c['State']['StartedAt'], image_id=c['Image'], image=c['Config']['Image'])


def persist(receipt):
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')


def protected_files():
    pi = HOME_ROOT / 'thor/pi'
    files = []
    for step, date in [('200000', '20261010-r1'), ('230000', '20261010-r1'),
                       ('240000', '20261014-r1')]:
        cp = pi / 'checkpoints/lego-pi05-rtc-base-50h-20260921'
        files.extend([cp / step / 'params/_METADATA',
                      cp / (step + '-pytorch-fp32-r1') / 'model.safetensors',
                      pi / f'artifacts/rtc-50h-{step}-trt-tf32-cache80-7step-{date}/sampler.engine'])
    cp = pi / 'checkpoints/earbuds-pi05-rtc-clean-v2-excl-pantheon-retry4-20261022'
    files.extend([cp / '62000/params/_METADATA', cp / '62000-pytorch-fp32-r1/model.safetensors',
                  pi / 'artifacts/rtc-earbuds-clean-v2-62000-trt-tf32-cache80-7step-20261030-r1/sampler.engine',
                  pi / 'checkpoints/pi05_base/params/_METADATA',
                  pi / 'checkpoints/pi05_base_pytorch_fp32_v1/model.safetensors'])
    return {str(p): (p.stat().st_ino, p.stat().st_size) for p in files}


def main():
    if os.geteuid() != 0 or 'Thor' not in Path('/proc/device-tree/model').read_text():
        raise RuntimeError('Run as root on Thor only')
    if RECEIPT.exists():
        raise FileExistsError('Frozen cleanup already attempted; inspect its receipt')
    for root in ROOTS:
        if not root.is_dir() or root.is_symlink() or root.resolve() != root:
            raise ValueError(f'Unexpected project root: {root}')
    mounts = command('findmnt', '-rn', '-o', 'TARGET').splitlines()
    if any(under(m, root) for m in mounts for root in ROOTS):
        raise RuntimeError('Nested mounted filesystem under a deletion target')
    expected = protected_files()
    containers = inspect_all()
    pi_ids = command('docker', 'image', 'ls', 'openpi-pi', '-q', '--no-trunc').splitlines()
    image_ids = sorted(set(command('docker', 'image', 'ls', '-q', '--no-trunc').splitlines()))
    images = json.loads(command('docker', 'image', 'inspect', *image_ids))
    family_ids = {i['Id'] for i in images if any(t.startswith(FAMILIES) for t in i.get('RepoTags', []))}
    selected, retained_pi = [], []
    for c in containers:
        image = c['Config']['Image']
        pi_related = image.startswith('openpi-pi:') or c['Image'] in pi_ids
        is_62000 = any('earbuds-pi05-rtc-clean-v2-excl-pantheon-retry4-20261022/62000' in m['Source']
                       for m in c['Mounts'])
        if pi_related and (c['Id'] == KEEP_PI or is_62000):
            retained_pi.append(snapshot(c))
            continue
        if touches(c) or image.startswith(FAMILIES) or c['Image'] in family_ids or pi_related:
            if c['State']['Running'] and c['Name'] != '/comfyui-h3':
                raise RuntimeError(f'Unexpected active deletion target: {c["Name"]}')
            selected.append(c)
    if not any(c['id'] == KEEP_PI for c in retained_pi):
        raise RuntimeError('Protected Lego 240000 container missing')
    selected_ids = {c['Id'] for c in selected}
    active = [c for c in containers if c['State']['Running'] and c['Id'] not in selected_ids]
    for c in active:
        for m in c['Mounts']:
            if any(under(str(root), Path(m['Source'])) or under(m['Source'], root) for root in ROOTS):
                raise RuntimeError(f'Protected active mount overlaps target: {c["Name"]}')
    for name in ('/yam-agent-perception', '/flora-annotation'):
        if not any(c['Name'] == name for c in active):
            raise RuntimeError(f'Expected protected job not running: {name}')
    image_plan = [dict(id=i['Id'], tag=t) for i in images for t in i.get('RepoTags', [])
                  if t.startswith(FAMILIES) and i['Id'] not in {c['Image'] for c in active}]
    receipt = dict(status='validated_not_deleted', started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   authorized_scope='2026-10-31: delete rl-training, comfyui-h3, xr1, openwam; remove other Pi stopped containers only',
                   permanent=True, roots=[dict(path=str(p), allocated_bytes=int(command('du', '-sxB1', str(p)).split()[0])) for p in ROOTS],
                   containers=[snapshot(c) for c in selected], image_tags=image_plan,
                   protected_active=[snapshot(c) for c in active], retained_pi=retained_pi,
                   protected_pi_files=expected, completed=[], image_skipped=[],
                   available_before=shutil.disk_usage(HOME_ROOT).free)
    STAGE.mkdir(parents=True, exist_ok=True)
    persist(receipt)
    try:
        for c in selected:
            if c['State']['Running']:
                subprocess.run(['docker', 'stop', '--timeout', '30', c['Id']], check=True)
            subprocess.run(['docker', 'rm', c['Id']], check=True)
            receipt['completed'].append(dict(container=c['Name'], id=c['Id']))
            persist(receipt)
        current = inspect_all()
        if any(touches(c) for c in current):
            raise RuntimeError('New container references a deletion root; do not remove files')
        for c in active:
            now = next(x for x in current if x['Id'] == c['Id'])
            if not now['State']['Running'] or now['State']['StartedAt'] != c['State']['StartedAt']:
                raise RuntimeError(f'Protected job state changed: {c["Name"]}')
        for root in ROOTS:
            shutil.rmtree(root)
            if root.exists():
                raise RuntimeError(f'Deletion incomplete: {root}')
            receipt['completed'].append(dict(directory=str(root)))
            persist(receipt)
        for image in image_plan:
            result = subprocess.run(['docker', 'image', 'rm', image['tag']], capture_output=True, text=True)
            if result.returncode:
                receipt['image_skipped'].append(image | dict(reason=result.stderr.strip()))
            else:
                receipt['completed'].append(dict(image_tag=image['tag'], image_id=image['id']))
            persist(receipt)
        if protected_files() != expected:
            raise RuntimeError('Protected Pi files changed')
        current = inspect_all()
        for c in active:
            now = next(x for x in current if x['Id'] == c['Id'])
            if not now['State']['Running'] or now['State']['StartedAt'] != c['State']['StartedAt']:
                raise RuntimeError(f'Protected job changed: {c["Name"]}')
        receipt.update(status='deleted_roots_and_containers_verified',
                       completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       available_after=shutil.disk_usage(HOME_ROOT).free,
                       remaining_containers=[snapshot(c) for c in current])
    except Exception as error:
        receipt.update(status='failed_partial_inspect_completed', error=str(error))
        raise
    finally:
        persist(receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'available_before', 'available_after', 'image_skipped')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
