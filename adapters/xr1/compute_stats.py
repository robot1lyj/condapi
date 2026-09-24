"""Stream XR-1 normalization from an immutable derived train manifest.

Uses Xiaomi's fixed rotm2aa_batch implementation and complete 30-step windows,
matching the native compute_normalize geometry without retaining all windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from adapters.xr1.common import VENDOR
from adapters.xr1.common import validate_stats

SLOTS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15)
HORIZON = 30


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _vector(traj: dict, group: str, key: str, columns: int, count: int) -> np.ndarray:
    value = np.asarray(traj[group][key], dtype=np.float32)
    if value.shape != (count, columns) or not np.isfinite(value).all():
        raise ValueError(f"invalid {group}.{key}")
    return value


def _actions(traj: dict, rotm2aa_batch, block_size: int = 128):
    count = traj["num_frames"]
    if count < HORIZON:
        raise ValueError("XR-1 training episode shorter than 30 frames")
    arms = [
        (
            _vector(traj, "proprios", f"{name}_ee_pos", 3, count),
            _vector(traj, "proprios", f"{name}_ee_rotm", 9, count).reshape(count, 3, 3),
            _vector(traj, "proprios", f"{name}_gripper_pos", 1, count),
            _vector(traj, "actions", f"{name}_ee_pos", 3, count),
            _vector(traj, "actions", f"{name}_ee_rotm", 9, count).reshape(count, 3, 3),
            _vector(traj, "actions", f"{name}_gripper_pos", 1, count),
        )
        for name in ("left", "right")
    ]
    steps = np.arange(HORIZON)
    for start in range(0, count - HORIZON + 1, block_size):
        stop = min(start + block_size, count - HORIZON + 1)
        indices = np.arange(start, stop)[:, None] + steps
        block = np.zeros((stop - start, HORIZON, 60), dtype=np.float32)
        for arm, (obs_pos, obs_rot, obs_grip, act_pos, act_rot, act_grip) in enumerate(arms):
            base_pos = obs_pos[start:stop]
            base_rot_t = np.swapaxes(obs_rot[start:stop], -1, -2)
            local_pos = np.matmul(base_rot_t[:, None], (act_pos[indices] - base_pos[:, None])[..., None])[..., 0]
            local_rot = np.matmul(base_rot_t[:, None], act_rot[indices])
            axis_angle = rotm2aa_batch(local_rot.reshape(-1, 3, 3)).reshape(stop - start, HORIZON, 3)
            offset = arm * 8
            block[:, :, offset : offset + 3] = local_pos
            block[:, :, offset + 3 : offset + 6] = axis_angle
            block[:, :, offset + 6 : offset + 7] = act_grip[indices] - obs_grip[start:stop, None]
        yield block


def compute(manifest_path: Path, output: Path, xr1_source: Path) -> dict:
    manifest_path, output, xr1_source = (Path(p).resolve() for p in (manifest_path, output, xr1_source))
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "yam_xr1_lego_eef_v1" or manifest.get("split") != "train":
        raise ValueError("normalization requires a complete 50h train manifest")
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or len(entries) != len(manifest["episode_ids"]):
        raise ValueError("derived train manifest is incomplete")
    upstream = json.loads((VENDOR / "UPSTREAM.json").read_text())
    if digest(xr1_source / "mibot/utils/io.py") != upstream["sha256"]["mibot/utils/io.py"]:
        raise ValueError("XR-1 action codec differs from selected upstream revision")
    sys.path.insert(0, str(xr1_source))
    from mibot.utils.io import rotm2aa_batch  # noqa: PLC0415

    total = 0
    moments = np.zeros((HORIZON, 60), dtype=np.float64)
    squares = np.zeros_like(moments)
    states = []
    hashes = {}
    for position, entry in enumerate(entries):
        path = Path(entry["json"]).resolve()
        if not path.is_file() or digest(path) != entry["json_sha256"]:
            raise ValueError(f"derived JSON changed: {path}")
        traj = json.loads(path.read_text())
        count = traj["num_frames"]
        if count != entry["frames"] or entry["episode_index"] != manifest["episode_ids"][position]:
            raise ValueError(f"derived episode identity mismatch: {path}")
        parts = []
        for arm in ("left", "right"):
            parts.append(_vector(traj, "proprios", f"{arm}_arm_joint", 6, count))
            parts.append(_vector(traj, "proprios", f"{arm}_gripper_pos", 1, count))
        states.append(np.concatenate(parts, axis=1))
        for block in _actions(traj, rotm2aa_batch):
            total += len(block)
            moments += block.sum(axis=0, dtype=np.float64)
            squares += np.square(block, dtype=np.float64).sum(axis=0)
        hashes[str(path)] = entry["json_sha256"]
        if (position + 1) % 100 == 0:
            print(json.dumps({"episodes": position + 1, "total": len(entries), "complete_windows": total}), flush=True)
    if not total:
        raise ValueError("no complete XR-1 action windows")
    mean = moments / total
    variance = np.maximum(squares / total - np.square(mean), 0)
    active = np.concatenate(states, axis=0)
    quantiles = np.quantile(active, [0.01, 0.99], axis=0)
    q01 = np.zeros((1, 60), dtype=np.float64)
    q99 = np.zeros((1, 60), dtype=np.float64)
    q01[0, SLOTS], q99[0, SLOTS] = quantiles
    result = {
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "train_json_sha256": hashes,
        "train_manifest_sha256": digest(manifest_path),
        "source_revision": upstream["revision"],
        "complete_action_windows": total,
        "state_frames": len(active),
    }
    validate_stats(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, prefix=f".{output.name}.", delete=False) as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
        temporary = Path(stream.name)
    temporary.rename(output)
    return {"output": str(output), "episodes": len(entries), "windows": total, "state_frames": len(active)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xr1-source", type=Path, default=VENDOR)
    args = parser.parse_args()
    print(json.dumps(compute(args.manifest, args.output, args.xr1_source)))


if __name__ == "__main__":
    main()
