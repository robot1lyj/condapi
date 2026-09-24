"""Dependency-free source and native-data preflight for XR-1."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "third_party/xr1"
ARMS = ("left", "right")
VIEWS = ("ego", "wrist_left", "wrist_right")


def verify_source():
    manifest = json.loads((VENDOR / "UPSTREAM.json").read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"XR-1 source changed: {name}")
    return manifest["revision"]


def _finite_vector(value, size, label):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain {size} values")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError(f"{label} contains a non-finite value")


def _rotation(value, label):
    rows = [value[0:3], value[3:6], value[6:9]]
    for row in rows:
        norm = sum(x * x for x in row)
        if abs(norm - 1.0) > 1e-2:
            raise ValueError(f"{label} is not a unit rotation matrix")
    for first, second in ((0, 1), (0, 2), (1, 2)):
        if abs(sum(a * b for a, b in zip(rows[first], rows[second]))) > 1e-2:
            raise ValueError(f"{label} is not orthogonal")
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    if abs(determinant - 1.0) > 1e-2:
        raise ValueError(f"{label} is not a proper rotation matrix")


def validate_episode(path):
    """Reject partial/ambiguous native labels before the GPU data loader starts."""
    path = Path(path).resolve()
    episode = json.loads(path.read_text())
    count = episode.get("num_frames")
    if type(count) is not int or count < 30:
        raise ValueError(f"{path}: at least 30 synchronized frames are required")
    for view in VIEWS:
        entries = episode.get("observations", {}).get(view)
        if not isinstance(entries, list) or len(entries) != 1:
            raise ValueError(f"{path}: missing {view} video")
        video = Path(entries[0].get("path", ""))
        if not video.is_absolute() or not video.is_file():
            raise ValueError(f"{path}: missing {view} video file")
    for arm in ARMS:
        for group, fields in (
            ("proprios", (("ee_pos", 3), ("ee_rotm", 9), ("arm_joint", 6), ("gripper_pos", 1))),
            ("actions", (("ee_pos", 3), ("ee_rotm", 9), ("gripper_pos", 1))),
        ):
            for field, size in fields:
                key = f"{arm}_{field}"
                series = episode.get(group, {}).get(key)
                if not isinstance(series, list) or len(series) != count:
                    raise ValueError(f"{path}: {group}.{key} must have {count} frames")
                for index, value in enumerate(series):
                    if field == "arm_joint":
                        if not isinstance(value, list) or len(value) != 6:
                            raise ValueError(f"{path}: {group}.{key}[{index}] requires 6 YAM joints")
                        _finite_vector(value, len(value), f"{group}.{key}[{index}]")
                    else:
                        _finite_vector(value, size, f"{group}.{key}[{index}]")
                        if field == "ee_rotm":
                            _rotation(value, f"{group}.{key}[{index}]")
    for group, fields in (("proprios", (("waist_pos", 1),)), ("actions", (("waist_pos", 1), ("base_vel", 3)))):
        for key, size in fields:
            series = episode.get(group, {}).get(key)
            if not isinstance(series, list) or len(series) != count:
                raise ValueError(f"{path}: {group}.{key} must have {count} frames")
            for index, value in enumerate(series):
                _finite_vector(value, size, f"{group}.{key}[{index}]")
    prompt = episode.get("instruction", {}).get("general")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"{path}: missing instruction.general")
    return {"path": str(path), "frames": count, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def validate_stats(stats):
    for key, rows, cols in (("mean", 30, 60), ("std", 30, 60), ("q01", 1, 60), ("q99", 1, 60)):
        value = stats.get(key)
        if not isinstance(value, list) or len(value) != rows:
            raise ValueError(f"XR-1 {key} must have {rows} rows")
        for row in value:
            _finite_vector(row, cols, key)
    if any(x < 0 for row in stats["std"] for x in row):
        raise ValueError("XR-1 standard deviations must be nonnegative")
    if any(a > b for a, b in zip(stats["q01"][0], stats["q99"][0])):
        raise ValueError("XR-1 state quantiles are reversed")
