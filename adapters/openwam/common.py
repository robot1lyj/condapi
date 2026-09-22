"""OpenWAM source selection and dependency-free YAM configuration checks."""

import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "third_party/openwam"
CAMERAS = [f"observation.images.{name}_rgb" for name in ("top", "left", "right")]


def activate_source():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(VENDOR))


def verify_source():
    manifest = json.loads((VENDOR / "UPSTREAM.json").read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Vendored OpenWAM source changed: {name}; review and update provenance")
    return manifest["revision"]


def validate_data_config(dl, architecture):
    if dl.get("type") != "yam_lerobot_v2" or dl.get("action_mode") != "joint":
        raise ValueError("Expected yam_lerobot_v2 with joint actions")
    if dl.get("action_semantics") != "absolute":
        raise ValueError("Only audited absolute YAM action targets are supported")
    if list(dl.get("camera_layout", [])) != CAMERAS or not dl.get("multiview"):
        raise ValueError("YAM requires all three ordered RGB cameras")
    if dl.get("normalize_mode") not in ("min-max", "z-score"):
        raise ValueError("Use YAM training-subset min-max or z-score statistics")
    frames, stride = dl.get("num_frames"), dl.get("video_stride")
    if type(frames) is not int or frames < 2 or type(stride) is not int or stride < 1:
        raise ValueError("Invalid temporal geometry")
    if (frames - 1) % stride or ((frames - 1) // stride) % 4:
        raise ValueError("Video window must have 4k+1 frames and include the final frame")
    for key in ("height", "width"):
        if type(dl.get(key)) is not int or dl[key] <= 0 or dl[key] % 32:
            raise ValueError(f"{key} must be a positive multiple of 32")
    if dl.get("binary_action_dims"):
        raise ValueError("YAM grippers must retain continuous recorded values")
    episodes = dl.get("episodes")
    if not isinstance(episodes, (list, tuple)) or not episodes:
        raise ValueError("An explicit nonempty training episode selection is required")
    if any(type(e) is not int or e < 0 for e in episodes) or len(set(episodes)) != len(episodes):
        raise ValueError("Invalid or duplicate episode ids")
    dim = architecture.get("action_dim")
    if dim != architecture.get("state_dim") or not architecture.get("use_proprioception"):
        raise ValueError("Matching state/action dimensions and proprioception are required")
    if dl.get("unify_action"):
        slots = dl.get("unify_action_map")
        if dim != 80 or not isinstance(slots, (list, tuple)) or len(slots) != 14:
            raise ValueError("80D checkpoints require an explicit 14-slot YAM mapping")
        if any(type(s) is not int or not 0 <= s < 80 for s in slots) or len(set(slots)) != 14:
            raise ValueError("YAM slots must be unique integers in [0,80)")
        if dl.get("unify_state_map") not in (None, slots):
            raise ValueError("YAM action and state slot maps must match")
    elif dim != 14 or dl.get("unify_action_map") or dl.get("unify_state_map"):
        raise ValueError("Non-unified YAM requires native 14D state/action heads")


def read_info(root):
    root = Path(root).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") not in ("v2.0", "v2.1"):
        raise ValueError("This adapter supports LeRobot v2.0/v2.1 only; never converts source data in place")
    fps = info.get("fps")
    if not isinstance(fps, (float, int)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Dataset metadata must provide positive finite fps")
    for key in ("action", "observation.state"):
        if info["features"][key]["shape"] != [14]:
            raise ValueError(f"Expected 14D {key}")
    for key in CAMERAS:
        if info["features"][key]["dtype"] != "video":
            raise ValueError(f"Expected video feature {key}")
    return info


def dataset_path(root, info, kind, episode, camera=None):
    root = Path(root).resolve()
    path = root / info[f"{kind}_path"].format(
        episode_index=episode, episode_chunk=episode // info["chunks_size"], video_key=camera
    )
    if not path.resolve().is_relative_to(root):
        raise ValueError("Dataset path escapes source root")
    return path
