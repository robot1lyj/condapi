"""Materialize a YAM XR-1 EEF sidecar as native expert-only JSON/videos.

Run YAM's ``hil.xr1_dataset`` exporter first. This reader does not import the
robot controller, write the source episode, or infer a train/validation split.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import av
import numpy as np

from adapters.xr1.common import validate_episode

VIEWS = {"top": "ego", "left": "wrist_left", "right": "wrist_right"}
ARRAY_SHAPES = {
    "observation_state": (14,),
    "expert_action": (14,),
    "observation_pose": (2, 4, 4),
    "action_pose": (2, 4, 4),
    "observation_gripper": (2,),
    "action_gripper": (2,),
    "source_tick": (),
    "source_time": (),
    "source_video_index": (3,),
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _arrays(path: Path, expected_rows: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(ARRAY_SHAPES):
            raise ValueError(f"{path}: unexpected sidecar arrays")
        arrays = {name: archive[name].copy() for name in ARRAY_SHAPES}
    for name, shape in ARRAY_SHAPES.items():
        value = arrays[name]
        if value.shape != (expected_rows, *shape) or not np.isfinite(value).all():
            raise ValueError(f"{path}: invalid {name} shape or value")
    if not np.issubdtype(arrays["source_tick"].dtype, np.integer):
        raise ValueError(f"{path}: ticks must be integers")
    if not np.issubdtype(arrays["source_video_index"].dtype, np.integer):
        raise ValueError(f"{path}: video indices must be integers")
    if np.any(np.diff(arrays["source_tick"]) != 1):
        raise ValueError(f"{path}: expert ticks are not contiguous")
    if np.any(np.diff(arrays["source_time"]) <= 0):
        raise ValueError(f"{path}: expert time is not increasing")
    if np.any(arrays["source_video_index"] < 0) or np.any(np.diff(arrays["source_video_index"], axis=0) < 0):
        raise ValueError(f"{path}: camera indices are invalid or move backward")
    if not np.allclose(arrays["observation_state"][:, [6, 13]], arrays["observation_gripper"], atol=1e-5):
        raise ValueError(f"{path}: observation gripper disagrees with state")
    if not np.allclose(arrays["expert_action"][:, [6, 13]], arrays["action_gripper"], atol=1e-5):
        raise ValueError(f"{path}: action gripper disagrees with submitted target")
    for name in ("observation_pose", "action_pose"):
        poses = arrays[name]
        rotations = poses[:, :, :3, :3]
        if not np.allclose(poses[:, :, 3, :], [0, 0, 0, 1], atol=1e-5):
            raise ValueError(f"{path}: {name} has invalid homogeneous row")
        if not np.allclose(rotations @ np.swapaxes(rotations, -1, -2), np.eye(3), atol=1e-4):
            raise ValueError(f"{path}: {name} has invalid rotation")
        if not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
            raise ValueError(f"{path}: {name} has improper rotation")
    return arrays


def _video_subset(source: Path, indices: np.ndarray, destination: Path) -> int:
    """Decode source indices in order; repeated camera frames remain repeated."""
    with ExitStack() as stack:
        reader = stack.enter_context(av.open(str(source)))
        writer = stack.enter_context(av.open(str(destination), "w"))
        frames = iter(reader.decode(video=0))
        position = -1
        current = None
        stream = None
        written = 0
        for requested in indices:
            index = int(requested)
            if index < position:
                raise ValueError(f"{source}: video index moved backward")
            while position < index:
                try:
                    current = next(frames)
                except StopIteration as exc:
                    raise ValueError(f"{source}: missing source video frame {index}") from exc
                position += 1
            if stream is None:
                stream = writer.add_stream("libx264", rate=30, options={"crf": "18", "preset": "fast"})
                stream.width, stream.height = current.width, current.height
                stream.pix_fmt = "yuv420p"
            frame = av.VideoFrame.from_ndarray(current.to_ndarray(format="rgb24"), format="rgb24")
            for packet in stream.encode(frame):
                writer.mux(packet)
            written += 1
        if stream is not None:
            for packet in stream.encode():
                writer.mux(packet)
    if written != len(indices):
        raise ValueError(f"{source}: selected frame count mismatch")
    with av.open(str(destination)) as output:
        decoded = sum(1 for _ in output.decode(video=0))
    if decoded != written:
        raise ValueError(f"{destination}: encoded video has {decoded} frames, expected {written}")
    return written


def _native_episode(
    arrays: dict[str, np.ndarray], videos: dict[str, Path], instruction: str, final_dir: Path, outcome: str
):
    count = len(arrays["source_tick"])
    prompt = (
        "The following observations are captured from multiple views.\n"
        "# Ego View\n<image>\n# Left-Wrist View\n<image>\n"
        f"# Right-Wrist View\n<image>\nGenerate robot actions for the task:\n{instruction}"
    )
    data = {
        "trajectory_type": "success" if outcome == "success" else "ongoing",
        "num_frames": count,
        "instruction": {
            "general": [
                {
                    "images": [f"observations.{view}" for view in VIEWS.values()],
                    "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}],
                }
            ]
        },
        "observations": {
            view: [{"path": str(final_dir / videos[role].name), "start": 0, "crop_bbox": None}]
            for role, view in VIEWS.items()
        },
        "proprios": {"waist_pos": [[0.0]] * count},
        "actions": {"waist_pos": [[0.0]] * count, "base_vel": [[0.0, 0.0, 0.0]] * count},
    }
    for arm_index, arm in enumerate(("left", "right")):
        offset = arm_index * 7
        for group, poses, grippers in (
            ("proprios", arrays["observation_pose"], arrays["observation_gripper"]),
            ("actions", arrays["action_pose"], arrays["action_gripper"]),
        ):
            data[group][f"{arm}_ee_pos"] = poses[:, arm_index, :3, 3].tolist()
            data[group][f"{arm}_ee_rotm"] = poses[:, arm_index, :3, :3].reshape(count, 9).tolist()
            data[group][f"{arm}_gripper_pos"] = grippers[:, arm_index, None].tolist()
        data["proprios"][f"{arm}_arm_joint"] = arrays["observation_state"][:, offset : offset + 6].tolist()
    return data


def materialize(sidecar: Path, source_episode: Path, destination: Path, split: str, instruction: str) -> dict:
    sidecar, source_episode, destination = (Path(p).resolve() for p in (sidecar, source_episode, destination))
    if split not in ("train", "val") or not instruction.strip():
        raise ValueError("explicit train/val split and task instruction are required")
    if destination.exists() or destination.is_relative_to(source_episode) or destination.is_relative_to(sidecar):
        raise ValueError("output must be a new directory outside source and sidecar")
    side_manifest = json.loads((sidecar / "manifest.json").read_text())
    manifest = json.loads((source_episode / "manifest.json").read_text())
    if (
        side_manifest.get("schema") != "yam_xr1_eef_sidecar_v1"
        or side_manifest.get("coordinate_frame") != "each_arm_base/linear_4310/grasp_site"
    ):
        raise ValueError("unrecognized YAM EEF sidecar contract")
    if (
        side_manifest.get("camera_roles") != list(VIEWS)
        or side_manifest.get("fps") != 30
        or side_manifest.get("action_horizon") != 30
        or side_manifest.get("expert_only") is not True
    ):
        raise ValueError("sidecar camera, rate, horizon, or expert contract mismatch")
    if manifest.get("schema") not in ("yam_hil_v1", "yam_hil_v2") or manifest.get("mock") is not False:
        raise ValueError("only real YAM HIL recordings are eligible")
    if (
        manifest.get("outcome") not in ("success", "failure", "unknown")
        or manifest.get("error")
        or not np.isclose(float(manifest.get("fps", 0)), 30)
    ):
        raise ValueError("source episode must be complete, error-free, and 30 Hz")
    if instruction.strip() != manifest.get("task"):
        raise ValueError("instruction must equal the source episode task")
    origin = Path(side_manifest["source_episode"]).resolve()
    if origin.name != source_episode.name:
        raise ValueError("sidecar/source episode identity mismatch")
    entries = side_manifest.get("segments")
    if not isinstance(entries, list) or not entries:
        raise ValueError("sidecar has no expert segments")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    output_entries = []
    published = False
    try:
        for index, entry in enumerate(entries):
            filename = entry["file"]
            source_npz = (sidecar / filename).resolve()
            if not source_npz.is_relative_to(sidecar) or source_npz.suffix != ".npz":
                raise ValueError("sidecar NPZ path escapes source directory")
            rows = entry["rows"]
            if type(rows) is not int or rows < 1:
                raise ValueError("invalid expert segment length")
            arrays = _arrays(source_npz, rows)
            if rows < 30:
                continue  # Native XR-1 requires a full 30-step window.
            segment_ref = entry.get("source_segment", "")
            relative = Path(segment_ref).resolve().relative_to(origin) if segment_ref else Path(".")
            source_segment = (source_episode / relative).resolve()
            if not source_segment.is_relative_to(source_episode):
                raise ValueError("source video path escapes episode")
            name = f"expert_{index:06d}"
            temp_dir = temporary / name
            final_dir = destination / name
            temp_dir.mkdir()
            videos = {}
            source_hashes = {}
            for column, role in enumerate(VIEWS):
                original_video = source_segment / f"{role}.mp4"
                videos[role] = temp_dir / f"{role}.mp4"
                source_hashes[str(original_video)] = _sha256(original_video)
                _video_subset(original_video, arrays["source_video_index"][:, column], videos[role])
            native = _native_episode(arrays, videos, instruction, final_dir, manifest.get("outcome", "unknown"))
            json_path = temp_dir / "episode.json"
            json_path.write_text(json.dumps(native, ensure_ascii=False, allow_nan=False) + "\n")
            provenance = {
                "source_episode": str(source_episode),
                "source_manifest_sha256": _sha256(source_episode / "manifest.json"),
                "source_sidecar": str(source_npz),
                "sidecar_sha256": _sha256(source_npz),
                "source_video_sha256": source_hashes,
                "source_segment": str(source_segment),
                "source_tick": arrays["source_tick"].tolist(),
                "source_time": arrays["source_time"].tolist(),
                "source_video_index": arrays["source_video_index"].tolist(),
                "split": split,
                "coordinate_frame": side_manifest["coordinate_frame"],
                "action_source": "submitted_action",
            }
            (temp_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False) + "\n")
            output_entries.append(
                {
                    "json": str(final_dir / "episode.json"),
                    "json_sha256": _sha256(json_path),
                    "video_sha256": {role: _sha256(video) for role, video in videos.items()},
                    "rows": rows,
                    "source_segment": str(source_segment),
                    "source_tick_start": int(arrays["source_tick"][0]),
                    "source_tick_end": int(arrays["source_tick"][-1]),
                }
            )
        if not output_entries:
            raise ValueError("no complete 30-frame expert segment")
        (temporary / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "yam_xr1_native_eef_v1",
                    "split": split,
                    "source_episode": str(source_episode),
                    "sidecar_manifest_sha256": _sha256(sidecar / "manifest.json"),
                    "segments": output_entries,
                    "skipped_short_segments": len(entries) - len(output_entries),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        temporary.rename(destination)
        published = True
        for entry in output_entries:
            validate_episode(entry["json"])
    except Exception:
        # Both directories belong to this invocation; no source or prior output is removed.
        shutil.rmtree(destination if published else temporary)
        raise
    return {
        "output": str(destination),
        "split": split,
        "expert_segments": len(output_entries),
        "skipped_short_segments": len(entries) - len(output_entries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--source-episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--instruction", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(args.sidecar, args.source_episode, args.output, args.split, args.instruction),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
