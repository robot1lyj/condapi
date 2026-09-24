"""Convert an explicit YAM LeRobot v3 episode selection to XR-1 EEF JSON.

Original Parquet and shared AV1 videos are read-only. The server's native
decord 0.6.0 cannot decode those AV1 files, so selected episode ranges are
materialized as H.264 in a new directory with exact source offsets recorded.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import av
import mujoco
import numpy as np
import pyarrow.parquet as pq

from adapters.xr1.common import validate_episode

CAMERAS = {"top": "ego", "left": "wrist_left", "right": "wrist_right"}
IMAGE_KEYS = {role: f"observation.images.{role}_rgb" for role in CAMERAS}
MODEL = Path(__file__).resolve().parents[2] / "third_party/yam-fk-model/yam_linear4310.xml"
TASK = "sort the legos into containers by color"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_fk_bundle(xml: Path) -> str:
    provenance_path = xml.parent / "PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("coordinate_frame") != "each_arm_base/linear_4310/grasp_site":
        raise ValueError("unrecognized YAM FK frame")
    if provenance.get("xml_sha256") != digest(xml):
        raise ValueError("YAM FK XML hash differs from its source record")
    for name, expected in provenance["meshes_sha256"].items():
        path = (xml.parent / name).resolve()
        if not path.is_relative_to(xml.parent) or digest(path) != expected:
            raise ValueError(f"YAM FK mesh hash mismatch: {name}")
    return digest(provenance_path)


class ForwardKinematics:
    """Official YAM + linear_4310 grasp_site at zero gripper displacement."""

    def __init__(self, xml: Path):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        if self.site < 0 or self.model.nq != 8:
            raise ValueError("unexpected YAM grasp_site or joint layout")
        for index in range(6):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index)
            if name != f"joint{index + 1}":
                raise ValueError("YAM joint order does not match 14D source contract")

    def poses(self, values: np.ndarray) -> np.ndarray:
        count = len(values)
        result = np.empty((count, 2, 4, 4), dtype=np.float64)
        result[:, :, 3, :] = [0, 0, 0, 1]
        self.data.qpos[:] = 0
        for row, joints in enumerate(values):
            for arm, offset in enumerate((0, 7)):
                self.data.qpos[:6] = joints[offset : offset + 6]
                mujoco.mj_kinematics(self.model, self.data)
                result[row, arm, :3, :3] = self.data.site_xmat[self.site].reshape(3, 3)
                result[row, arm, :3, 3] = self.data.site_xpos[self.site]
        return result


def _episode_metadata(root: Path, info: dict) -> dict[int, dict]:
    paths = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not paths:
        raise ValueError("missing LeRobot v3 episode metadata")
    table = pq.read_table(
        paths,
        columns=[
            "episode_index",
            "length",
            "tasks",
            "data/chunk_index",
            "data/file_index",
            *(
                f"videos/{key}/{field}"
                for key in IMAGE_KEYS.values()
                for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
            ),
        ],
    )
    rows = {int(row["episode_index"]): row for row in table.to_pylist()}
    if len(rows) != len(table) or len(rows) != info["total_episodes"]:
        raise ValueError("duplicate or incomplete episode metadata")
    return rows


def _path(root: Path, template: str, chunk: int, file: int, video_key: str = "") -> Path:
    path = (root / template.format(chunk_index=chunk, file_index=file, video_key=video_key)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"missing or escaping LeRobot source file: {path}")
    return path


def _row_data(root: Path, info: dict, metadata: dict, episode: int):
    path = _path(root, info["data_path"], metadata["data/chunk_index"], metadata["data/file_index"])
    columns = ["episode_index", "frame_index", "timestamp", "task_index", "observation.state", "action"]
    table = pq.read_table(path, columns=columns, filters=[("episode_index", "=", episode)])
    count = int(metadata["length"])
    if len(table) != count or count < 30:
        raise ValueError(f"episode {episode}: missing rows or fewer than 30 frames")
    values = table.to_pydict()
    if values["frame_index"] != list(range(count)) or set(values["episode_index"]) != {episode}:
        raise ValueError(f"episode {episode}: frame/episode index mismatch")
    if len(set(values["task_index"])) != 1:
        raise ValueError(f"episode {episode}: multiple task indices")
    time = np.asarray(values["timestamp"], dtype=np.float64)
    if not np.allclose(time, np.arange(count) / info["fps"], atol=1e-3):
        raise ValueError(f"episode {episode}: timestamp/fps mismatch")
    state = np.asarray(values["observation.state"], dtype=np.float64)
    action = np.asarray(values["action"], dtype=np.float64)
    if (
        state.shape != (count, 14)
        or action.shape != state.shape
        or not np.isfinite(state).all()
        or not np.isfinite(action).all()
    ):
        raise ValueError(f"episode {episode}: invalid 14D state/action")
    return path, time, state, action


def _native(state, action, observation_pose, target_pose, videos, instruction):
    count = len(state)
    prompt = (
        "The following observations are captured from multiple views.\n"
        "# Ego View\n<image>\n# Left-Wrist View\n<image>\n"
        f"# Right-Wrist View\n<image>\nGenerate robot actions for the task:\n{instruction}"
    )
    result = {
        "trajectory_type": "ongoing",
        "num_frames": count,
        "instruction": {
            "general": [
                {
                    "images": [f"observations.{view}" for view in CAMERAS.values()],
                    "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}],
                }
            ]
        },
        "observations": {
            view: [{"path": str(videos[role][0]), "start": videos[role][1], "crop_bbox": None}]
            for role, view in CAMERAS.items()
        },
        "proprios": {"waist_pos": [[0.0]] * count},
        "actions": {"waist_pos": [[0.0]] * count, "base_vel": [[0.0, 0.0, 0.0]] * count},
    }
    for arm, name in enumerate(("left", "right")):
        offset = arm * 7
        for group, poses, joints in (("proprios", observation_pose, state), ("actions", target_pose, action)):
            result[group][f"{name}_ee_pos"] = poses[:, arm, :3, 3].tolist()
            result[group][f"{name}_ee_rotm"] = poses[:, arm, :3, :3].reshape(count, 9).tolist()
            result[group][f"{name}_gripper_pos"] = joints[:, offset + 6, None].tolist()
        result["proprios"][f"{name}_arm_joint"] = state[:, offset : offset + 6].tolist()
    return result


def _transcode_range(source: Path, start: int, count: int, destination: Path) -> None:
    """Copy one exact episode frame range to H.264 for native decord 0.6.0."""
    with ExitStack() as stack:
        reader = stack.enter_context(av.open(str(source)))
        writer = stack.enter_context(av.open(str(destination), "w"))
        video = reader.streams.video[0]
        if video.average_rate is None or abs(float(video.average_rate) - 30) > 1e-3:
            raise ValueError(f"unexpected source video rate: {source}")
        encoder = None
        written = 0
        for index, frame in enumerate(reader.decode(video=0)):
            if index < start:
                continue
            if index >= start + count:
                break
            if encoder is None:
                encoder = writer.add_stream("libx264", rate=30, options={"crf": "18", "preset": "fast"})
                encoder.width, encoder.height = frame.width, frame.height
                encoder.pix_fmt = "yuv420p"
            rgb = av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")
            for packet in encoder.encode(rgb):
                writer.mux(packet)
            written += 1
        if encoder is not None:
            for packet in encoder.encode():
                writer.mux(packet)
    if written != count:
        raise ValueError(f"{source}: decoded {written} of {count} requested frames from {start}")
    with av.open(str(destination)) as output:
        frames = output.streams.video[0].frames or sum(1 for _ in output.decode(video=0))
        if frames != count:
            raise ValueError(f"{destination}: encoded frame count mismatch")


def convert(
    root: Path,
    selection: Path | None,
    output: Path,
    split: str,
    xml: Path = MODEL,
    *,
    resume: bool = False,
    video_workers: int = 1,
) -> dict:
    root, output, xml = Path(root).resolve(), Path(output).resolve(), Path(xml).resolve()
    if type(video_workers) is not int or not 1 <= video_workers <= 3:
        raise ValueError("video_workers must be 1..3")
    if (
        split not in ("train", "val")
        or root.name != split
        or output.is_relative_to(root)
        or (output.exists() and not resume)
    ):
        raise ValueError("explicit split and a new output outside the source repo are required")
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0" or info.get("fps") != 30:
        raise ValueError("expected YAM LeRobot v3 at 30 fps")
    for key in ("observation.state", "action"):
        if info["features"][key]["shape"] != [14]:
            raise ValueError(f"unexpected {key} schema")
    if any(info["features"][key]["dtype"] != "video" for key in IMAGE_KEYS.values()):
        raise ValueError("missing three video features")
    metadata = _episode_metadata(root, info)
    if selection is None:
        if split != "val":
            raise ValueError("training requires an explicit episode selection")
        episodes = sorted(metadata)
        selection_hash = None
    else:
        selection = Path(selection).resolve()
        episodes = json.loads(selection.read_text())
        if (
            not isinstance(episodes, list)
            or len(set(episodes)) != len(episodes)
            or any(type(i) is not int for i in episodes)
        ):
            raise ValueError("selection must be a unique integer episode list")
        selection_hash = digest(selection)
    if not episodes or set(episodes) - metadata.keys():
        raise ValueError("selection includes missing episodes")
    source_manifest = root / "conversion_manifest.json"
    fk_bundle_sha256 = verify_fk_bundle(xml)
    identity = {
        "schema": "yam_xr1_lego_eef_v1",
        "split": split,
        "source_repo": str(root),
        "source_info_sha256": digest(info_path),
        "source_manifest_sha256": digest(source_manifest),
        "selection_sha256": selection_hash,
        "episode_ids": episodes,
        "converter_sha256": digest(Path(__file__).resolve()),
        "fk_model": str(xml),
        "fk_model_sha256": digest(xml),
        "fk_bundle_sha256": fk_bundle_sha256,
        "coordinate_frame": "each_arm_base/linear_4310/grasp_site",
        "source_action": "absolute joint target, same LeRobot row",
        "source_unit_status": "radian inferred from publisher contract; robot calibration unverified",
        "derived_video": "H.264/yuv420p, libx264 crf18 preset fast, one source episode range per file",
        "video_workers": video_workers,
    }
    if resume:
        if json.loads((output / "identity.json").read_text()) != identity:
            raise ValueError("resume identity differs from this source, selection, or FK model")
    else:
        output.mkdir(parents=True)
        (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
        (output / "episodes").mkdir()
    fk = ForwardKinematics(xml)
    converted = []
    for episode in episodes:
        row = metadata[episode]
        if row["tasks"] != [TASK]:
            raise ValueError(f"episode {episode}: unexpected task {row['tasks']}")
        source_parquet = _path(root, info["data_path"], row["data/chunk_index"], row["data/file_index"])
        videos = {}
        for role, key in IMAGE_KEYS.items():
            prefix = f"videos/{key}"
            start = row[f"{prefix}/from_timestamp"] * info["fps"]
            end = row[f"{prefix}/to_timestamp"] * info["fps"]
            if (
                not np.isfinite([start, end]).all()
                or abs(start - round(start)) > 1e-3
                or abs(end - start - row["length"]) > 1e-3
            ):
                raise ValueError(f"episode {episode}: invalid {role} video offset")
            path = _path(root, info["video_path"], row[f"{prefix}/chunk_index"], row[f"{prefix}/file_index"], key)
            videos[role] = (path, round(start))
        final_dir = output / "episodes" / f"episode_{episode:06d}"
        path = final_dir / "episode.json"
        provenance_path = final_dir / "provenance.json"
        if resume and path.exists():
            validated = validate_episode(path)
            if validated["frames"] != row["length"] or not provenance_path.is_file():
                raise ValueError(f"episode {episode}: incomplete previous output")
            recorded = json.loads(provenance_path.read_text())
            if recorded["source_parquet"] != str(source_parquet) or recorded["video_start"] != {
                role: pair[1] for role, pair in videos.items()
            }:
                raise ValueError(f"episode {episode}: previous source identity differs")
            converted.append(
                {
                    "episode_index": episode,
                    "json": str(path),
                    "json_sha256": validated["sha256"],
                    "frames": validated["frames"],
                    "source_parquet": str(source_parquet),
                    "video_start": recorded["video_start"],
                }
            )
            continue
        source_parquet, time, state, action = _row_data(root, info, row, episode)
        observation_pose = fk.poses(state)
        target_pose = fk.poses(action)
        temp_dir = Path(tempfile.mkdtemp(prefix=f".episode_{episode:06d}.", dir=output / "episodes"))
        try:
            derived_videos = {}
            with ThreadPoolExecutor(max_workers=video_workers) as pool:
                futures = {}
                for role, (source_video, start) in videos.items():
                    derived = temp_dir / f"{role}.mp4"
                    futures[role] = pool.submit(_transcode_range, source_video, start, len(time), derived)
                    derived_videos[role] = (final_dir / derived.name, 0)
                for future in futures.values():
                    future.result()
            native = _native(state, action, observation_pose, target_pose, derived_videos, TASK)
            (temp_dir / "episode.json").write_text(json.dumps(native, ensure_ascii=False, allow_nan=False) + "\n")
            (temp_dir / "provenance.json").write_text(
                json.dumps(
                    {
                        "source_episode_index": episode,
                        "source_parquet": str(source_parquet),
                        "source_timestamps": time.tolist(),
                        "source_frame_index_start": 0,
                        "video_start": {role: pair[1] for role, pair in videos.items()},
                        "source_video": {role: str(pair[0]) for role, pair in videos.items()},
                        "derived_video_sha256": {role: digest(temp_dir / f"{role}.mp4") for role in videos},
                        "fk_model_sha256": identity["fk_model_sha256"],
                    }
                )
                + "\n"
            )
            temp_dir.rename(final_dir)
        except Exception:
            shutil.rmtree(temp_dir)
            raise
        validated = validate_episode(path)
        converted.append(
            {
                "episode_index": episode,
                "json": str(path),
                "json_sha256": validated["sha256"],
                "frames": len(time),
                "source_parquet": str(source_parquet),
                "video_start": {role: pair[1] for role, pair in videos.items()},
            }
        )
        if len(converted) % 25 == 0:
            print(json.dumps({"converted": len(converted), "total": len(episodes), "episode": episode}), flush=True)
    (output / "manifest.json").write_text(json.dumps({**identity, "episodes": converted}, indent=2) + "\n")
    return {"output": str(output), "episodes": len(converted), "frames": sum(x["frames"] for x in converted)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--fk-model", type=Path, default=MODEL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--video-workers", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            convert(
                args.source,
                args.selection,
                args.output,
                args.split,
                args.fk_model,
                resume=args.resume,
                video_workers=args.video_workers,
            )
        )
    )


if __name__ == "__main__":
    main()
