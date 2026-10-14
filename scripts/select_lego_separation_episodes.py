"""Find lifted, horizontal-gripper episodes outside a frozen LeRobot selection.

This is an offline pose heuristic, not a separation-success classifier. Source
Parquet, videos, split and training assets remain read-only. Use the dedicated
YAM data environment (NumPy, PyArrow, MuJoCo); no trainer is imported.
"""

# Optional data/FK dependencies are loaded only for a scan, not for rule tests.
# ruff: noqa: PLC0415

import argparse
from datetime import UTC
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selection(path):
    values = json.loads(Path(path).read_text())
    if not isinstance(values, list) or any(type(i) is not int for i in values) or len(set(values)) != len(values):
        raise ValueError("selection must contain unique integer episode IDs")
    return set(values)


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)


def find_events(positions, wrist, axes, gripper, frames, fps, stride):
    """Angles use grasp_site +Z (flange-to-fingertips) and nominal base +Z up.

    Wrist rise is reported separately because rotating a tool alone can lift
    its grasp_site. Same-arm downward history is required; axial roll cannot
    trigger a hit. No gripper threshold silently removes valid demonstrations.
    """
    tilt = np.degrees(np.arccos(np.clip(-axes[:, 2], -1, 1)))
    horizontal = np.abs(axes[:, 2]) <= np.sin(np.deg2rad(20))
    events = []
    lookback = int(np.ceil(4 * fps / stride))
    minimum = int(np.ceil(0.2 * fps / stride))
    for start, stop in runs(horizontal):
        if stop - start < minimum or start == 0:
            continue
        before = np.arange(max(0, start - lookback), start)
        downward = before[tilt[before] <= 45]
        if not len(downward):
            continue
        # Most recent downward pose, not an unrelated episode-wide minimum.
        baseline = int(downward[-1])
        peak = start + int(np.argmax(positions[start:stop, 2]))
        rise = float(positions[peak, 2] - positions[baseline, 2])
        wrist_rise = float(wrist[peak, 2] - wrist[baseline, 2])
        change = float(np.max(tilt[start:stop]) - tilt[baseline])
        if rise < 0.025 or change < 25:
            continue
        duration = float((frames[stop - 1] - frames[start] + stride) / fps)
        closed_fraction = float(np.mean(gripper[start:stop] < 0.6))
        # A ranking aid only: contact, stuck bricks and success require video.
        score = (
            min(duration, 3) * 3
            + min(change, 90) / 3
            + min(rise, 0.2) * 100
            + np.clip(wrist_rise, 0, 0.2) * 60
            + closed_fraction * 20
        )
        events.append(
            {
                "start_frame": int(frames[start]),
                "end_frame_exclusive": int(frames[stop - 1] + stride),
                "peak_frame": int(frames[peak]),
                "baseline_frame": int(frames[baseline]),
                "duration_s": duration,
                "nominal_grasp_rise_m": rise,
                "nominal_wrist_rise_m": wrist_rise,
                "baseline_tilt_deg": float(tilt[baseline]),
                "peak_tilt_deg": float(tilt[peak]),
                "max_tilt_deg": float(np.max(tilt[start:stop])),
                "closed_fraction_below_0_6": closed_fraction,
                "score": float(score),
                "separation_success": None,
            }
        )
    return events


def find_bimanual_events(poses, state, frames, fps, stride, unilateral):
    """Require simultaneous horizontal long axes and a preceding lift/turn.

    Bases have independent origins; only their nominal vertical directions are
    used. No uncalibrated cross-arm distance is inferred. Either hand can hold
    its sideways pose while the other approaches; both need not turn together.
    """
    horizontal = np.all(np.abs(poses[:, :, 2, 2]) <= np.sin(np.deg2rad(20)), axis=1)
    minimum = int(np.ceil(0.3 * fps / stride))
    result = []
    for start, stop in runs(horizontal):
        if stop - start < minimum:
            continue
        begin, end = int(frames[start]), int(frames[stop - 1] + stride)
        parents = [e for e in unilateral if e["start_frame"] < end and e["end_frame_exclusive"] > begin]
        if not parents:
            continue
        parent = max(parents, key=lambda e: e["score"])
        grip = state[start:stop][:, [6, 13]]
        both_closed = float(np.mean(np.all(grip < 0.6, axis=1)))
        either_closed = float(np.mean(np.any(grip < 0.6, axis=1)))
        duration = (end - begin) / fps
        event = dict(parent)
        event.update(
            {
                "start_frame": begin,
                "end_frame_exclusive": end,
                "peak_frame": int(frames[(start + stop - 1) // 2]),
                "duration_s": duration,
                "arm": "both",
                "lifting_arm": parent["arm"],
                "both_closed_fraction_below_0_6": both_closed,
                "either_closed_fraction_below_0_6": either_closed,
                "score": parent["score"] + 30 * both_closed + 10 * either_closed + 3 * min(duration, 3),
                "left_tilt_range_deg": np.degrees(np.arccos(np.clip(-poses[start:stop, 0, 2, 2], -1, 1)))
                .take([0, -1])
                .tolist(),
                "right_tilt_range_deg": np.degrees(np.arccos(np.clip(-poses[start:stop, 1, 2, 2], -1, 1)))
                .take([0, -1])
                .tolist(),
            }
        )
        result.append(event)
    return sorted(result, key=lambda e: e["score"], reverse=True)


class ForwardKinematics:
    def __init__(self, model_path):
        import mujoco

        self.mujoco = mujoco
        provenance = json.loads(model_path.with_name("PROVENANCE.json").read_text())
        if provenance["coordinate_frame"] != "each_arm_base/linear_4310/grasp_site":
            raise ValueError("unexpected FK coordinate frame")
        if sha256(model_path) != provenance["xml_sha256"]:
            raise ValueError("FK XML identity changed")
        for name, expected in provenance["meshes_sha256"].items():
            asset = (model_path.parent / name).resolve()
            if not asset.is_relative_to(model_path.parent.resolve()) or sha256(asset) != expected:
                raise ValueError("FK mesh identity changed")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        if self.model.nq != 8:
            raise ValueError("unexpected FK joint layout")
        for i in range(6):
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) != f"joint{i + 1}":
                raise ValueError("unexpected FK joint order")
        self.grasp = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        self.wrist = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
        if min(self.grasp, self.wrist) < 0:
            raise ValueError("missing FK sites")

    def poses(self, state):
        result = np.empty((len(state), 2, 3, 3))
        self.data.qpos[:] = 0
        for i, row in enumerate(state):
            for arm, offset in enumerate((0, 7)):
                self.data.qpos[:6] = row[offset : offset + 6]
                self.mujoco.mj_kinematics(self.model, self.data)
                result[i, arm, 0] = self.data.site_xpos[self.grasp]
                result[i, arm, 1] = self.data.site_xpos[self.wrist]
                result[i, arm, 2] = self.data.site_xmat[self.grasp].reshape(3, 3)[:, 2]
        return result


def scan(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    source, output = args.source.resolve(), args.output.resolve()
    if output.is_relative_to(source) or source.name != "train" or args.stride < 1:
        raise ValueError("use an independent output, explicit train split and positive stride")
    output.mkdir(parents=True, exist_ok=False)
    info_path, manifest_path = source / "meta/info.json", source / "conversion_manifest.json"
    info, manifest = json.loads(info_path.read_text()), json.loads(manifest_path.read_text())
    expected_names = [
        f"{arm}_{joint}" for arm in ("left", "right") for joint in (*[f"j{i}" for i in range(1, 7)], "gripper")
    ]
    if info["codebase_version"] != "v3.0" or info["fps"] != 30:
        raise ValueError("expected YAM v3 at 30fps")
    for key in ("observation.state", "action"):
        if info["features"][key]["shape"] != [14] or info["features"][key]["names"] != expected_names:
            raise ValueError("unexpected source joint contract")
    contract = manifest["contract"]
    if contract["joint_unit"] != "radian" or contract["action_mode"] != "absolute":
        raise ValueError("requires documented absolute/radian contract")
    excluded = selection(args.exclude)
    metadata_paths = sorted((source / "meta/episodes").rglob("*.parquet"))
    metadata = pq.read_table(metadata_paths).to_pylist()
    by_id = {row["episode_index"]: row for row in metadata}
    if len(by_id) != len(metadata) or len(by_id) != info["total_episodes"] or not excluded <= by_id.keys():
        raise ValueError("duplicate/incomplete episode inventory or unknown excluded IDs")
    origins = {row["episode_index"]: row for row in manifest["episodes"]}
    pool = sorted(set(by_id) - excluded)
    identities = {str(p): sha256(p) for p in [info_path, manifest_path, args.exclude, *metadata_paths]}
    fk = ForwardKinematics(args.model.resolve())
    columns = ["episode_index", "frame_index", "timestamp", "observation.state", "action"]
    inventory, candidates, source_files = [], [], {}
    cached_path, table = None, None
    started = time.monotonic()
    with (output / "episodes.jsonl").open("x") as stream:
        for number, episode in enumerate(pool, 1):
            row = by_id[episode]
            relative = info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"])
            path = (source / relative).resolve()
            if not path.is_relative_to(source):
                raise ValueError("source path escapes dataset")
            if path != cached_path:
                source_files[str(path)] = {
                    "sha256": sha256(path),
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                table = pq.read_table(path, columns=columns)
                cached_path = path
            import pyarrow.compute as pc

            values = table.filter(pc.equal(table["episode_index"], episode)).to_pydict()
            n = row["length"]
            frames, timestamps = np.asarray(values["frame_index"]), np.asarray(values["timestamp"])
            state, action = np.asarray(values["observation.state"]), np.asarray(values["action"])
            if (
                len(frames) != n
                or not np.array_equal(frames, np.arange(n))
                or not np.allclose(timestamps, np.arange(n) / info["fps"], atol=1e-3, rtol=0)
                or state.shape != (n, 14)
                or action.shape != (n, 14)
                or not np.isfinite(state).all()
                or not np.isfinite(action).all()
            ):
                raise ValueError(f"invalid source rows in episode {episode}")
            sample = np.arange(0, n, args.stride)
            poses = fk.poses(state[sample])
            events = []
            for arm, name in enumerate(("left", "right")):
                found = find_events(
                    poses[:, arm, 0],
                    poses[:, arm, 1],
                    poses[:, arm, 2],
                    state[sample, arm * 7 + 6],
                    sample,
                    info["fps"],
                    args.stride,
                )
                for event in found:
                    event["arm"] = name
                    event["end_frame_exclusive"] = min(n, event["end_frame_exclusive"])
                events.extend(found)
            events.sort(key=lambda e: e["score"], reverse=True)
            unilateral = events
            if args.bimanual:
                events = find_bimanual_events(poses, state[sample], sample, info["fps"], args.stride, unilateral)
                for event in events:
                    event["end_frame_exclusive"] = min(n, event["end_frame_exclusive"])
            origin = origins[episode]
            if origin["length"] != n:
                raise ValueError("source provenance length differs")
            item = {
                "episode_index": episode,
                "source_episode_index": origin["source_episode_index"],
                "source_revision": origin["source_revision"],
                "length": n,
                "full_h50_windows": max(0, n - 49),
                "data_path": str(path),
                "events": events,
                "unilateral_events": unilateral if args.bimanual else [],
                "videos": {},
            }
            for name in ("top", "left", "right"):
                key = f"observation.images.{name}_rgb"
                prefix = f"videos/{key}/"
                video = source / info["video_path"].format(
                    video_key=key, chunk_index=row[prefix + "chunk_index"], file_index=row[prefix + "file_index"]
                )
                if not video.is_file():
                    raise ValueError(f"missing source video {video}")
                item["videos"][name] = {
                    "path": str(video),
                    "start_frame": round(row[prefix + "from_timestamp"] * info["fps"]),
                    "size": video.stat().st_size,
                    "mtime_ns": video.stat().st_mtime_ns,
                }
            stream.write(json.dumps(item) + "\n")
            inventory.append(item)
            if events:
                candidates.append(item)
            if number % 100 == 0 or number == len(pool):
                print(
                    f"SCAN {number}/{len(pool)} candidates={len(candidates)} elapsed_s={time.monotonic() - started:.1f}",
                    flush=True,
                )
    for path, expected in identities.items():
        if sha256(path) != expected:
            raise ValueError("source identity changed during scan")
    for path, sig in source_files.items():
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (sig["size"], sig["mtime_ns"]):
            raise ValueError("source parquet changed during scan")
    candidates.sort(key=lambda i: i["events"][0]["score"], reverse=True)
    summary = {
        "schema": "yam_lego_separation_candidates_v1",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "source": str(source),
        "excluded_selection": str(args.exclude.resolve()),
        "input_identities": identities,
        "script_sha256": sha256(__file__),
        "fk_model": str(args.model.resolve()),
        "fk_model_sha256": sha256(args.model),
        "source_contract": contract,
        "pose_frame": "each_arm_base; nominal +Z up; grasp_site +Z toward fingertips",
        "independent_robot_calibration": False,
        "stride": args.stride,
        "sampling_hz": info["fps"] / args.stride,
        "bimanual_required": args.bimanual,
        "rules": {
            "horizontal_tolerance_deg": 20,
            "prior_downward_max_deg": 45,
            "lookback_s": 4,
            "minimum_horizontal_s": 0.2,
            "minimum_nominal_grasp_rise_m": 0.025,
            "minimum_tilt_change_deg": 25,
            "minimum_bimanual_horizontal_s": 0.3 if args.bimanual else None,
        },
        "pool_episodes": len(pool),
        "pool_frames": sum(i["length"] for i in inventory),
        "pool_hours": sum(i["length"] for i in inventory) / info["fps"] / 3600,
        "candidate_episodes": len(candidates),
        "candidate_events": sum(len(i["events"]) for i in candidates),
        "video_decode_audit": "not_performed_by_pose_scan",
        "semantic_review": "pending",
        "training_ready": False,
        "elapsed_s": time.monotonic() - started,
    }
    for name, value in [
        ("summary.json", summary),
        ("candidates.json", candidates),
        ("source_files.json", source_files),
    ]:
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("pool_episodes", "pool_hours", "candidate_episodes", "candidate_events", "elapsed_s")
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--bimanual", action="store_true")
    scan(parser.parse_args())


if __name__ == "__main__":
    main()
