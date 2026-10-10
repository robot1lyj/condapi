"""Read immutable YAM recordings and export grasp candidates and image jobs."""

import argparse
from collections import Counter
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import av
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/parts-rl/src"))
from parts_rl.candidates import closing_events


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("frames", "traces", "clips"):
        (args.output / name).mkdir()
    labels = json.loads((args.source / "operator_labels_v2.json").read_text())
    episodes, candidates, jobs, selections = [], [], {}, defaultdict(dict)
    for label in labels:
        folder = args.source / "raw" / label["source_path"]
        manifest = json.loads((folder / "manifest.json").read_text())
        arrays, records, segments = defaultdict(list), [], []
        for seg in manifest["segments"]:
            segment = folder / seg["path"]
            with h5py.File(segment / "samples.h5", "r") as handle:
                size = int(handle["committed_rows"][()])
                for key in ("time", "tick", "epoch", "measured_state", "measured_state__valid",
                            "observation_valid", "video_indices", "camera_host_received_at", "camera_sequence"):
                    arrays[key].append(handle[key][:size])
                records.extend(json.loads(x) for x in handle["details"][:size])
                segments.extend([seg["path"]] * size)
        a = {key: np.concatenate(parts) for key, parts in arrays.items()}
        n = len(records)
        number = label["operator_row"]
        ep = {"number": number, "episode_id": label["episode_id"], "source_path": label["source_path"],
              "split": "validation" if number == 1 else "train_candidate", "rows": n,
              "start_time": float(a["time"][0]), "operator_whole_task_counts": {
                  key: label[key] for key in ("initial_remaining", "newly_correct", "wrong", "unplaced")},
              "behavior_identity_status": label["behavior_identity_status"]}
        episodes.append(ep)
        trace = {"time": a["time"].tolist(), "tick": a["tick"].tolist(), "epoch": a["epoch"].tolist(),
                 "arms": {}}
        for arm, axis, column in (("left", 6, 1), ("right", 13, 2)):
            poses = [record.get("parts", {}).get("arms", {}).get(arm, {}) for record in records]
            z = np.array([p.get("position", [None] * 3)[2] if p.get("pose_valid") else np.nan for p in poses], float)
            g = a["measured_state"][:, axis]
            valid = np.isfinite(z) & np.isfinite(g) & a["measured_state__valid"] & a["observation_valid"]
            trace["arms"][arm] = {"openness": g.tolist(), "base_z_m": [float(x) if np.isfinite(x) else None for x in z],
                                   "valid": valid.tolist()}
            events = closing_events(a["time"], g, valid, a["epoch"], a["tick"])
            for serial, event in enumerate(events, 1):
                entry, close, end = (event[k] for k in ("entry", "close", "end"))
                # Use the recorded proposal entry, never a future minimum height.
                ref = entry
                cid = f"E{number:02d}-{arm[0].upper()}{serial:03d}"
                candidate = {"candidate_id": cid, "episode_number": number, "episode_id": ep["episode_id"],
                             "arm": arm, "split": ep["split"], "entry_row": entry, "close_row": close,
                             "end_row": end, "end_reason": event["end_reason"],
                             "entry_time": float(a["time"][entry]), "close_time": float(a["time"][close]),
                             "end_time": float(a["time"][end]), "reference_row": ref, "reference_base_z_m": float(z[ref]),
                             "reference_method": "candidate_entry_base_z",
                             "pose_frame": poses[ref].get("pose_frame"), "pose_ref": poses[ref].get("pose_ref"),
                             "up_axis_status": "base_z_provisional_not_table_calibrated",
                             "suggested_label": "uncertain", "review_label": None, "approved_reward": None,
                             "samples": [], "all_rewards_unapproved": True}
                selected = {entry, close, ref, end}
                for target in np.arange(a["time"][entry], a["time"][end] + 0.001, 0.20):
                    row = int(np.searchsorted(a["time"], target))
                    if row <= end:
                        selected.add(row)
                for row in sorted(selected):
                    index = int(a["video_indices"][row, column])
                    segment = segments[row]
                    frame_key = f"E{number:02d}-{segment}-{arm}-{index:06d}"
                    path = f"frames/{frame_key}.jpg"
                    visual = float(a["camera_host_received_at"][row, column])
                    visual_valid = bool(np.isfinite(visual))
                    sample = {"row": row, "tick": int(a["tick"][row]), "epoch": int(a["epoch"][row]),
                              "time": float(a["time"][row]), "visual_time": visual if visual_valid else None,
                              "frame_key": frame_key, "image": path, "video_frame_index": index,
                              "source_segment": segment, "source_video": str(folder / segment / f"{arm}.mp4"),
                              "openness": float(g[row]), "base_z_m": float(z[row]) if np.isfinite(z[row]) else None,
                              "valid": bool(valid[row] and visual_valid), "camera_sequence": float(a["camera_sequence"][row, column])}
                    candidate["samples"].append(sample)
                    jobs[frame_key] = {"frame_key": frame_key, "image": path, "arm": arm}
                    selections[sample["source_video"]][index] = (frame_key, path)
                candidates.append(candidate)
        write_json(args.output / "traces" / f"E{number:02d}.json", trace)
        print(json.dumps({"episode": number, "rows": n, "candidates": sum(c["episode_number"] == number for c in candidates)}), flush=True)
    for video, picks in selections.items():
        found = set()
        with av.open(video) as container:
            container.streams.video[0].codec_context.thread_count = 2
            for index, frame in enumerate(container.decode(video=0)):
                if index in picks:
                    key, path = picks[index]
                    frame.to_image().convert("RGB").save(args.output / path, quality=88)
                    jobs[key].update(sha256=hashlib.sha256((args.output / path).read_bytes()).hexdigest(),
                                     source_video=video, video_frame_index=index, pts=frame.pts, time_base=str(frame.time_base))
                    found.add(index)
                if index >= max(picks):
                    break
        if found != set(picks):
            raise RuntimeError(f"missing selected video frames: {video}")
    write_json(args.output / "episodes.json", episodes)
    write_json(args.output / "candidates-unscored.json", candidates)
    write_json(args.output / "image-jobs.json", list(jobs.values()))
    write_json(args.output / "preparation.json", {
        "schema": "parts_grasp_candidate_preparation_v1", "source": str(args.source),
        "episodes": len(episodes), "candidates": len(candidates), "images": len(jobs),
        "per_arm": dict(Counter(c["arm"] for c in candidates)), "all_rewards_unapproved": True,
        "training_executed": False, "up_axis_status": "provisional_base_z_requires_review",
        "sampling_interval_s": 0.20, "lift_delta_m": 0.05, "visual_hold_s": 1.0,
        "note": "Closures are proposals, including possible adjustments or non-grasps; no terminal C/N is reused as grasp reward."})
    print(json.dumps({"complete": True, "candidates": len(candidates), "images": len(jobs)}), flush=True)


if __name__ == "__main__":
    main()
