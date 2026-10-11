"""Reconstruct physical attempts and synchronized evidence; CPU only, new output."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import av
import h5py
import numpy as np
from PIL import Image
from PIL import ImageDraw


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def physical_events(times, g, z, state_valid, epochs, ticks):
    # Missing camera observations must not reset a physically continuous attempt.
    physical = state_valid & np.isfinite(g) & np.isfinite(z)
    events = []
    active = None
    peak = None
    peak_row = None
    armed = False
    minimum = 1.0
    for row in range(len(times)):
        connected = row == 0 or (
            epochs[row] == epochs[row - 1]
            and ticks[row] == ticks[row - 1] + 1
            and 0 < times[row] - times[row - 1] <= 0.15
        )
        if not physical[row] or not connected:
            if active is not None:
                events[active].update(end=max(events[active]["close"], row - 1), end_reason="trace_gap")
            active = None
            peak = None
            peak_row = None
            armed = False
            if not physical[row]:
                continue
        value = float(g[row])
        if active is not None:
            minimum = min(minimum, value)
            if value >= 0.65 and value - minimum >= 0.18 and times[row] - times[events[active]["close"]] >= 0.2:
                events[active].update(end=row, end_reason="reopened")
                active = None
                peak = value
                peak_row = row
                armed = True
            continue
        if peak is None or value >= peak:
            peak = value
            peak_row = row
        armed |= value >= 0.65
        # Retain the open reference through slow closure. The old 0.8s peak
        # expiry lost real attempts even after camera-gap repair.
        if armed and value <= 0.62 and peak - value >= 0.18:
            events.append(
                {"entry": max(peak_row, row - 15), "close": row, "end": len(times) - 1, "end_reason": "episode_end"}
            )
            active = len(events) - 1
            minimum = value
            armed = False
    return events


def select_rows(times, close, end, valid, z):
    available = np.flatnonzero(valid & (np.arange(len(times)) >= max(0, close - 15)) & (np.arange(len(times)) <= end))
    if not len(available):
        return []
    duration = times[end] - times[close]
    targets = [
        times[close] - 0.25,
        times[close],
        times[close] + 0.20,
        times[close] + 0.45,
        times[close] + 0.75,
        times[close] + 1.15,
        times[close] + 1.7,
        times[close] + duration * 0.65,
        times[end],
    ]
    post = available[available >= close]
    if len(post):
        targets.append(times[post[np.argmax(z[post])]])
    return sorted({int(available[np.argmin(abs(times[available] - t))]) for t in targets if t <= times[end] + 0.02})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reuse-images", type=Path)
    args = p.parse_args()
    root = args.output
    root.mkdir(exist_ok=False)
    for sub in ["images", "montages", "traces", "run-scripts"]:
        (root / sub).mkdir()
    labels = json.loads((args.source / "operator_labels_v2.json").read_text())
    previous = json.loads((args.previous / "candidates.json").read_text())
    attempts = []
    image_jobs = {}
    picks = defaultdict(dict)
    episodes = []
    source_hashes = []
    for lab in labels:
        ep = lab["operator_row"]
        folder = args.source / "raw" / lab["source_path"]
        manifest = json.loads((folder / "manifest.json").read_text())
        arrays = defaultdict(list)
        details = []
        segments = []
        for seg in manifest["segments"]:
            h5 = folder / seg["path"] / "samples.h5"
            with h5.open("rb") as f:
                source_hashes.append({"path": str(h5), "sha256": hashlib.file_digest(f, "sha256").hexdigest()})
            with h5py.File(h5, "r") as h:
                n = int(h["committed_rows"][()])
                for key in [
                    "time",
                    "tick",
                    "epoch",
                    "measured_state",
                    "measured_state__valid",
                    "observation_valid",
                    "video_indices",
                    "camera_host_received_at",
                    "camera_sequence",
                ]:
                    arrays[key].append(h[key][:n])
                details.extend(json.loads(x) for x in h["details"][:n])
                segments.extend([seg["path"]] * n)
        a = {k: np.concatenate(v) for k, v in arrays.items()}
        t = a["time"]
        n_rows = len(t)
        eprec = {
            "number": ep,
            "episode_id": lab["episode_id"],
            "source_path": lab["source_path"],
            "start_time": float(t[0]),
            "rows": n_rows,
            "split": "validation" if ep == 1 else "train_candidate",
        }
        episodes.append(eprec)
        traces = {
            "time": t.tolist(),
            "epoch": a["epoch"].tolist(),
            "tick": a["tick"].tolist(),
            "observation_valid": a["observation_valid"].tolist(),
            "arms": {},
        }
        for arm, axis, col in [("left", 6, 1), ("right", 13, 2)]:
            poses = [d.get("parts", {}).get("arms", {}).get(arm, {}) for d in details]
            xyz = np.array(
                [q.get("position", [np.nan] * 3) if q.get("pose_valid") else [np.nan] * 3 for q in poses], float
            )
            z = xyz[:, 2]
            g = a["measured_state"][:, axis]
            events = physical_events(t, g, z, a["measured_state__valid"], a["epoch"], a["tick"])
            traces["arms"][arm] = {"openness": g.tolist(), "position_m": xyz.tolist()}
            for serial, event in enumerate(events, 1):
                entry, close, end = [event[k] for k in ["entry", "close", "end"]]
                # Stop result evidence before release, rather than classifying normal empty-after-placement.
                running_min = np.minimum.accumulate(g[close : end + 1])
                release = np.flatnonzero(
                    (g[close : end + 1] - running_min >= 0.18)
                    & (g[close : end + 1] >= 0.5)
                    & (t[close : end + 1] - t[close] >= 0.2)
                )
                inspect_end = max(close, int(close + release[0] - 1)) if len(release) else end
                fresh = (
                    (a["video_indices"][:, col] >= 0)
                    & (a["camera_host_received_at"][:, col] > 0)
                    & (t - a["camera_host_received_at"][:, col] >= 0)
                    & (t - a["camera_host_received_at"][:, col] <= 0.3)
                    & np.isfinite(z)
                )
                rows = select_rows(t, close, inspect_end, fresh, z)
                cid = f"E{ep:02d}-{arm[0].upper()}{serial:03d}-v3"
                rec = {
                    "candidate_id": cid,
                    "episode_number": ep,
                    "episode_id": lab["episode_id"],
                    "arm": arm,
                    "split": eprec["split"],
                    "entry_row": entry,
                    "close_row": close,
                    "end_row": end,
                    "inspection_end_row": inspect_end,
                    "end_reason": event["end_reason"],
                    "close_time": float(t[close]),
                    "end_time": float(t[end]),
                    "reference_row": close,
                    "reference_base_z_m": float(z[close]),
                    "reference_method": "recorded_closure_row_not_future_minimum",
                    "frames": [],
                    "suggested_label": None,
                    "review_label": None,
                    "approved_reward": None,
                }
                for row in rows:
                    refs = {}
                    for view, column in [("top", 0), ("left", 1), ("right", 2)]:
                        index = int(a["video_indices"][row, column])
                        capture = float(a["camera_host_received_at"][row, column])
                        if index < 0 or capture <= 0 or not 0 <= t[row] - capture <= 0.3:
                            continue
                        key = f"E{ep:02d}-{segments[row]}-{view}-{index:06d}"
                        name = f"images/{key}.jpg"
                        video = str(folder / segments[row] / f"{view}.mp4")
                        image_jobs[key] = {
                            "id": key,
                            "image": name,
                            "episode_number": ep,
                            "view": view,
                            "source_video": video,
                            "video_frame_index": index,
                        }
                        picks[video][index] = key
                        refs[view] = {"image": name, "key": key, "video_frame_index": index, "capture_time": capture}
                    rec["frames"].append(
                        {
                            "row": row,
                            "tick": int(a["tick"][row]),
                            "epoch": int(a["epoch"][row]),
                            "time": float(t[row]),
                            "since_close_s": float(t[row] - t[close]),
                            "openness": float(g[row]),
                            "position_m": xyz[row].tolist(),
                            "lift_from_closure_m": float(z[row] - z[close]),
                            "views": refs,
                        }
                    )
                old = [
                    c["candidate_id"]
                    for c in previous
                    if c["episode_number"] == ep
                    and c["arm"] == arm
                    and (close - 20 <= c["close_row"] <= end or abs(c["close_row"] - close) <= 20)
                ]
                rec["legacy_candidate_ids"] = old
                attempts.append(rec)
        write(root / "traces" / f"E{ep:02d}.json", traces)
        print(
            json.dumps({"episode": ep, "reconstructed": sum(c["episode_number"] == ep for c in attempts)}), flush=True
        )
    cache_map = {}
    if args.reuse_images:
        cache_map = {j["id"]: j for j in json.loads((args.reuse_images / "image-jobs.json").read_text())}
    for video, rows in picks.items():
        found = set()
        if args.reuse_images:
            for i, key in rows.items():
                cached = args.reuse_images / image_jobs[key]["image"]
                if cached.exists():
                    old = cache_map[key]
                    assert old["source_video"] == video
                    assert old["video_frame_index"] == i
                    assert hashlib.sha256(cached.read_bytes()).hexdigest() == old["sha256"]
                    shutil.copyfile(cached, root / image_jobs[key]["image"])
                    image_jobs[key].update({k: old[k] for k in ["sha256", "pts", "time_base"]})
                    found.add(i)
        remaining = set(rows) - found
        if not remaining:
            continue
        with av.open(video) as c:
            c.streams.video[0].codec_context.thread_count = 2
            for i, f in enumerate(c.decode(video=0)):
                if i in remaining:
                    key = rows[i]
                    dest = root / image_jobs[key]["image"]
                    f.to_image().convert("RGB").save(dest, quality=94)
                    image_jobs[key].update(
                        sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), pts=f.pts, time_base=str(f.time_base)
                    )
                    found.add(i)
                if i >= max(remaining):
                    break
        assert found == set(rows), video
    for rec in attempts:
        frames = rec["frames"]
        canvas = Image.new("RGB", (960, 30 + len(frames) * 264), "#eef2f7")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"{rec['candidate_id']} | {rec['arm']} grasp | WRIST / CENTER ZOOM / TOP", fill="#111827")
        for i, f in enumerate(frames):
            y = 30 + i * 264
            draw.text(
                (8, y + 3),
                f"row {f['row']} t={f['since_close_s']:+.2f}s g={f['openness']:.2f} dz={f['lift_from_closure_m']:+.3f}m",
                fill="#111827",
            )
            if rec["arm"] in f["views"]:
                im = Image.open(root / f["views"][rec["arm"]]["image"]).convert("RGB")
                canvas.paste(im.resize((320, 240)), (0, y + 24))
                canvas.paste(im.crop((160, 180, 480, 480)).resize((320, 240)), (320, y + 24))
            if "top" in f["views"]:
                canvas.paste(
                    Image.open(root / f["views"]["top"]["image"]).convert("RGB").resize((320, 240)), (640, y + 24)
                )
        name = f"montages/{rec['candidate_id']}.jpg"
        canvas.save(root / name, quality=94)
        rec["montage"] = name
        rec["montage_sha256"] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    mapping = [
        {
            "old_id": c["candidate_id"],
            "new_ids": [r["candidate_id"] for r in attempts if c["candidate_id"] in r["legacy_candidate_ids"]],
        }
        for c in previous
    ]
    write(root / "candidates-prepared.json", attempts)
    write(root / "episodes.json", episodes)
    write(root / "image-jobs.json", list(image_jobs.values()))
    write(root / "legacy-id-mapping.json", mapping)
    write(root / "source-h5-hashes.json", source_hashes)
    write(
        root / "preparation.json",
        {
            "schema": "physical_grasp_review_v3",
            "source": str(args.source),
            "previous": str(args.previous),
            "episodes": len(episodes),
            "attempt_proposals": len(attempts),
            "images": len(image_jobs),
            "legacy_candidates": len(previous),
            "legacy_unmapped": sum(not m["new_ids"] for m in mapping),
            "tracking": False,
            "training_executed": False,
            "camera_gaps_do_not_terminate_physical_attempts": True,
            "height_gate": False,
            "reference": "closure row only; height is supporting evidence",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    print(json.dumps({"complete": True, "attempts": len(attempts), "images": len(image_jobs)}), flush=True)


if __name__ == "__main__":
    main()
