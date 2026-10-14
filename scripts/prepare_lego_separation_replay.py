"""Freeze and audit an offline specialist/replay selection; never train.

References the existing read-only LeRobot train split, preserves complete
episodes, and balances replay duration without resampling or rewriting rows.
The resulting episode list is usable by the native subset loader. Pool ratios
are measured in frames; this does not introduce a weighted training sampler.
"""

# ruff: noqa: PLC0415

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC
from datetime import datetime
import json
from pathlib import Path
import random
import time

import numpy as np

from scripts.select_lego_separation_episodes import ForwardKinematics
from scripts.select_lego_separation_episodes import find_bimanual_events
from scripts.select_lego_separation_episodes import find_events
from scripts.select_lego_separation_episodes import selection
from scripts.select_lego_separation_episodes import sha256


def balanced_replay(base_ids, metadata, target_frames, count, seed):
    """Seeded ten source-index strata, then swaps within strata to match time."""
    groups = [list(map(int, group)) for group in np.array_split(sorted(base_ids), 10)]
    rng = random.Random(seed)
    sizes = [len(group) for group in np.array_split(np.arange(count), 10)]
    chosen = [rng.sample(group, size) for group, size in zip(groups, sizes, strict=True)]
    total = sum(metadata[i]["length"] for group in chosen for i in group)
    for _ in range(200):
        error, best = abs(total - target_frames), None
        if error <= 1:
            break
        for g, group in enumerate(groups):
            unused = sorted(set(group) - set(chosen[g]))
            for slot, old in enumerate(chosen[g]):
                for new in unused:
                    change = metadata[new]["length"] - metadata[old]["length"]
                    if abs(total + change - target_frames) < error:
                        error, best = abs(total + change - target_frames), (g, slot, new, change)
        if best is None:
            break
        g, slot, new, change = best
        chosen[g][slot] = new
        total += change
    return sorted(i for group in chosen for i in group)


def decode_video(reference, length):
    import av

    path = Path(reference["path"])
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    if before != (reference["size"], reference["mtime_ns"]):
        raise ValueError("video identity changed")
    start, count = reference["start_frame"], 0
    with av.open(str(path)) as reader:
        video = reader.streams.video[0]
        video.codec_context.thread_count = 1
        reader.seek(int(start / 30 / video.time_base), stream=video, backward=True)
        for frame in reader.decode(video):
            if frame.pts is None:
                raise ValueError("missing video PTS")
            actual = float(frame.pts * frame.time_base * 30)
            index = round(actual)
            if abs(actual - index) > 0.05:
                raise ValueError("video frame off 30fps clock")
            if index < start:
                continue
            if index >= start + length:
                break
            if index != start + count or (frame.width, frame.height) != (224, 224):
                raise ValueError("video frame gap/duplicate or unexpected image size")
            count += 1
    if count != length:
        raise ValueError(f"incomplete episode video {count} != {length}")
    digest = sha256(path)
    if (path.stat().st_size, path.stat().st_mtime_ns) != before:
        raise ValueError("video changed during decode")
    return {"path": str(path), "sha256": digest, "decoded_frames": count, "start_frame": start, "status": "passed"}


def prepare(args):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    source, output = args.source.resolve(), args.output.resolve()
    if output.is_relative_to(source) or source.name != "train":
        raise ValueError("require independent output and train source")
    output.mkdir(parents=True, exist_ok=False)
    info_path, conversion_path = source / "meta/info.json", source / "conversion_manifest.json"
    info, conversion = json.loads(info_path.read_text()), json.loads(conversion_path.read_text())
    summary = json.loads((args.scan / "summary.json").read_text())
    if not summary["bimanual_required"] or summary["source"] != str(source):
        raise ValueError("requires matching bimanual scan")
    for path, digest in summary["input_identities"].items():
        if sha256(path) != digest:
            raise ValueError("scan identity no longer matches source")
    base = selection(args.base_selection)
    if sha256(args.base_selection) != sha256(summary["excluded_selection"]):
        raise ValueError("replay pool differs from excluded 50h")
    metadata = {
        r["episode_index"]: r for r in pq.read_table(sorted((source / "meta/episodes").rglob("*.parquet"))).to_pylist()
    }
    origins = {r["episode_index"]: r for r in conversion["episodes"]}
    rejected = set(json.loads(args.reject.read_text())) if args.reject else set()
    candidates = json.loads((args.scan / "candidates.json").read_text())
    chosen = [i for i in candidates if i["episode_index"] not in rejected][: args.count]
    if len(chosen) != args.count or any(i["episode_index"] in base for i in chosen):
        raise ValueError("insufficient distinct specialist episodes outside base")
    specialists = {i["episode_index"]: i for i in chosen}
    replay_ids = balanced_replay(base, metadata, sum(i["length"] for i in chosen), args.count, args.seed)
    all_ids = sorted(set(specialists) | set(replay_ids))
    if len(all_ids) != 2 * args.count:
        raise ValueError("overlapping selection pools")
    records = []
    for episode in all_ids:
        row, origin = metadata[episode], origins[episode]
        item = {
            "episode_index": episode,
            "source_episode_index": origin["source_episode_index"],
            "source_revision": origin["source_revision"],
            "length": row["length"],
            "full_h50_windows": max(0, row["length"] - 49),
            "pool": "specialist" if episode in specialists else "replay",
            "data_path": str(
                source
                / info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"])
            ),
            "events": specialists[episode]["events"][:3] if episode in specialists else [],
            "semantic_review": "pose_candidate_not_success_label",
            "videos": {},
        }
        for role in ("top", "left", "right"):
            key = f"observation.images.{role}_rgb"
            prefix = f"videos/{key}/"
            video = source / info["video_path"].format(
                video_key=key, chunk_index=row[prefix + "chunk_index"], file_index=row[prefix + "file_index"]
            )
            item["videos"][role] = {
                "path": str(video),
                "start_frame": round(row[prefix + "from_timestamp"] * 30),
                "size": video.stat().st_size,
                "mtime_ns": video.stat().st_mtime_ns,
            }
        records.append(item)
    pool_stats = {}
    for name in ("specialist", "replay"):
        members = [i for i in records if i["pool"] == name]
        pool_stats[name] = {
            "episodes": len(members),
            "frames": sum(i["length"] for i in members),
            "full_h50_windows": sum(i["full_h50_windows"] for i in members),
        }
        pool_stats[name]["hours"] = pool_stats[name]["frames"] / 108000
    total_frames = sum(s["frames"] for s in pool_stats.values())
    for stats in pool_stats.values():
        stats["uniform_frame_sampler_probability"] = stats["frames"] / total_frames
    manifest = {
        "schema": "yam_lego_specialist_replay_v1",
        "source": str(source),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_info_sha256": sha256(info_path),
        "source_manifest_sha256": sha256(conversion_path),
        "base_selection_sha256": sha256(args.base_selection),
        "scan_summary_sha256": sha256(args.scan / "summary.json"),
        "selection_algorithm": "ranked bimanual lift/turn candidates; seeded 10-stratum replay with within-stratum duration balancing",
        "seed": args.seed,
        "source_values_modified": False,
        "complete_episodes": True,
        "validation_split_included": False,
        "new_specialist_split": "not_assigned; preserve existing independent val",
        "semantic_success_labels": "unknown; no automatic success certification",
        "pool_stats": pool_stats,
        "training_recipe_status": "awaiting_explicit_recipe_confirmation; no config or training job generated",
        "episodes": records,
    }
    for name, ids in [
        ("specialist-episodes.json", sorted(specialists)),
        ("replay-episodes.json", replay_ids),
        ("episodes.json", all_ids),
    ]:
        (output / name).write_text(json.dumps(ids, indent=2) + "\n")
    (output / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("SELECTION", json.dumps(pool_stats), flush=True)
    fk = ForwardKinematics(args.model.resolve())
    scan_files = json.loads((args.scan / "source_files.json").read_text())
    started = time.monotonic()
    audits = []
    with (output / "audit.jsonl").open("x") as stream, ThreadPoolExecutor(max_workers=2) as executor:
        for number, item in enumerate(records, 1):
            path = Path(item["data_path"])
            digest = sha256(path)
            if item["pool"] == "specialist" and digest != scan_files[str(path)]["sha256"]:
                raise ValueError("specialist parquet differs from scan")
            table = pq.read_table(
                path, columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"]
            )
            data = table.filter(pc.equal(table["episode_index"], item["episode_index"])).to_pydict()
            n = item["length"]
            state, action = np.asarray(data["observation.state"]), np.asarray(data["action"])
            if (
                state.shape != (n, 14)
                or action.shape != (n, 14)
                or not np.isfinite(state).all()
                or not np.isfinite(action).all()
                or data["frame_index"] != list(range(n))
                or not np.allclose(data["timestamp"], np.arange(n) / 30, atol=1e-3, rtol=0)
            ):
                raise ValueError(f"invalid numeric/timing rows {item['episode_index']}")
            event_count = None
            if item["pool"] == "specialist":
                frames = np.arange(n)
                poses = fk.poses(state)
                unilateral = []
                for arm, name in enumerate(("left", "right")):
                    unilateral.extend(
                        {**e, "arm": name}
                        for e in find_events(
                            poses[:, arm, 0], poses[:, arm, 1], poses[:, arm, 2], state[:, arm * 7 + 6], frames, 30, 1
                        )
                    )
                events = find_bimanual_events(poses, state, frames, 30, 1, unilateral)
                if not events:
                    raise ValueError(f"no full-rate bimanual event {item['episode_index']}")
                event_count = len(events)
                item["full_rate_events"] = events[:3]
            futures = {role: executor.submit(decode_video, ref, n) for role, ref in item["videos"].items()}
            videos = {role: future.result() for role, future in futures.items()}
            audit = {
                "episode_index": item["episode_index"],
                "pool": item["pool"],
                "status": "passed",
                "numeric_frames": n,
                "data_sha256": digest,
                "full_rate_bimanual_events": event_count,
                "videos": videos,
            }
            stream.write(json.dumps(audit) + "\n")
            stream.flush()
            audits.append(audit)
            if number % 10 == 0:
                print(f"AUDIT {number}/{len(records)} elapsed_s={time.monotonic() - started:.1f}", flush=True)
    result = {
        "schema": "yam_lego_specialist_replay_audit_v1",
        "status": "passed",
        "episodes": len(audits),
        "specialist_episodes_full_rate_pose_passed": args.count,
        "numeric_frames": total_frames,
        "video_episode_ranges": len(audits) * 3,
        "decoded_frames": total_frames * 3,
        "pool_stats": pool_stats,
        "elapsed_s": time.monotonic() - started,
        "script_sha256": sha256(__file__),
        "selector_script_sha256": sha256(Path(__file__).with_name("select_lego_separation_episodes.py")),
        "audit_jsonl_sha256": sha256(output / "audit.jsonl"),
        "separation_success_verified": False,
        "independent_robot_units_and_capture_timing_verified": False,
    }
    manifest["integrity_audit"] = "passed; full numeric rows, three video ranges, full-rate specialist pose"
    (output / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "audit-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    identities = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    (output / "sha256.json").write_text(json.dumps(identities, indent=2) + "\n")
    print("DATA_AUDIT_COMPLETE", json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "scan", "base-selection", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reject", type=Path)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
