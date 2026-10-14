"""Resume complete-episode data audits in CPU workers; never import a trainer."""

# ruff: noqa: PLC0415

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from scripts.prepare_lego_separation_replay import decode_video
from scripts.select_lego_separation_episodes import ForwardKinematics
from scripts.select_lego_separation_episodes import find_bimanual_events
from scripts.select_lego_separation_episodes import find_events
from scripts.select_lego_separation_episodes import sha256


def audit_one(item, model):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    path = Path(item["data_path"])
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    digest = sha256(path)
    table = pq.read_table(
        path, columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"], use_threads=False
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
    events = []
    if item["pool"] == "specialist":
        fk = ForwardKinematics(Path(model))
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
    videos = {role: decode_video(ref, n) for role, ref in item["videos"].items()}
    if (path.stat().st_size, path.stat().st_mtime_ns) != before:
        raise ValueError("parquet changed during audit")
    return {
        "episode_index": item["episode_index"],
        "pool": item["pool"],
        "status": "passed",
        "numeric_frames": n,
        "data_sha256": digest,
        "data_size": before[0],
        "data_mtime_ns": before[1],
        "full_rate_bimanual_events": len(events) if events else None,
        "full_rate_events": events[:3],
        "videos": videos,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed", type=Path)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("bounded CPU worker count required")
    manifest = json.loads(args.dataset.read_text())
    source = Path(manifest["source"])
    if (
        sha256(source / "meta/info.json") != manifest["source_info_sha256"]
        or sha256(source / "conversion_manifest.json") != manifest["source_manifest_sha256"]
    ):
        raise ValueError("dataset identity changed")
    args.output.mkdir(parents=True, exist_ok=False)
    by_id = {i["episode_index"]: i for i in manifest["episodes"]}
    results = {}
    if args.completed:
        for line in args.completed.read_text().splitlines():
            prior = json.loads(line)
            item = by_id[prior["episode_index"]]
            if prior["status"] != "passed" or sha256(item["data_path"]) != prior["data_sha256"]:
                raise ValueError("completed numeric audit identity changed")
            for role, reference in item["videos"].items():
                path = Path(reference["path"])
                if (path.stat().st_size, path.stat().st_mtime_ns) != (reference["size"], reference["mtime_ns"]):
                    raise ValueError("completed video audit signature changed")
                if sha256(path) != prior["videos"][role]["sha256"]:
                    raise ValueError("completed video audit identity changed")
            results[item["episode_index"]] = prior
    pending = [i for i in manifest["episodes"] if i["episode_index"] not in results]
    started = time.monotonic()
    print(f"RESUME reuse={len(results)} pending={len(pending)} workers={args.workers}", flush=True)
    with (args.output / "audit.jsonl").open("x") as stream, ProcessPoolExecutor(max_workers=args.workers) as executor:
        for prior in results.values():
            stream.write(json.dumps(prior) + "\n")
        futures = [executor.submit(audit_one, item, str(args.model)) for item in pending]
        from concurrent.futures import as_completed

        for future in as_completed(futures):
            result = future.result()
            results[result["episode_index"]] = result
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            if len(results) % 10 == 0:
                print(f"AUDIT {len(results)}/{len(by_id)} elapsed_s={time.monotonic() - started:.1f}", flush=True)
    if set(results) != set(by_id):
        raise ValueError("incomplete audit")
    summary = {
        "schema": "yam_lego_specialist_replay_audit_v2",
        "status": "passed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "episodes": len(results),
        "specialist_episodes_full_rate_pose_passed": sum(i["pool"] == "specialist" for i in by_id.values()),
        "numeric_frames": sum(i["length"] for i in by_id.values()),
        "video_episode_ranges": len(results) * 3,
        "decoded_frames": sum(i["length"] for i in by_id.values()) * 3,
        "pool_stats": manifest["pool_stats"],
        "dataset_manifest_sha256": sha256(args.dataset),
        "audit_jsonl_sha256": sha256(args.output / "audit.jsonl"),
        "script_sha256": sha256(__file__),
        "reused_completed_audit": str(args.completed) if args.completed else None,
        "reused_completed_audit_sha256": sha256(args.completed) if args.completed else None,
        "elapsed_s": time.monotonic() - started,
        "separation_success_verified": False,
        "independent_robot_units_and_capture_timing_verified": False,
    }
    (args.output / "audit-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("DATA_AUDIT_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
