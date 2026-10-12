"""Portable offline CPU curator for original HIL episodes (Thor or GPU server).

First run exports candidates and actual windows. After Qwen/DINO and review,
rerun into a NEW release with --rewards to produce training records/rewards.
Never runs a neural network or changes any source episode.
"""

# ruff: noqa: E402

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/pld-rl/src"))
from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.curator import associate_rewards
from yam_pld_rl.curator import physical_candidates
from yam_pld_rl.curator import reconstruct_intervals
from yam_pld_rl.data import sha256
from yam_pld_rl.data import write_json
from yam_pld_rl.kinematics import YamKinematics


def read_episode(folder):
    import h5py  # noqa: PLC0415 -- needed only by the portable offline entrypoint

    rows, hashes = [], {}
    required = ("time", "tick", "epoch", "measured_state", "observation_state", "submitted_action", "is_intervention",
                "video_indices", "camera_host_received_at")
    for path in sorted(folder.glob("segment_*/samples.h5")):
        hashes[str(path)] = sha256(path)
        with h5py.File(path) as f:
            n = int(f["committed_rows"][()])
            if n < 0 or any(len(f[k]) < n for k in (*required, "details")):
                raise ValueError("invalid committed row count")
            data = {k: f[k][:n] for k in required}
            masks = {k: f[k + "__valid"][:n] for k in required if k + "__valid" in f}
            for i, raw in enumerate(f["details"][:n]):
                details = json.loads(raw)
                clock_valid = all(bool(masks[k][i]) for k in ("time", "tick", "epoch"))
                rows.append({"time": float(data["time"][i]), "tick": int(data["tick"][i]), "epoch": int(data["epoch"][i]),
                             "episode_id": folder.name, "measured_state": data["measured_state"][i].tolist(),
                             "measured_state_valid": clock_valid and bool(masks["measured_state"][i]),
                             "submitted_action": data["submitted_action"][i].tolist(),
                             "submitted_valid": clock_valid and bool(masks["submitted_action"][i]),
                             "is_intervention": bool(data["is_intervention"][i]),
                             "intervention_valid": bool(masks["is_intervention"][i]),
                             "camera_times": data["camera_host_received_at"][i].tolist(),
                             "camera_valid": masks["camera_host_received_at"][i].tolist(),
                             "video_indices": data["video_indices"][i].tolist(), "details": details,
                             "raw_ref": f"{folder.parent.name}/{folder.name}/{path.parent.name}/samples.h5#row={i}",
                             "videos": {v: str(path.parent / f"{v}.mp4") for v in ("top", "left", "right")}})
    if not rows:
        raise ValueError(f"no committed native HIL rows: {folder}")
    # Capture provenance for the original videos without copying or modifying them.
    for video in sorted({p for row in rows for p in row["videos"].values()}):
        if not Path(video).is_file() or not Path(video).stat().st_size:
            raise ValueError(f"missing source video; cannot publish visual provenance: {video}")
        hashes[video] = sha256(video)
    return rows, hashes


def export_evidence(candidates, output):
    """Produce the existing Qwen/DINO prepared-candidate image interface."""
    import av  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from PIL import ImageDraw  # noqa: PLC0415

    (output / "images").mkdir()
    (output / "montages").mkdir()
    jobs, picks = {}, defaultdict(dict)
    for c in candidates:
        for f in c["frames"]:
            for view, ref in f["views"].items():
                key = f"E{c['episode_number']:02d}-{Path(ref['source_video']).parent.name}-{view}-{ref['video_frame_index']:06d}"
                ref.update(key=key, image=f"images/{key}.jpg")
                jobs[key] = {"id": key, "image": ref["image"], "episode_number": c["episode_number"],
                             "source_video": ref["source_video"], "video_frame_index": ref["video_frame_index"], "view": view}
                picks[ref["source_video"]][ref["video_frame_index"]] = key
    for video, wanted in picks.items():
        found = set()
        with av.open(video) as container:
            container.streams.video[0].codec_context.thread_count = 2
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    key = wanted[index]
                    path = output / jobs[key]["image"]
                    frame.to_image().convert("RGB").save(path, quality=94)
                    jobs[key].update(sha256=sha256(path), pts=frame.pts, time_base=str(frame.time_base))
                    found.add(index)
                if index >= max(wanted):
                    break
        if found != set(wanted):
            raise ValueError(f"video does not contain claimed frames: {video}")
    for c in candidates:
        frames = c["frames"]
        if not frames:
            continue
        montage = Image.new("RGB", (960, 30 + 260 * len(frames)), "#eef2f7")
        draw = ImageDraw.Draw(montage)
        draw.text((8, 8), c["candidate_id"], fill="#111827")
        for i, f in enumerate(frames):
            y = 30 + 260 * i
            draw.text((8, y), f"row={f['row']} t={f['since_close_s']:+.2f}s dz={f['lift_from_closure_m']:+.3f}m", fill="#111827")
            for col, view in enumerate((c["arm"], "top")):
                if view in f["views"]:
                    with Image.open(output / f["views"][view]["image"]) as image:
                        montage.paste(image.resize((320, 240)), (col * 320, y + 20))
        c["montage"] = f"montages/{c['candidate_id']}.jpg"
        montage.save(output / c["montage"], quality=94)
        c["montage_sha256"] = sha256(output / c["montage"])
    write_json(output / "image-jobs.json", list(jobs.values()))
    # Unqualified cases remain in candidates.json; do not feed evidence-free
    # attempts to a forced binary VLM prompt.
    write_json(output / "candidates-prepared.json", [c for c in candidates if c["evidence_eligible"] and c.get("montage")])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, nargs="+", required=True)
    p.add_argument("--validation-episodes", nargs="*", default=[], help="resolved episode paths; split at episode level")
    p.add_argument("--rewards", type=Path, help="reviewed candidate_id/status/approved/reviewer/evidence_rows JSON list")
    p.add_argument("--export-evidence", action="store_true", help="decode selected images for the existing Qwen/DINO tools")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    folders = [p.resolve() for p in args.episodes]
    if len(folders) != len(set(folders)):
        p.error("duplicate raw episode")
    validation = {str(Path(p).resolve()) for p in args.validation_episodes}
    if not validation.issubset({str(p) for p in folders}):
        p.error("validation episode is not an input episode")
    args.output.mkdir(parents=True, exist_ok=False)
    fk = YamKinematics(ROOT / "third_party/yam-fk-model/yam_linear4310.xml")
    approved = json.loads(args.rewards.read_text()) if args.rewards else []
    candidates, windows, records, labels, audit, hashes = [], [], [], [], [], {}
    for number, folder in enumerate(folders, 1):
        rows, source_hashes = read_episode(folder)
        hashes.update(source_hashes)
        c = physical_candidates(rows, fk, number)
        for candidate in c:
            candidate["source_release_sha256"] = fingerprint(source_hashes)
            candidate["candidate_sha256"] = fingerprint(candidate)
        w, rejected = reconstruct_intervals(rows)
        split = "val" if str(folder) in validation else "train"
        relevant = [r for r in approved if r["candidate_id"] in {q["candidate_id"] for q in c}]
        linked, rewards, association = associate_rewards(w, c, relevant, rows, split)
        candidates.extend(c)
        windows.extend(w)
        records.extend(linked)
        labels.extend(rewards)
        audit.append({"episode": str(folder), "split": split, "raw_rows": len(rows), "physical_candidates": len(c),
                      "actual_intervals": len(w), "execution_exclusions": rejected, "association_exclusions": association})
    if {r["candidate_id"] for r in approved} - {c["candidate_id"] for c in candidates}:
        raise ValueError("review contains candidates outside this exact episode release")
    if args.export_evidence:
        export_evidence(candidates, args.output)
    for name, value in (("candidates", candidates), ("rewards", labels), ("audit", audit)):
        write_json(args.output / f"{name}.json", value)
    for name, value in (("windows", windows), ("records", records)):
        (args.output / f"{name}.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in value))
    outputs = {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()}
    write_json(args.output / "manifest.json", {"schema": "pld_curator_v2", "episodes_in_order": list(map(str, folders)),
               "sources_sha256": hashes, "outputs_sha256": outputs,
               "review_sha256": sha256(args.rewards) if args.rewards else None,
               "ready_for_replay_compilation": bool(records), "robot_success_rate": None})
    print(json.dumps({"candidates": len(candidates), "actual_intervals": len(windows), "reviewed_windows": len(labels)}))


if __name__ == "__main__":
    main()
