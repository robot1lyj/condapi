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
from yam_pld_rl.grasp_outcomes import attach_outcomes
from yam_pld_rl.kinematics import YamKinematics
from yam_pld_rl.native_recording import read_episode
from yam_pld_rl.probe_audit import audit_probe


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
    prepared = []
    for c in candidates:
        if c["evidence_eligible"] and c.get("montage"):
            # Keep full raw evidence for review, but a local-reward judge must
            # not inspect a later Pi recovery as if the specialist achieved it.
            frames = [f for f in c["frames"] if f["tick"] <= c.get("scoring_end_tick", float("inf"))]
            if any(f["lift_from_closure_m"] >= .05 for f in frames):
                prepared.append({**c, "frames": frames})
    write_json(output / "candidates-prepared.json", prepared)
    outcomes = []
    for c in candidates:
        proof = c.get("outcome")
        if not proof:
            continue
        suggestion = None
        if proof["kinematic_complete"] and proof["fresh_rows"]:
            if not proof["close_confirmed"] and proof["reason"] == "forecast_timeout":
                suggestion = "no_closure"
            elif proof["close_confirmed"] and proof["max_lift_m"] < .05 and proof["reason"] in {"reopened", "closed_timeout", "timeout"}:
                suggestion = "not_lifted"
        outcomes.append({**c, "proposed_status": suggestion, "approved": False,
                         "review_required": "confirm grasp intent for kinematic failures; inspect held/empty/dropped visually"})
    write_json(output / "outcomes-prepared.json", outcomes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, nargs="+", required=True)
    p.add_argument("--validation-episodes", nargs="*", default=[], help="resolved episode paths; split at episode level")
    p.add_argument("--rewards", type=Path, help="reviewed candidate_id/status/approved/reviewer/evidence_rows JSON list")
    p.add_argument("--export-evidence", action="store_true", help="decode selected images for the existing Qwen/DINO tools")
    p.add_argument("--probe-audit", action="store_true", help="audit diagnostic probe; never publish training rewards")
    p.add_argument("--segmentation", choices=("legacy_v2", "relative_v3"), default="relative_v3")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.probe_audit and args.rewards:
        p.error("diagnostic probe cannot be combined with --rewards")
    folders = [p.resolve() for p in args.episodes]
    if len(folders) != len(set(folders)):
        p.error("duplicate raw episode")
    validation = {str(Path(p).resolve()) for p in args.validation_episodes}
    if not validation.issubset({str(p) for p in folders}):
        p.error("validation episode is not an input episode")
    args.output.mkdir(parents=True, exist_ok=False)
    fk = YamKinematics(ROOT / "third_party/yam-fk-model/yam_linear4310.xml")
    approved = json.loads(args.rewards.read_text()) if args.rewards else []
    candidates, windows, records, labels, audit, hashes, probe_reports, policy_events = [], [], [], [], [], {}, [], []
    episode_ids = set()
    for number, folder in enumerate(folders, 1):
        rows, events, source_hashes = read_episode(folder)
        native_manifest = json.loads((folder / "manifest.json").read_text())
        if rows[0]["episode_id"] in episode_ids:
            raise ValueError("duplicate native episode UUID across inputs")
        episode_ids.add(rows[0]["episode_id"])
        policy_events.extend(events)
        if args.probe_audit:
            probe_reports.append(audit_probe(rows, events, native_manifest))
        hashes.update(source_hashes)
        c = physical_candidates(rows, fk, number, segmentation=args.segmentation)
        w, rejected = reconstruct_intervals(rows, events, diagnostics=args.probe_audit,
                                            action_limits=native_manifest.get("action_limits"))
        attach_outcomes(c, w, rows, fk, number)
        groups = defaultdict(list)
        for interval in w:
            groups[interval["proposal"]["window_id"]].append(interval)
        for candidate in c:
            matched = []
            for window in groups.values():
                first, last = window[0]["proposal"], window[-1]["receipt"]
                if (last.get("window_end") and first["arm"] == candidate["arm"] and first["epoch"] == candidate["epoch"]
                        and first["start_tick"] <= candidate["close_tick"] < last["end_tick"]):
                    matched.append(last["end_tick"])
            if len(matched) == 1:
                candidate["scoring_end_tick"] = matched[0]
            candidate["source_release_sha256"] = fingerprint(source_hashes)
            candidate["candidate_sha256"] = fingerprint(candidate)
        split = "val" if str(folder) in validation else "train"
        relevant = [r for r in approved if r["candidate_id"] in {q["candidate_id"] for q in c}]
        linked, rewards, association = ([], [], []) if args.probe_audit else associate_rewards(w, c, relevant, rows, split)
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
    for name, value in (("windows", windows), ("records", records), ("policy-events", policy_events)):
        (args.output / f"{name}.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in value))
    if args.probe_audit:
        write_json(args.output / "probe-audit.json", {"status": "passed" if all(r["status"] == "passed" for r in probe_reports) else "failed",
                   "episodes": probe_reports, "training_ready": False, "robot_success_rate": None})
    outputs = {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()}
    write_json(args.output / "manifest.json", {"schema": "pld_curator_v2", "producer": "Replay-Curator",
               "producer_files_sha256": {name: sha256(ROOT / name) for name in (
                   "scripts/pld_rl/curate_recording.py", "packages/pld-rl/src/yam_pld_rl/native_recording.py",
                   "packages/pld-rl/src/yam_pld_rl/probe_audit.py", "packages/pld-rl/src/yam_pld_rl/curator.py",
                   "packages/pld-rl/src/yam_pld_rl/grasp_outcomes.py",
                   "packages/pld-rl/src/yam_pld_rl/grasp_events.py", "packages/pld-rl/src/yam_pld_rl/kinematics.py")},
               "episodes_in_order": list(map(str, folders)),
               "sources_sha256": hashes, "outputs_sha256": outputs,
               "review_sha256": sha256(args.rewards) if args.rewards else None,
               "ready_for_replay_compilation": bool(records) and not args.probe_audit,
               "purpose": "probe_diagnostics" if args.probe_audit else "reward_association", "robot_success_rate": None})
    print(json.dumps({"candidates": len(candidates), "actual_intervals": len(windows), "reviewed_windows": len(labels)}))
    if args.probe_audit and any(r["status"] != "passed" for r in probe_reports):
        raise SystemExit(2)  # Keep evidence, but never report a failed audit as a successful job.


if __name__ == "__main__":
    main()
