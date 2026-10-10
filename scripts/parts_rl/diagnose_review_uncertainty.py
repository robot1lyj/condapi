"""Audit existing review decisions using CPU arrays; never publish rewards."""

import argparse
from collections import Counter
from collections import defaultdict
from itertools import pairwise
import json
from pathlib import Path

from classify_full_grasp_review import aggregate
from classify_full_grasp_review import evidence_span
import h5py
import numpy as np
from prepare_full_grasp_review import sha
from prepare_full_grasp_review import write


def counts(items):
    return dict(Counter(items))


def maximum_lift(z, mask, reference):
    return float(np.max(z[mask] - reference)) if np.any(mask) else None


def boundary_audit(candidates, raw_root, episodes_path):
    """Read only recorded H5 flags, without inferring why acquisition omitted them."""
    records, sources = [], []
    for episode in json.loads(episodes_path.read_text()):
        selected = [c for c in candidates if c["episode_number"] == episode["number"]
                    and c["end_reason"] == "trace_gap"]
        folder = raw_root / episode["source_path"]
        offset = 0
        for segment in json.loads((folder / "manifest.json").read_text())["segments"]:
            path = folder / segment["path"] / "samples.h5"
            sources.append({"path": str(path), "sha256": sha(path)})
            with h5py.File(path, "r") as handle:
                n = int(handle["committed_rows"][()])
                valid, times = handle["observation_valid"][:n], handle["time"][:n]
                for candidate in selected:
                    row = candidate["end_row"] + 1 - offset
                    if not 0 < row < n:
                        continue
                    detail = json.loads(handle["details"][row])
                    pose = detail.get("parts", {}).get("arms", {}).get(candidate["arm"], {})
                    next_valid = row
                    while next_valid < n and not valid[next_valid]:
                        next_valid += 1
                    records.append({"candidate_id": candidate["candidate_id"],
                                    "classification_reason": candidate["classification_reason"],
                                    "source_h5": str(path), "source_row": row,
                                    "measured_valid": bool(handle["measured_state__valid"][row]),
                                    "observation_valid": bool(valid[row]), "pose_valid": pose.get("pose_valid"),
                                    "epoch_same": bool(handle["epoch"][row] == handle["epoch"][row - 1]),
                                    "tick_contiguous": bool(handle["tick"][row] == handle["tick"][row - 1] + 1),
                                    "dt_s": float(times[row] - times[row - 1]),
                                    "camera_received_at": handle["camera_host_received_at"][row].tolist(),
                                    "video_indices": handle["video_indices"][row].tolist(),
                                    "invalid_run_rows": next_valid - row,
                                    "next_valid_after_s": float(times[next_valid] - times[row - 1]) if next_valid < n else None,
                                    "phase": detail.get("phase"), "source": detail.get("source"),
                                    "policy_valid": detail.get("policy_valid")})
            offset += n
    if len(records) != sum(c["end_reason"] == "trace_gap" for c in candidates):
        raise ValueError("boundary audit did not cover every interrupted candidate")
    return {"records": records, "sources": sources, "episodes_sha256": sha(episodes_path),
            "sources_read_only": True}


def stable_runs(frames, label):
    runs, current = [], []
    for frame in frames:
        previous = current[-1] if current else None
        connected = previous is not None and frame["epoch"] == previous["epoch"]
        connected = connected and 0 < frame["visual_time"] - previous["visual_time"] <= .25
        connected = connected and frame["camera_sequence"] > previous["camera_sequence"]
        connected = connected and frame["sha256"] != previous["sha256"]
        if frame["label"] != label or not connected:
            if len(current) >= 2 and current[-1]["visual_time"] - current[0]["visual_time"] >= .12:
                runs.append(current)
            current = []
        if frame["label"] == label:
            current.append(frame)
    if len(current) >= 2 and current[-1]["visual_time"] - current[0]["visual_time"] >= .12:
        runs.append(current)
    return [{"first_row": r[0]["row"], "last_row": r[-1]["row"],
             "start": r[0]["visual_time"], "end": r[-1]["visual_time"], "frames": len(r)} for r in runs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--episodes", type=Path)
    args = parser.parse_args()
    if (args.raw_root is None) != (args.episodes is None):
        parser.error("--raw-root and --episodes must be supplied together")
    candidates = json.loads((args.dataset / "candidates.json").read_text())
    frames = json.loads((args.dataset / "frame-classifications.json").read_text())
    by_id = {f["id"]: f for f in frames}
    traces = {ep: json.loads((args.traces / f"E{ep:02d}.json").read_text()) for ep in range(1, 15)}
    records = []
    for candidate in candidates:
        selected = [by_id[key] for key in candidate["eligible_frame_ids"]]
        reproduced = aggregate(selected)
        if (reproduced[0], reproduced[3]) != (candidate["suggested_label"], candidate["classification_reason"]):
            raise ValueError("existing candidate decision not reproduced")
        postclose = [s for s in candidate["samples"] if candidate["close_row"] <= s["row"] <= candidate["end_row"]]
        valid = [s for s in postclose if s["valid"]]
        closed = [s for s in valid if s["openness"] < .6]
        lifted = [s for s in closed if s["lift_delta_m"] is not None and s["lift_delta_m"] >= .05]
        fresh = [s for s in lifted if s["visual_time"] is not None and 0 <= s["time"] - s["visual_time"] <= .25]
        if [s["frame_key"] for s in fresh] != candidate["eligible_frame_ids"]:
            raise ValueError("eligibility not reproduced")
        exclusion = None
        for stage, subset in [("no_postclose_sample", postclose), ("no_valid_pose", valid),
                              ("never_closed_below_0.6", closed), ("closed_but_no_entry_relative_5cm", lifted),
                              ("no_fresh_camera_observation", fresh)]:
            if not subset:
                exclusion = stage
                break
        trace = traces[candidate["episode_number"]]
        arm = trace["arms"][candidate["arm"]]
        rows = np.arange(candidate["close_row"], candidate["end_row"] + 1)
        z = np.array(arm["base_z_m"], dtype=float)[rows]
        g = np.array(arm["openness"], dtype=float)[rows]
        v = np.array(arm["valid"], dtype=bool)[rows] & np.isfinite(z) & np.isfinite(g)
        entry_z = candidate["reference_base_z_m"]
        close_z = arm["base_z_m"][candidate["close_row"]]
        gaps = [b["visual_time"] - a["visual_time"] for a, b in pairwise(selected)]
        labels = {f["label"] for f in selected}
        h_runs = stable_runs(selected, "held_at_snapshot")
        e_runs = stable_runs(selected, "empty_at_snapshot")
        if candidate["classification_reason"] == "insufficient_or_conflicting_evidence":
            if labels == {"uncertain"}:
                detail = "all_frames_visual_uncertain"
            elif evidence_span(selected, "empty_at_snapshot") >= .12:
                detail = "stable_empty_blocked_by_uncertain_frame"
            elif len(selected) == 1:
                detail = "single_eligible_frame"
            else:
                detail = "no_contiguous_same_label_pair"
        else:
            detail = exclusion if not selected else candidate["classification_reason"]
        # Diagnostic only: an attempt may survive a brief missing image, while
        # that image and RL transitions across it must still remain excluded.
        extension_end = candidate["end_row"]
        if candidate["end_reason"] == "trace_gap":
            minimum = min(arm["openness"][candidate["close_row"]:extension_end + 1])
            for row in range(extension_end + 1, len(trace["time"])):
                connected = trace["epoch"][row] == trace["epoch"][row - 1]
                connected = connected and trace["tick"][row] == trace["tick"][row - 1] + 1
                connected = connected and 0 < trace["time"][row] - trace["time"][row - 1] <= .15
                position, opening = arm["base_z_m"][row], arm["openness"][row]
                if not connected or position is None or not np.isfinite(position) or not np.isfinite(opening):
                    break
                extension_end = row
                minimum = min(minimum, opening)
                elapsed = trace["time"][row] - candidate["close_time"]
                if (opening >= .7 and opening - minimum >= .18 and elapsed >= .2) or elapsed >= 8:
                    break
        extension_rows = range(candidate["end_row"] + 1, extension_end + 1)
        extension_lifted = [row for row in extension_rows if arm["valid"][row]
                            and arm["openness"][row] < .6 and arm["base_z_m"][row] - entry_z >= .05]
        records.append({"candidate_id": candidate["candidate_id"], "episode_number": candidate["episode_number"],
                        "arm": candidate["arm"], "label": candidate["suggested_label"],
                        "reason": candidate["classification_reason"], "detail": detail,
                        "end_reason": candidate["end_reason"], "postclose_samples": len(postclose),
                        "valid_samples": len(valid), "closed_samples": len(closed), "lifted_samples": len(lifted),
                        "eligible_frames": len(selected), "entry_minus_close_z_m": entry_z - close_z,
                        "sampled_max_closed_lift_m": max((s["lift_delta_m"] for s in closed), default=None),
                        "dense_max_closed_lift_m": maximum_lift(z, v & (g < .6), entry_z),
                        "dense_max_any_opening_lift_m": maximum_lift(z, v, entry_z),
                        "dense_entry_lift_5cm_closed_rows": int(np.sum(v & (g < .6) & (z - entry_z >= .05))),
                        "dense_close_lift_5cm_closed_rows": int(np.sum(v & (g < .6) & (z - close_z >= .05))),
                        "dense_entry_lift_5cm_g_below_0.7_rows": int(np.sum(v & (g < .7) & (z - entry_z >= .05))),
                        "frame_labels": [f["label"] for f in selected], "max_selected_gap_s": max(gaps, default=None),
                        "diagnostic_extended_end_row": extension_end,
                        "diagnostic_later_lifted_rows": len(extension_lifted),
                        "held_runs": h_runs, "empty_runs": e_runs,
                        "first_held_before_first_empty": bool(h_runs and e_runs and h_runs[0]["end"] < e_runs[0]["start"]),
                        "frames": [{"id": f["id"], "row": f["row"], "dt_to_candidate_end_s": candidate["end_time"] - f["time"],
                                    "label": f["label"], "roi_label": f["roi"]["label"] if f["roi"] else None,
                                    "cls_label": f["cls"]["label"] if f["cls"] else None,
                                    "roi_margin": f["roi"]["margin"] if f["roi"] else None,
                                    "cls_margin": f["cls"]["margin"] if f["cls"] else None,
                                    "openness": f["openness"], "reason": f["reason"]} for f in selected]})
    unknown = [r for r in records if r["label"] == "uncertain"]
    missing = [r for r in unknown if r["reason"] == "no_eligible_snapshot"]
    mixed = [r for r in unknown if r["reason"] == "mixed_held_and_empty"]
    insufficient = [r for r in unknown if r["reason"] == "insufficient_or_conflicting_evidence"]
    sensitivities = {}
    for name in ["roi_only", "stable_empty_allows_unknown", "held_stable_ignores_mixed"]:
        results = []
        changed = []
        for candidate in candidates:
            selected = [by_id[key] for key in candidate["eligible_frame_ids"]]
            label = candidate["suggested_label"]
            if name == "roi_only":
                diagnostic = [{**f, "label": f["roi"]["label"] if f["roi"] else "uncertain"} for f in selected]
                label = aggregate(diagnostic)[0]
            elif name == "stable_empty_allows_unknown":
                if "held_at_snapshot" not in {f["label"] for f in selected} and evidence_span(selected, "empty_at_snapshot") >= .12:
                    label = "empty_observed"
            elif name == "held_stable_ignores_mixed":
                if evidence_span(selected, "held_at_snapshot") >= .12:
                    label = "held_observed"
            results.append(label)
            if label != candidate["suggested_label"]:
                changed.append({"candidate_id": candidate["candidate_id"], "before": candidate["suggested_label"], "after": label})
        sensitivities[name] = {"counts": counts(results), "changed": changed,
                               "accuracy_measured": False, "applied": False}
    reference_jobs = json.loads((args.dataset / "reference/jobs.json").read_text())
    reference_labels = {r["candidate_id"]: r["label"] for r in json.loads((args.dataset / "reference/labels.json").read_text())["records"]}
    reference_by_episode = defaultdict(Counter)
    for job in reference_jobs:
        reference_by_episode[job["episode_number"]][reference_labels[job["id"]]] += 1
    visual_unknown = [f for f in frames if f["reason"] == "feature_ambiguous_or_disagree"]
    summary = {"candidates": len(records), "uncertain": len(unknown),
               "reason_counts": counts(r["reason"] for r in unknown),
               "no_eligible_details": counts(r["detail"] for r in missing),
               "no_eligible_dense_eligible_but_missed_sampling": [r["candidate_id"] for r in missing if r["dense_entry_lift_5cm_closed_rows"] > 0],
               "no_eligible_if_close_reference_diagnostic_only": [r["candidate_id"] for r in missing if r["dense_close_lift_5cm_closed_rows"] > 0],
               "no_eligible_if_g_below_0.7_diagnostic_only": [r["candidate_id"] for r in missing if r["dense_entry_lift_5cm_g_below_0.7_rows"] > 0],
               "no_eligible_with_5cm_lift_any_opening": [r["candidate_id"] for r in missing if r["dense_max_any_opening_lift_m"] is not None and r["dense_max_any_opening_lift_m"] >= .05],
               "no_eligible_trace_gap": sum(r["end_reason"] == "trace_gap" for r in missing),
               "no_eligible_later_lift_if_attempt_survives_missing_image": [r["candidate_id"] for r in missing if r["diagnostic_later_lifted_rows"] > 0],
               "all_candidate_end_reasons": counts(r["end_reason"] for r in records),
               "mixed_stable_held": sum(bool(r["held_runs"]) for r in mixed),
               "mixed_stable_empty": sum(bool(r["empty_runs"]) for r in mixed),
               "mixed_both_stable": sum(bool(r["held_runs"] and r["empty_runs"]) for r in mixed),
               "mixed_held_stable_then_empty_stable": sum(r["first_held_before_first_empty"] for r in mixed),
               "mixed_no_stable_pair": sum(not r["held_runs"] and not r["empty_runs"] for r in mixed),
               "insufficient_details": counts(r["detail"] for r in insufficient),
               "visual_unknown_branch_pairs": counts(f["roi"]["label"] + "/" + f["cls"]["label"] for f in visual_unknown),
               "frame_reason_counts": counts(f["reason"] for f in frames),
               "reference_by_episode": {str(k): dict(v) for k, v in reference_by_episode.items()},
               "per_episode": {str(ep): counts(r["reason"] for r in records if r["episode_number"] == ep) for ep in range(1, 15)},
               "sensitivity_not_accuracy": sensitivities,
               "automatic_tracking": False, "training_executed": False, "rewards_published": False,
               "dataset_modified": False}
    boundary = boundary_audit(candidates, args.raw_root, args.episodes) if args.raw_root else None
    if boundary:
        bs = boundary["records"]
        bm = [r for r in bs if r["classification_reason"] == "no_eligible_snapshot"]
        summary["raw_boundary_audit"] = {
            "boundaries": len(bs), "measured_and_pose_valid": sum(r["measured_valid"] and r["pose_valid"] for r in bs),
            "observation_invalid": sum(not r["observation_valid"] for r in bs),
            "tick_and_epoch_continuous": sum(r["tick_contiguous"] and r["epoch_same"] for r in bs),
            "all_camera_received_zero": sum(all(t == 0 for t in r["camera_received_at"]) for r in bs),
            "no_eligible_boundaries": len(bm), "no_eligible_missing_row_counts": counts(r["invalid_run_rows"] for r in bm),
            "no_eligible_max_recovery_s": max(r["next_valid_after_s"] for r in bm),
        }
    args.output.mkdir(exist_ok=False)
    if boundary:
        write(args.output / "boundaries.json", boundary)
    write(args.output / "diagnostics.json", {"summary": summary, "records": records,
          "sources": {"dataset": str(args.dataset), "traces": str(args.traces),
                      "candidates_sha256": sha(args.dataset / "candidates.json"),
                      "frames_sha256": sha(args.dataset / "frame-classifications.json"),
                      "script_sha256": sha(Path(__file__))}})
    write(args.output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ["per_episode", "sensitivity_not_accuracy"]}, indent=2))
    print(json.dumps({k: v["counts"] for k, v in sensitivities.items()}, indent=2))


if __name__ == "__main__":
    main()
