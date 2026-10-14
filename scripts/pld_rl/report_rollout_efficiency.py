"""Reproducible tabletop attempt count, independent of success/empty prediction.

Count relative-v3 physical closures whose closure/reference FK is in the fixed
scene's grasp workspace. Empty grasps count too. Report excluded closures and
forecast-only windows separately; this is an operational count, not human truth.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"packages/pld-rl/src"))
from yam_pld_rl.data import sha256
from yam_pld_rl.kinematics import YamKinematics
from yam_pld_rl.rollout_metrics import episode_metrics
from yam_pld_rl.trajectory import in_workspace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--count-reviews", type=Path, help="candidate-SHA-bound count corrections; never reward labels")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import h5py  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    candidate_path = args.dataset/"curator-v1/candidates.json"
    cs = json.loads(candidate_path.read_text())
    reviews = {r["candidate_id"]: r for r in json.loads(args.count_reviews.read_text())} if args.count_reviews else {}
    contract = json.loads((args.bundle/"manifest.json").read_text())["contract"]
    fk = YamKinematics(args.bundle/"fk.xml")
    sources = json.loads((args.dataset/"source-snapshot-v2.json").read_text())
    folders = {e["manifest"]["episode_id"]: args.dataset/"raw-v2"/Path(sources["source_root"]).name/e["path"]
               for e in sources["episodes"]}
    rows, decisions, inputs = [], [], {str(candidate_path): sha256(candidate_path),
                                      str(args.bundle/"manifest.json"): sha256(args.bundle/"manifest.json")}
    if args.count_reviews:
        inputs[str(args.count_reviews)] = sha256(args.count_reviews)
    for audit_path in sorted((args.dataset/"batch-audit-v1").glob("E*.json")):
        audit = json.loads(audit_path.read_text())
        eid, count = audit["episode_id"], Counter()
        states = []
        for path in sorted(folders[eid].glob("segment_*/samples.h5")):
            inputs[str(path)] = sha256(path)
            with h5py.File(path) as f:
                n = int(f["committed_rows"][()])
                states.append(f["measured_state"][:n])
        states = np.concatenate(states)
        for c in (c for c in cs if c["episode_id"] == eid):
            count["raw"] += 1
            if c.get("kind") == "no_closure":
                count["forecast_only"] += 1
                decisions.append({"candidate_id": c["candidate_id"], "counted": False,
                                  "reason": "forecast-only window, no physical closure"})
                continue
            off = 0 if c["arm"] == "left" else 7
            points = [fk.position(states[i, off:off+6]) for i in (c["close_row"], c["reference_row"])]
            in_area = any(in_workspace(x, c["arm"], contract) for x in points)
            count["outside_pose"] += int(not in_area)
            counted = in_area
            reason = "physical closure in grasp workspace" if counted else "outside calibrated grasp workspace"
            review = reviews.get(c["candidate_id"])
            if review:
                if review["candidate_sha256"] != c["candidate_sha256"] or type(review["count_as_attempt"]) is not bool:
                    raise ValueError("count review belongs to different candidate/evidence")
                counted, reason = review["count_as_attempt"], review["reason"]
                count["count_reviews"] += 1
            count["confirmed" if counted else "outside"] += 1
            if counted:
                count[c["arm"]] += 1
            decisions.append({"candidate_id": c["candidate_id"], "counted": bool(counted),
                              "arm": c["arm"], "closure_and_reference_xyz_m": [x.tolist() for x in points],
                              "reason": reason, "count_review": review})
        evaluations = [e["details"]["evaluation"] for e in audit["raw_evaluation_events"]]
        label = evaluations[-1] if evaluations else {"C": None, "N": 10}
        row = {"episode_number": audit["number"], "episode_id": eid,
               "behavior_mode": "collect", "count_source": "relative_v3_grasp_workspace_with_count_review_v2",
               "raw_candidates": count["raw"], "outside_grasp_workspace": count["outside_pose"],
               "excluded_non_target_closures": count["outside"], "count_reviews": count["count_reviews"],
               "forecast_only_windows": count["forecast_only"], "left_attempts": count["left"],
               "right_attempts": count["right"],
               **episode_metrics(initial_objects=label["N"], completed_objects=label["C"],
                                 attempts=count["confirmed"], duration_s=audit["duration_s"])}
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/"episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    result = {"schema": "pld_rollout_efficiency_v2", "episodes": rows,
              "total_completed": sum(r["completed_objects"] for r in rows if r["completed_objects"] is not None),
              "total_initial": sum(r["initial_objects"] for r in rows),
              "total_counted_attempts": sum(r["physical_attempts"] for r in rows),
              "full_completion_episodes": sum(r["full_completion"] is True for r in rows),
              "actor_eval_tested": False, "automatic_count_accuracy": None,
              "count_semantics": "Fixed relative closure/reopen hysteresis and calibrated FK workspace; includes empty grasps; not inferred from holding; chunks/windows not counted as attempts",
              "score_semantics": "completed:min(1,N/A); incomplete:0; C/A is task efficiency, not local grasp success rate",
              "count_decisions": decisions, "inputs_sha256": inputs}
    (args.output/"summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("episodes", "inputs_sha256", "count_decisions")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
