"""Join operator completion labels and frozen visual attempt counts, CPU only."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"packages/pld-rl/src"))
from yam_pld_rl.data import sha256
from yam_pld_rl.rollout_metrics import episode_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--intent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cs = json.loads((args.dataset/"curator-v1/candidates.json").read_text())
    rows, inputs = [], {}
    for audit_path in sorted((args.dataset/"batch-audit-v1").glob("E*.json")):
        audit = json.loads(audit_path.read_text())
        eid = audit["episode_id"]
        count = Counter()
        for c in (c for c in cs if c["episode_id"] == eid):
            path = args.intent/(c["candidate_id"]+".json")
            if not path.exists():
                count["uncertain"] += 1
                continue
            decision = json.loads(path.read_text())
            if decision.get("candidate_sha256") != c["candidate_sha256"]:
                raise ValueError("visual intent evidence belongs to another candidate")
            inputs[str(path)] = sha256(path)
            intent = (decision.get("decision") or {}).get("grasp_intent", "uncertain")
            count["confirmed" if intent is True else "non_grasp" if intent is False else "uncertain"] += 1
        evaluations = [e["details"]["evaluation"] for e in audit["raw_evaluation_events"]]
        label = evaluations[-1] if evaluations else {"C": None, "N": 10}
        # Unknown visual intent yields a count interval, never a fabricated exact total.
        attempts = count["confirmed"] if not count["uncertain"] else None
        row = {"episode_number": audit["number"], "episode_id": eid,
               "behavior_mode": "collect", "count_source": "frozen_relative_v3_plus_visual_intent",
               "raw_candidates": sum(count.values()), "non_grasp_candidates": count["non_grasp"],
               "unresolved_intent": count["uncertain"], "attempts_lower_bound": count["confirmed"],
               "attempts_upper_bound": count["confirmed"]+count["uncertain"],
               **episode_metrics(initial_objects=label["N"], completed_objects=label["C"],
                                 attempts=attempts, duration_s=audit["duration_s"])}
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/"episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    result = {"schema": "pld_rollout_efficiency_v1", "episodes": rows,
              "total_completed": sum(r["completed_objects"] for r in rows if r["completed_objects"] is not None),
              "total_initial": sum(r["initial_objects"] for r in rows),
              "full_completion_episodes": sum(r["full_completion"] is True for r in rows),
              "actor_eval_tested": False, "automatic_count_accuracy": None,
              "count_semantics": "relative closure/reopen hysteresis plus visible grasp intent; forecast-only intent attempts remain identifiable in source candidates; chunks/windows are not attempts",
              "score_semantics": "completed episodes: min(1,N/A); incomplete:0; unknown count remains unknown; C/A is end-to-end throughput, not local grasp success rate",
              "inputs_sha256": inputs}
    (args.output/"summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("episodes", "inputs_sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
