"""Join terminal holding suggestions and scoped visual grasp intent; CPU only."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src"))
from yam_pld_rl.data import sha256
from yam_pld_rl.data import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curator", type=Path, required=True)
    p.add_argument("--vision", type=Path, required=True)
    p.add_argument("--intent", type=Path, required=True)
    p.add_argument("--intent-output", default="vlm-intent")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    candidates = json.loads((args.curator / "candidates.json").read_text())
    held = {c["candidate_id"]: c for c in json.loads((args.vision / "candidates.json").read_text())}
    results = []
    for original in candidates:
        c = {**original, **held.get(original["candidate_id"], {})}
        path = args.intent / args.intent_output / (original["candidate_id"] + ".json")
        if path.exists():
            raw = json.loads(path.read_text())
            if raw["candidate_sha256"] != original["candidate_sha256"]:
                raise ValueError("intent belongs to different candidate")
            d = raw.get("decision") or {}
            if d and not raw.get("parse_error"):
                c.update(intent_evidence_candidate_sha256=original["candidate_sha256"],
                         intent_evidence_sha256=sha256(path),
                         grasp_intent_confirmed=d["grasp_intent"] is True,
                         held_then_lost_confirmed=d["held_then_lost"] is True)
                proof = original["outcome"]
                status = None
                if d["grasp_intent"] is False and d["confidence"] == "high":
                    c["model_non_grasp"] = True
                elif d["grasp_intent"] is False:
                    c["audit_flags"] = [*c.get("audit_flags", []), "weak_negative_intent_not_excluded"]
                elif d["held_then_lost"] is True:
                    status = "dropped"
                elif d["grasp_intent"] is True:
                    if not proof["close_confirmed"] and proof["reason"] == "forecast_timeout":
                        status = "no_closure"
                    elif (proof["max_lift_m"] < .05 and proof["reason"] in
                          {"reopened", "closed_timeout", "timeout"}):
                        status = "not_lifted"
                if status:
                    c.update(suggested_status=status, confidence=d["confidence"],
                             reasons=[d["reason"]])
        # Original immutable identity, frames and proof always win over adapters.
        c.update({k: original[k] for k in ("candidate_sha256", "frames", "outcome") if k in original})
        results.append(c)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "candidates.json", results)
    summary = {"candidates": len(results),
               "holding_suggestions": dict(Counter(c.get("suggested_label") for c in results)),
               "goal_failure_suggestions": dict(Counter(c.get("suggested_status") for c in results)),
               "non_grasp": sum(c.get("model_non_grasp") is True for c in results),
               "curator_sha256": sha256(args.curator / "manifest.json"),
               "vision_sha256": sha256(args.vision / "candidates.json"),
               "producer_sha256": sha256(Path(__file__)), "labels_are_model_suggestions": True,
               "accuracy": None, "tracking": False}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
