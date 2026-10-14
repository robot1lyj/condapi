"""CPU-only audit of opt-in short closing support, never a policy evaluation.

Rebuild the old recorded decoder first. On newly admitted contexts, try a
synthetic maximum feasible closing target to measure control authority only.
No actor/critic is loaded, no optimizer is run, and no robot is contacted.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"packages/pld-rl/src"))
from yam_pld_rl.block_contract import feasible_bounds
from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.block_contract import map_action
from yam_pld_rl.block_contract import validate_contract
from yam_pld_rl.curator import native_replies
from yam_pld_rl.data import sha256
from yam_pld_rl.native_recording import read_episode
from yam_pld_rl.trajectory import planning_context
from yam_pld_rl.trajectory import trajectory_decode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    old = manifest.get("eval_contract", manifest["contract"])
    new = copy.deepcopy(old)
    new["trajectory"]["short_closing_min_steps"] = 5
    validate_contract(new)
    rows, events, sources = read_episode(args.episode)
    replies, cancelled = native_replies(rows, events)
    results, context_error = [], 0.
    for key, reply in replies.items():
        r = reply["reply"]
        t = r.get("pld") or {}
        if (key in cancelled or r.get("discarded") or r.get("error")
                or not t.get("gate", {}).get("active") or "trajectory_context" not in t):
            continue
        if t["snapshot_sha256"] != sha256(args.manifest):
            raise ValueError("recorded snapshot differs from the supplied manifest")
        arm, delay = t["arm"], t["delay_steps"]
        base = np.asarray(t["physical_base"], dtype=np.float64)
        previous, state = np.asarray(t["previous_command"], np.float32), np.asarray(t["state"], np.float32)
        gate = t["gate"]
        old_ctx = planning_context(base, previous, state, gate, arm, delay, old)
        recorded = t["trajectory_context"]
        if old_ctx.keys() != recorded.keys():
            raise ValueError("recorded context fields changed")
        for name, value in old_ctx.items():
            if name in ("bias", "weight") or isinstance(value, float):
                a, b = np.asarray(value), np.asarray(recorded[name])
                context_error = max(context_error, float(np.abs(a-b).max()))
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
            elif value != recorded[name]:
                raise ValueError(f"old planning context changed at {t['decision_id']}: {name}")
        decoded, _ = trajectory_decode(base, t["action"], arm, old, old_ctx)
        np.testing.assert_allclose(decoded, t["physical_actions"], atol=1e-6, rtol=0)
        new_ctx = planning_context(base, previous, state, gate, arm, delay, new)
        result = {"decision_id": t["decision_id"], "window_id": t["window_id"], "arm": arm,
                  "phase": gate["phase"], "old_reason": old_ctx["reason"], "new_reason": new_ctx["reason"],
                  "short_closing_admitted": bool(new_ctx.get("closing_only") and new_ctx["reason"] is None)}
        if result["short_closing_admitted"]:
            positive, negative = feasible_bounds(base, previous, arm, delay, new, context=new_ctx)
            np.testing.assert_array_equal(positive, np.zeros(7))
            np.testing.assert_array_equal(negative[:6], np.zeros(6))
            probe = map_action(np.r_[np.zeros(6), -1.], positive, negative)
            out, projection = trajectory_decode(base, probe, arm, new, new_ctx)
            bridge, _ = trajectory_decode(base, np.zeros(7), arm, new, new_ctx)
            offset = 0 if arm == "left" else 7
            np.testing.assert_array_equal(out[:, offset:offset+6], bridge[:, offset:offset+6])
            np.testing.assert_array_equal(out[:delay], base[:delay])
            np.testing.assert_array_equal(out[new_ctx["end_index"]:], base[new_ctx["end_index"]:])
            other = slice(7, 14) if arm == "left" else slice(0, 7)
            np.testing.assert_array_equal(out[:, other], base[:, other])
            result.update(editable_ticks=new_ctx["end_index"]-delay,
                          maximum_closing_vs_same_bridge=float((out[:, offset+6]-bridge[:, offset+6]).min()),
                          new_actor_joint_target_contribution=0., projection=projection)
        elif old_ctx["reason"] is None:
            if fingerprint(old_ctx) != fingerprint(new_ctx):
                raise ValueError("normal full-horizon path changed")
        results.append(result)
    candidate = [r for r in results if r["short_closing_admitted"]]
    summary = {"schema": "pld_short_closing_authority_audit_v1", "old_decoder_verified": len(results),
               "maximum_recorded_context_float_reconstruction_error": context_error,
               "correcting_requests": sum(r["phase"] == "correcting" for r in results),
               "old_reasons": dict(Counter(r["old_reason"] for r in results if r["phase"] == "correcting")),
               "new_reasons": dict(Counter(r["new_reason"] for r in results if r["phase"] == "correcting")),
               "short_closing_admitted_requests": len(candidate),
               "short_closing_windows": len({r["window_id"] for r in candidate}),
               "closing_authority_requests": sum(r["maximum_closing_vs_same_bridge"] < -1e-7 for r in candidate),
               "maximum_closing_vs_same_bridge": min((r["maximum_closing_vs_same_bridge"] for r in candidate), default=0.),
               "old_contract_sha256": fingerprint(old), "candidate_contract_sha256": fingerprint(new),
               "not_a_policy_rollout": True, "actor_loaded": False, "training": False, "deployed": False,
               "sources_sha256": sources, "manifest_sha256": sha256(args.manifest), "results": results}
    with args.output.open("x") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("sources_sha256", "results")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
