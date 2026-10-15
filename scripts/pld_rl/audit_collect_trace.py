"""CPU-only inspection of real collect traces, no training or actuator I/O."""

import argparse
from collections import Counter
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src"))
from yam_pld_rl.block_contract import decode
from yam_pld_rl.block_contract import feasible_bounds
from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.block_contract import map_action
from yam_pld_rl.collector import validate_collector_trace
from yam_pld_rl.curator import native_replies
from yam_pld_rl.curator import reconstruct_intervals
from yam_pld_rl.data import sha256
from yam_pld_rl.native_recording import read_episode
from yam_pld_rl.trajectory import in_workspace
from yam_pld_rl.trajectory import model_fk
from yam_pld_rl.trajectory import planning_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.episode.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    m = json.loads(args.bundle_manifest.read_text())
    snapshot = sha256(args.bundle_manifest)
    if not m["contract"].get("trajectory"):
        parser.error("continuous trajectory contract required")
    c = m["contract"]
    rows, events, hashes = read_episode(folder)
    reply_map, cancelled = native_replies(rows, events)
    accepted = [
        v["reply"]
        for key, v in reply_map.items()
        if key not in cancelled and not v["reply"].get("discarded") and not v["reply"].get("error")
    ]
    checked = 0
    errors = []
    reason = Counter()
    support = Counter()
    max_decode_error = 0.0
    for r in accepted:
        p = r.get("pld") or {}
        if p.get("snapshot_sha256") != snapshot:
            raise ValueError("foreign snapshot")
        if p.get("contract_sha256") != fingerprint(c):
            raise ValueError("foreign contract")
        if p.get("mode") != "collect":
            raise ValueError("not collect mode")
        if not p["gate"]["active"]:
            continue
        try:
            validate_collector_trace(p, m)
            arm = p["arm"]
            off = 0 if arm == "left" else 7
            d = p["delay_steps"]
            b = np.asarray(p["physical_base"])
            q = np.asarray(p["physical_actions"])
            ctx = planning_context(b, p["previous_command"], p["state"], p["gate"], arm, d, c)
            pos, neg = feasible_bounds(b, p["previous_command"], arm, d, c, context=ctx)
            if not p["gate"]["apply_residual"]:
                pos, neg = np.zeros(7), np.zeros(7)
            np.testing.assert_allclose(pos, p["scale"], atol=1e-6, rtol=0)
            np.testing.assert_allclose(neg, p["negative_scale"], atol=1e-6, rtol=0)
            np.testing.assert_allclose(p["action"], map_action(p["sample_u"], pos, neg), atol=1e-6, rtol=0)
            predicted = decode(b, p["action"], arm, d, c, context=ctx)
            max_decode_error = max(max_decode_error, float(abs(predicted - q).max()))
            np.testing.assert_allclose(predicted, q, atol=1e-6, rtol=0)
            np.testing.assert_array_equal(q[:d], b[:d])
            np.testing.assert_array_equal(q[:, 7 - off : 14 - off], b[:, 7 - off : 14 - off])
            np.testing.assert_array_equal(np.asarray(p["actions"])[:d], np.asarray(p["raw_base"])[:d])
            np.testing.assert_array_equal(q[ctx["end_index"] :], b[ctx["end_index"] :])
            delta = q - b
            assert np.all(delta <= np.asarray(c.get("positive_radius", c["radius"])) + 1e-7)
            assert np.all(-delta <= np.asarray(c.get("negative_radius", c["radius"])) + 1e-7)
            editable = (abs(np.asarray(ctx["bias"])).sum(1) + abs(np.asarray(ctx["weight"]))) > 1e-12
            active = slice(off, off + 7)
            step_mask = editable | np.r_[False, editable[:-1]]
            assert np.all(
                abs(np.diff(np.vstack((p["previous_command"], q)), axis=0)[step_mask, active])
                <= np.asarray(c["max_target_step"])[active] + 2e-7
            )
            acc_mask = editable[2:] | editable[1:-1] | editable[:-2]
            assert np.all(
                abs(np.diff(q, n=2, axis=0)[acc_mask, active])
                <= np.asarray(c["trajectory"]["max_target_second_difference"])[active] + 2e-7
            )
            interior = np.arange(49) >= d
            assert np.all(
                abs(np.diff(delta, axis=0)[interior, active]) <= np.asarray(c["max_residual_step"])[active] + 2e-7
            )
            interior_acc = np.arange(48) > d
            assert np.all(
                abs(np.diff(delta, n=2, axis=0)[interior_acc, active])
                <= np.asarray(c["trajectory"]["max_residual_second_difference"])[active] + 2e-7
            )
            reason[str(ctx["reason"])] += 1
            support["parent_zero_support"] += int(
                np.sum(np.asarray(p["actor_scale"]) + np.asarray(p["actor_negative_scale"])) == 0
            )
            support["expanded_nonzero_support"] += int(np.sum(pos + neg) > 0)
            checked += 1
        except (ValueError, AssertionError, KeyError) as e:
            errors.append({"decision_id": p["decision_id"], "error": str(e)})
    intervals, rejected = reconstruct_intervals(
        rows, events, action_limits=json.loads((folder / "manifest.json").read_text()).get("action_limits")
    )
    commands = []
    windows = defaultdict(list)
    geometry_violations = []
    for i, row in enumerate(rows):
        sel = row["details"].get("policy_selection") or {}
        token = sel.get("request") or {}
        if not token:
            continue
        v = reply_map.get((token["epoch"], str(token["request_id"])))
        if not v:
            continue
        p = v["reply"].get("pld") or {}
        if not p.get("arm"):
            continue
        arm = p["arm"]
        off = 0 if arm == "left" else 7
        k = sel["model_index"]
        b = np.asarray(p["physical_base"][k])
        q = np.asarray(p["physical_actions"][k])
        delta = q - b
        if not np.any(abs(delta) > 1e-9):
            continue
        xyz = model_fk().position(q[off : off + 6])
        if not in_workspace(xyz, arm, c):
            geometry_violations.append({"row": i, "arm": arm, "xyz": xyz.tolist()})
        commands.append(
            {
                "row": i,
                "time_s": row["time"] - rows[0]["time"],
                "arm": arm,
                "window": p["window_id"],
                "gripper_delta": float(delta[off + 6]),
                "base_g": float(b[off + 6]),
                "target_g": float(q[off + 6]),
                "measured_g": float(row["measured_state"][off + 6]),
                "phase": p["gate"]["phase"],
                "joint_peak_rad": float(abs(delta[off : off + 6]).max()),
                "measured_z": float(model_fk().position(np.asarray(row["measured_state"])[off : off + 6])[2]),
            }
        )
        windows[p["window_id"]].append(commands[-1])

    def stats(xs):
        return np.percentile(xs, [0, 5, 50, 95, 99, 100]).tolist() if xs else None

    by_arm = {}
    for arm in ["all", "left", "right"]:
        xs = [x for x in commands if arm == "all" or x["arm"] == arm]
        ww = [x for x in windows.values() if arm == "all" or x[0]["arm"] == arm]
        by_arm[arm] = {
            "nonzero_ticks": len(xs),
            "nonzero_windows": len(ww),
            "gripper_delta_quantiles_min_p05_p50_p95_p99_max": stats([x["gripper_delta"] for x in xs]),
            "extra_close_ge_0_05_ticks": sum(x["gripper_delta"] <= -0.05 for x in xs),
            "extra_close_ge_0_10_ticks": sum(x["gripper_delta"] <= -0.10 for x in xs),
            "windows_extra_close_ge_0_05": sum(min(x["gripper_delta"] for x in w) <= -0.05 for w in ww),
            "windows_extra_close_ge_0_10": sum(min(x["gripper_delta"] for x in w) <= -0.10 for w in ww),
            "minimum_target_g": min((x["target_g"] for x in xs), default=None),
        }
    out = {
        "status": "passed" if not errors and not geometry_violations else "failed",
        "episode": str(folder),
        "snapshot_sha256": snapshot,
        "contract_sha256": fingerprint(c),
        "sources_sha256": hashes,
        "producer_sha256": sha256(__file__),
        "accepted_replies": len(accepted),
        "active_traces_checked": checked,
        "trace_errors": errors,
        "max_decode_error": max_decode_error,
        "context_reasons": dict(reason),
        "support": dict(support),
        "by_arm": by_arm,
        "actual_intervals": len(intervals),
        "interval_statuses": dict(Counter(r["receipt"]["status"] for r in intervals)),
        "execution_exclusions": rejected,
        "geometry_violations": geometry_violations,
        "quantile_order": [0, 0.05, 0.5, 0.95, 0.99, 1],
        "training": False,
        "reward_published": False,
        "geometry_is_not_semantic_placement_ground_truth": True,
    }
    (output_dir / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    (output_dir / "commands.json").write_text(json.dumps(commands) + "\n")
    (output_dir / "intervals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in intervals))
    print(json.dumps(out, indent=2))

    if errors or geometry_violations:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
