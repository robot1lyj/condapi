"""Offline probe protocol audit, never a reward or robot-success evaluator."""

from collections import Counter

import numpy as np

from yam_pld_rl.block_contract import SCHEMA
from yam_pld_rl.curator import native_replies


def same(a, b, shape):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    # JSON [] loses the second dimension of a valid zero-length RTC prefix.
    if shape == (0, 14) and a.size == b.size == 0:
        return True
    return a.shape == b.shape == shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a, b, rtol=0, atol=2e-6)


def audit_probe(rows, events, manifest):
    replies, cancelled = native_replies(rows, events)
    observations = {(r["epoch"], r["details"].get("obs_id")): r for r in rows if r["observation_valid"]}
    by_tick = {(r["epoch"], r["tick"]): r for r in rows}
    counts, errors, notes = Counter(), [], []
    identity = manifest.get("pld") or {}
    if identity.get("mode") != "probe":
        raise ValueError("--probe-audit requires a native probe episode")
    for key, entry in replies.items():
        reply = entry["reply"]
        if reply.get("discarded") is not False or reply.get("error") or key in cancelled:
            counts["rejected_or_cancelled_replies"] += 1
            continue
        try:
            p, token = reply["pld"], reply["token"]
            request, context = reply["pld_request"], reply["request_context"]
            if (p["schema"] != SCHEMA or p["mode"] != "probe" or p["episode_id"] != manifest["episode_id"]
                    or p.get("replay_eligible") is not False or p.get("residual_applied") is not False):
                raise ValueError("probe identity/diagnostic flags differ")
            for field in ("run_id", "mode", "contract_sha256", "snapshot_sha256"):
                if p[field] != identity[field]:
                    raise ValueError(f"manifest differs from trace: {field}")
            for field in ("schema", "mode", "run_id", "episode_id", "contract_sha256", "snapshot_sha256",
                          "epoch", "request_id", "observation_id"):
                if request[field] != p[field]:
                    raise ValueError(f"request differs from trace: {field}")
            for field in ("epoch", "request_id", "observation_id"):
                if token[field] != p[field]:
                    raise ValueError(f"native token differs from trace: {field}")
            tick, d = token["observation_policy_tick"], token["rtc_delay_steps"]
            if p["start_tick"] != tick or p["delay_steps"] != d:
                raise ValueError("RTC timeline differs")
            actions, base = np.asarray(reply["actions"]), np.asarray(p["base"])
            if not same(actions, base, (50, 14)) or not np.array_equal(actions, base) or not np.array_equal(actions, p["actions"]):
                raise ValueError("probe changed the Pi chunk")
            if not same(actions[:d], context["committed_actions"], (d, 14)):
                raise ValueError("RTC prefix changed")
            obs = observations[(token["epoch"], token["observation_id"])]
            if not all(obs["camera_valid"]) or not all(index >= 0 for index in obs["video_indices"]):
                raise ValueError("original observation lacks three-view video references")
            if not same(p["state"], obs["observation_state"], (14,)):
                raise ValueError("trace differs from original observation")
            if (context["previous_command_tick"] != tick - 1
                    or context["previous_command_submitted_at"] > token["observed_at"]
                    or not same(p["previous_command"], request["previous_command"], (14,))):
                raise ValueError("previous submitted command context differs")
            prior = by_tick.get((token["epoch"], tick - 1))
            if prior is None:
                notes.append({"request": list(key), "reason": "previous tick outside this episode; context retained"})
            elif not prior["submitted_valid"] or not same(p["previous_command"], prior["submitted_action"], (14,)):
                raise ValueError("previous command differs from actual submission")
            gate = p["gate"]
            if type(gate["active"]) is not bool:
                raise ValueError("invalid Thor gate")
            if gate["active"]:
                counts["active_" + p["arm"]] += 1
            counts["accepted_probe_replies"] += 1
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            errors.append({"request": list(key), "reason": str(exc), "refs": entry["refs"]})
    limits = manifest.get("action_limits") or {}
    lower, upper = np.asarray(limits.get("lower")), np.asarray(limits.get("upper"))
    have_limits = lower.shape == upper.shape == (14,)
    for row in rows:
        selection = row["details"].get("policy_selection") or {}
        if selection.get("fusion") != "rtc":
            continue
        try:
            token = selection["request"]
            key = (token["epoch"], str(token["request_id"]))
            source = replies[key]["reply"]
            index = selection["model_index"]
            if (key in cancelled or source.get("discarded") is not False or source.get("error")
                    or selection["target_tick"] != row["tick"] or token["epoch"] != row["epoch"]
                    or index != row["tick"] - token["observation_policy_tick"] or not 0 <= index < 50
                    or not row["submitted_valid"]):
                raise ValueError("invalid native RTC action source")
            target = np.asarray(source["actions"])[index]
            if not same(target, row["submitted_action"], (14,)):
                if have_limits and same(np.clip(target, lower, upper), row["submitted_action"], (14,)):
                    counts["known_sdk_clamped_ticks"] += 1
                else:
                    raise ValueError("actual command differs from source and declared SDK bounds")
            counts["source_verified_ticks"] += 1
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            errors.append({"raw_ref": row["raw_ref"], "reason": str(exc)})
    complete_visual = sum(all(v >= 0 for v in r["video_indices"]) and all(r["camera_valid"]) for r in rows)
    passed = not errors and counts["accepted_probe_replies"] > 0 and counts["source_verified_ticks"] > 0 and complete_visual > 0
    return {"status": "passed" if passed else "failed", "episode_id": manifest["episode_id"],
            "counts": dict(counts), "raw_rows": len(rows), "three_view_referenced_rows": complete_visual,
            "sidecar_events": len(events), "cancelled_requests": len(cancelled), "errors": errors, "notes": notes,
            "training_ready": False, "robot_success_rate": None, "semantic_gate_review_required": True}
