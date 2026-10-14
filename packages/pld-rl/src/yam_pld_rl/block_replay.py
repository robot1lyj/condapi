"""Join immutable proposals, actual execution receipts and delayed rewards."""

from collections import defaultdict

import numpy as np

from yam_pld_rl.block_contract import array
from yam_pld_rl.block_contract import decode
from yam_pld_rl.block_contract import feasible_bounds
from yam_pld_rl.block_contract import feature_dim
from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.block_contract import integer
from yam_pld_rl.block_contract import map_action
from yam_pld_rl.block_contract import observation_vector
from yam_pld_rl.block_contract import physical_baseline
from yam_pld_rl.block_contract import validate_contract
from yam_pld_rl.block_contract import window_context
from yam_pld_rl.block_contract import wire_plan
from yam_pld_rl.collector import execution_contract
from yam_pld_rl.collector import validate_collector
from yam_pld_rl.collector import validate_collector_trace
from yam_pld_rl.trajectory import planning_context


def compile_replay(records, labels, contract, snapshots):
    """Reject a broken attempt as a unit, never reconnect across a lost command.

    Curator assigns attempt IDs after physical segmentation. Actual intervals
    follow native RTC observations, not the planned residual envelope length.
    Snapshots maps accepted manifest hashes to full immutable manifest content.
    """
    validate_contract(contract)
    cid = fingerprint(contract)
    feature_identity = None
    for manifest in snapshots.values():
        validate_collector(manifest)
        if manifest["contract"] != contract:
            raise ValueError("mixed contracts in replay")
        identity = (manifest["encoder_sha256"], fingerprint(manifest["normalization"]))
        if feature_identity is not None and identity != feature_identity:
            raise ValueError("mixed visual/normalization identities")
        feature_identity = identity
    label_map = {}
    for label in labels:
        key = (label["run_id"], label["episode_id"], label["attempt_id"])
        if key in label_map:
            raise ValueError("duplicate attempt reward; create an explicit new reward release")
        label_map[key] = label
    groups, seen = defaultdict(list), set()
    executed_ticks = defaultdict(set)
    for row in records:
        p = row["proposal"]
        key = (p["run_id"], p["episode_id"], p["attempt_id"])
        unique = (*key[:2], p["decision_id"])
        if unique in seen:
            raise ValueError("duplicate decision ID")
        seen.add(unique)
        groups[key].append(row)
        receipt = row.get("receipt", {})
        if receipt.get("status") == "executed":
            slot = (*key[:2], integer(receipt["epoch"], "epoch"))
            claim = set(receipt["ticks"])
            if executed_ticks[slot].intersection(claim):
                raise ValueError("overlapping executed decisions: cannot choose which reward owns these ticks")
            executed_ticks[slot].update(claim)
    # A physical tick cannot be assigned to two residual decisions, even if
    # their active arms differ. Canceled/rejected proposals remain in raw logs.
    occupied = defaultdict(set)
    excluded, rows = [], []
    for key, attempt in groups.items():
        try:
            label = label_map[key]
            modern = contract.get("reward_contract") == "drained_grasp_v1"
            statuses = {"held", "empty", "not_lifted", "no_closure", "dropped"} if modern else {"held", "empty"}
            if label["status"] not in statuses or not label.get("reviewer") or not label.get("evidence"):
                raise ValueError("not a reviewed binary outcome with evidence")
            needs_lift = label["status"] in {"held", "empty"}
            if (not np.isfinite(label["lift_m"]) or (needs_lift and label["lift_m"] < 0.05) or label.get("view_valid") is not True):
                raise ValueError("missing lift/visual evidence; absence of segmentation is not empty")
            if label["status"] == "held" and label.get("held_after_lift") is not True:
                raise ValueError("success must be retained after lifting")
            if label["status"] == "empty" and label.get("held_after_lift") is not False:
                raise ValueError("contradictory or missing empty outcome evidence")
            attempt.sort(key=lambda r: r["proposal"]["start_tick"])
            last = attempt[-1]["proposal"]
            if modern:
                proof = label["outcome_proof"]
                end_info = attempt[-1]["receipt"]["window_end_metadata"]
                stop, settle = proof["stop_tick"], proof["drain_until_tick"]
                if (proof["schema"] != "drained_grasp_v1" or proof["terminal_tick"] != last["end_tick"]
                        or proof["status"] != label["status"] or not proof["kinematic_complete"]
                        or proof["window_id"] != last["window_id"]
                        or end_info.get("gate_version") != 3 or end_info["stop_tick"] != stop
                        or end_info["drain_until_tick"] != settle
                        or last["end_tick"] < settle + contract["gate"]["settle_observation_ticks"]):
                    raise ValueError("unverified drained reward endpoint")
                correcting = [r["proposal"] for r in attempt if r["proposal"]["gate"].get("phase") == "correcting"]
                if not correcting:
                    raise ValueError("no correction phase")
                latest_pending = max(p["start_tick"]+(p["trajectory_context"]["end_index"]
                    if contract.get("trajectory") else p["delay_steps"]+contract["block_steps"])-1 for p in correcting)
                if contract.get("trajectory"):
                    stop_proposal = next((r["proposal"] for r in attempt if r["proposal"]["start_tick"] == stop), None)
                    if stop_proposal is None:
                        raise ValueError("missing accepted stop proposal")
                    latest_pending = min(latest_pending, stop+stop_proposal["delay_steps"]-1)
                if settle < latest_pending:
                    raise ValueError("reward precedes pending correction tail")
                if last["gate"].get("phase") != "draining" or any(
                    r["proposal"]["start_tick"] >= stop and r["proposal"]["gate"].get("apply_residual") is not False for r in attempt
                ):
                    raise ValueError("new correction generated after stop")
                if label["status"] == "not_lifted" and (not proof["close_confirmed"] or proof["max_lift_m"] >= .05
                        or not proof["grasp_intent_confirmed"] or proof["reason"] not in {"reopened", "closed_timeout", "timeout"}):
                    raise ValueError("invalid no-lift goal failure")
                if label["status"] == "no_closure" and (proof["close_confirmed"] or proof["reason"] != "forecast_timeout"
                        or not proof["grasp_intent_confirmed"]):
                    raise ValueError("invalid no-closure goal failure")
                if label["status"] == "dropped" and not proof["held_then_lost_confirmed"]:
                    raise ValueError("unreviewed dropped outcome")
            if last["decision_id"] != label["final_decision_id"] or last["end_tick"] != label["terminal_tick"]:
                raise ValueError("reward endpoint differs from actual transition")
            if not attempt[0]["proposal"]["start_tick"] <= label["close_tick"] <= label["outcome_tick"] <= last["end_tick"]:
                raise ValueError("outcome outside intervention window; cannot credit later Pi recovery")
            arm, mode, snapshot = last["arm"], last["mode"], last["snapshot_sha256"]
            if mode not in {"probe", "collect"}:
                raise ValueError("shadow/evaluation are not training data")
            if label.get("split") not in {"train", "val"}:
                raise ValueError("preassigned episode split required")
            local, claims, previous_end, epoch = [], [], None, None
            previous_time = None
            for index, record in enumerate(attempt):
                p, r = record["proposal"], record["receipt"]
                if (p["schema"] != contract["schema"] or p["contract_sha256"] != cid or p["snapshot_sha256"] not in snapshots
                        or p["snapshot_sha256"] != snapshot or p["arm"] != arm or p["mode"] != mode):
                    raise ValueError("attempt changes contract/snapshot/arm/mode")
                if execution_contract(snapshots[snapshot], mode) != contract:
                    raise ValueError("execution coordinates differ from requested replay")
                begin, end = integer(p["start_tick"], "start"), integer(p["end_tick"], "end")
                delay = integer(p["delay_steps"], "delay")
                if not 0 < end - begin <= 50:
                    raise ValueError("actual interval outside proposal horizon")
                if previous_end is not None and begin != previous_end:
                    raise ValueError("gap/overlap between decisions; no fabricated adjacency")
                epoch_now = integer(r["epoch"], "epoch")
                if epoch is not None and epoch_now != epoch:
                    raise ValueError("controller epoch changed mid-attempt")
                epoch = epoch_now
                if r["decision_id"] != p["decision_id"] or r["status"] != "executed" or r.get("source_verified") is not True:
                    raise ValueError("actual interval/source not verified")
                if p["epoch"] != epoch_now or p["window_id"] != last["window_id"]:
                    raise ValueError("online window or epoch changed")
                if r["start_tick"] != begin or r["end_tick"] != end:
                    raise ValueError("actual execution interval mismatch")
                if r.get("human_intervention") is not False or r.get("client_modified") is not False:
                    raise ValueError("HIL/client modification is not expert execution")
                if not r.get("observation_ref") or not r.get("next_observation_ref"):
                    raise ValueError("raw observation/video provenance missing")
                if (index and attempt[index - 1]["receipt"]["next_observation_ref"] != r["observation_ref"]):
                    raise ValueError("next observation is not next decision observation")
                n = end - begin
                ticks = np.asarray(r["ticks"])
                if not np.array_equal(ticks, np.arange(begin, end)):
                    raise ValueError("missing/repeated control ticks")
                # Float64 timestamps are mandatory in storage (float32 loses
                # subsecond precision at long IPC monotonic uptimes).
                times = np.asarray(r["monotonic_times"], dtype=np.float64)
                if (times.shape != (n + 1,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or np.max(np.diff(times)) > 0.15
                        or (previous_time is not None and abs(times[0] - previous_time) > 1e-6)):
                    raise ValueError("execution timing gap")
                dt = times[-1] - times[0]
                if abs(dt - n * contract["action_dt_s"]) > max(0.05, n * contract["action_dt_s"] * 0.15):
                    raise ValueError("actual time inconsistent with policy ticks")
                base = p.get("physical_base", p["base"])
                context = None
                if contract.get("trajectory"):
                    context = planning_context(base, p["previous_command"], p["state"], p["gate"], arm, delay, contract)
                    recorded_context = p["trajectory_context"]
                    for field in ("kind", "reason", "end_index", "delay_steps", "lift_guard_start_index", "lift_ceiling_m"):
                        if context[field] != recorded_context[field]:
                            raise ValueError("trajectory context metadata differs")
                    for field in ("bias", "weight"):
                        if not np.allclose(context[field], recorded_context[field], atol=1e-7, rtol=0):
                            raise ValueError("trajectory context differs")
                scale, negative_scale = feasible_bounds(base, p["previous_command"], arm, delay, contract, context=context)
                draining = p.get("gate", {}).get("apply_residual") is False
                if draining:
                    if contract["gate"].get("version") != 3 or p["gate"].get("phase") != "draining":
                        raise ValueError("unknown forced no-op phase")
                    scale, negative_scale = np.zeros(7, dtype=np.float32), np.zeros(7, dtype=np.float32)
                if contract.get("baseline_transform"):
                    baseline = physical_baseline(p["raw_base"], delay, contract)
                    if not np.array_equal(baseline, np.asarray(base)):
                        raise ValueError("physical Pi baseline differs from declared SDK transform")
                if not np.allclose(scale, array(p["scale"], (7,), "scale"), atol=1e-6, rtol=0):
                    raise ValueError("feasible scale differs")
                if not np.allclose(negative_scale, array(p.get("negative_scale", p["scale"]), (7,), "negative scale"), atol=1e-6, rtol=0):
                    raise ValueError("negative feasible scale differs")
                u, action = array(p["sample_u"], (7,), "u"), array(p["action"], (7,), "action")
                expected_a = np.zeros(7) if mode == "probe" else map_action(u, scale, negative_scale)
                if np.any(np.abs(u) > 1.000001) or not np.allclose(action, expected_a, atol=1e-6, rtol=0):
                    raise ValueError("wrong actor action")
                predicted = decode(base, action, arm, delay, contract, context=context)
                if not np.allclose(predicted, array(p.get("physical_actions", p["actions"]), (50, 14), "proposal actions"), atol=1e-6, rtol=0):
                    raise ValueError("proposal decoder mismatch")
                if contract.get("baseline_transform") and not np.array_equal(
                    wire_plan(predicted, p["raw_base"], delay), np.asarray(p["wire_actions"])
                ):
                    raise ValueError("wire plan changed native RTC prefix or physical suffix")
                actual = array(r["submitted_actions"], (n, 14), "actually submitted targets")
                if not np.allclose(actual, predicted[:n], atol=1e-6, rtol=0):
                    raise ValueError("actual actions differ; cannot relabel a clipped proposal as executed")
                realized = bool(np.any(np.abs(actual - np.asarray(base)[:n]) > 1e-6))
                if "actor_effects_before_handoff" in r:
                    realized = False
                    for effect in r["actor_effects_before_handoff"]:
                        effect_tick = integer(effect["tick"], "actor effect tick")
                        k = integer(effect["model_index"], "actor effect index")
                        if (effect_tick != begin + k or not 0 <= k < 50
                                or not begin <= effect_tick < label["terminal_tick"]
                                or not np.allclose(effect["submitted_action"], predicted[k], atol=1e-6, rtol=0)
                                or not np.any(np.abs(predicted[k] - np.asarray(base)[k]) > 1e-6)):
                            raise ValueError("invalid delayed native actor effect proof")
                        if modern and contract.get("trajectory") and effect_tick > settle:
                            raise ValueError("correction after the verified committed tail")
                        realized = True
                slot = (*key[:2], epoch)
                claims.append((slot, set(range(begin, end))))
                if occupied[slot].intersection(claims[-1][1]):
                    raise ValueError("overlap with another attempt")
                terminal = index == len(attempt) - 1
                dim = feature_dim(contract)
                x = array(p["x"], (dim,), "x")
                if not np.array_equal(x[-7:], scale):
                    raise ValueError("scale is not bound to policy observation")
                extra = {"negative_scale": negative_scale, "context": window_context(p["gate"], begin, p["base_z_m"][arm], contract["action_dt_s"])} if modern else {}
                reconstructed = observation_vector(x[:1536], p["state"], base, p["previous_command"],
                                                   arm, delay, p["elapsed_seconds"], scale,
                                                   snapshots[snapshot]["normalization"], **extra)
                if not np.allclose(x, reconstructed, atol=1e-6, rtol=0):
                    raise ValueError("causal observation vector mismatch")
                validate_collector_trace(p, snapshots[snapshot])
                if index and not np.allclose(p["previous_command"], attempt[index - 1]["receipt"]["submitted_actions"][-1], atol=1e-6, rtol=0):
                    raise ValueError("command continuity anchor changed")
                nx = np.zeros_like(x) if terminal else array(attempt[index + 1]["proposal"]["x"], (dim,), "nx")
                next_p = None if terminal else attempt[index+1]["proposal"]
                next_negative = np.zeros(7) if terminal else array(next_p.get("negative_scale", next_p["scale"]), (7,), "next negative scale")
                local.append({"x": x, "nx": nx, "scale": scale, "next_scale": nx[-7:], "action": action,
                              "negative_scale": negative_scale, "next_negative_scale": next_negative,
                              "reward": float(terminal and label["status"] == "held"), "terminal": float(terminal),
                              "discount": contract["discount_per_second"] ** dt, "dt": dt,
                              "online": mode == "collect", "realized_nonzero": realized, "train": label["split"] == "train",
                              "arm": arm == "right", "attempt": "/".join(key), "episode": "/".join(key[:2]),
                              "decision": p["decision_id"]})
                if contract.get("trajectory"):
                    off = 0 if arm == "left" else 7
                    # Only for actor coordinate initialization. This is NOT
                    # the executed action and never enters the critic target.
                    local[-1]["prior_action"] = array(p["actor_physical_center"], (7,), "prior center") / np.asarray(contract["radius"][off:off+7])
                previous_end, previous_time = end, times[-1]
            if mode == "collect" and not any(row["realized_nonzero"] for row in local):
                raise ValueError("collect window has no verified nonzero executed correction")
            rows.extend(local)
            for slot, claim in claims:
                occupied[slot].update(claim)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            excluded.append({"attempt": key, "reason": str(exc)})
    # The evaluation split is at episode level, never frames of one episode.
    episode_splits = defaultdict(set)
    for row in rows:
        episode_splits[row["episode"]].add(row["train"])
    if any(len(v) != 1 for v in episode_splits.values()):
        raise ValueError("episode split leakage")
    if not rows:
        raise ValueError(f"no eligible replay: {excluded}")
    data = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
    return data, {"accepted_attempts": len({r["attempt"] for r in rows}), "transitions": len(rows),
                  "excluded": excluded, "reward_location": "last verified native observation interval of a server-associated window"}
