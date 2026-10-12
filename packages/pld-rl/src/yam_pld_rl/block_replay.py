"""Join immutable proposals, actual execution receipts and delayed rewards."""

from collections import defaultdict

import numpy as np

from yam_pld_rl.block_contract import FEATURE_DIM
from yam_pld_rl.block_contract import array
from yam_pld_rl.block_contract import decode
from yam_pld_rl.block_contract import feasible_scale
from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.block_contract import integer
from yam_pld_rl.block_contract import observation_vector
from yam_pld_rl.block_contract import validate_contract


def compile_replay(records, labels, contract, snapshots):
    """Reject a broken attempt as a unit, never reconnect across a lost command.

    A label is keyed to the final completed block, NOT a human keypress time.
    Snapshots maps accepted manifest hashes to full immutable manifest content.
    """
    validate_contract(contract)
    cid = fingerprint(contract)
    feature_identity = None
    for manifest in snapshots.values():
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
    for row in records:
        p = row["proposal"]
        key = (p["run_id"], p["episode_id"], p["attempt_id"])
        unique = (*key[:2], p["decision_id"])
        if unique in seen:
            raise ValueError("duplicate decision ID")
        seen.add(unique)
        groups[key].append(row)
    # A physical tick cannot be assigned to two residual decisions, even if
    # their active arms differ. Canceled/rejected proposals remain in raw logs.
    occupied = defaultdict(set)
    excluded, rows = [], []
    for key, attempt in groups.items():
        try:
            label = label_map[key]
            if label["status"] not in {"held", "empty"} or not label.get("reviewer") or not label.get("evidence"):
                raise ValueError("not a reviewed binary outcome with evidence")
            if label["lift_m"] < 0.05 or label.get("view_valid") is not True:
                raise ValueError("missing lift/visual evidence; absence of segmentation is not empty")
            if label["status"] == "held" and label.get("held_after_lift") is not True:
                raise ValueError("success must be retained after lifting")
            attempt.sort(key=lambda r: r["proposal"]["start_tick"])
            last = attempt[-1]["proposal"]
            if last["decision_id"] != label["final_decision_id"] or last["end_tick"] != label["outcome_tick"]:
                raise ValueError("reward endpoint differs from completed block")
            arm, mode, snapshot = last["arm"], last["mode"], last["snapshot_sha256"]
            if mode not in {"probe", "collect"}:
                raise ValueError("shadow/evaluation are not training data")
            if label.get("split") not in {"train", "val"}:
                raise ValueError("preassigned episode split required")
            local, claims, previous_end, epoch = [], [], None, None
            previous_time = None
            for index, record in enumerate(attempt):
                p, r = record["proposal"], record["receipt"]
                if (p["contract_sha256"] != cid or p["snapshot_sha256"] not in snapshots
                        or p["snapshot_sha256"] != snapshot or p["arm"] != arm or p["mode"] != mode):
                    raise ValueError("attempt changes contract/snapshot/arm/mode")
                begin, end = integer(p["start_tick"], "start"), integer(p["end_tick"], "end")
                delay = integer(p["delay_steps"], "delay")
                if end - begin != delay + contract["block_steps"]:
                    raise ValueError("block duration mismatch")
                if previous_end is not None and begin != previous_end:
                    raise ValueError("gap/overlap between decisions; no fabricated adjacency")
                epoch_now = integer(r["epoch"], "epoch")
                if epoch is not None and epoch_now != epoch:
                    raise ValueError("controller epoch changed mid-attempt")
                epoch = epoch_now
                if r["decision_id"] != p["decision_id"] or r["status"] != "executed":
                    raise ValueError("proposal not completely executed")
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
                scale = feasible_scale(p["base"], p["previous_command"], arm, delay, contract)
                if not np.allclose(scale, array(p["scale"], (7,), "scale"), atol=1e-6, rtol=0):
                    raise ValueError("feasible scale differs")
                u, action = array(p["sample_u"], (7,), "u"), array(p["action"], (7,), "action")
                expected_a = np.zeros(7) if mode == "probe" else scale * u
                if np.any(np.abs(u) > 1.000001) or not np.allclose(action, expected_a, atol=1e-6, rtol=0):
                    raise ValueError("wrong actor action")
                predicted = decode(p["base"], action, arm, delay, contract)
                if not np.allclose(predicted, array(p["actions"], (50, 14), "proposal actions"), atol=1e-6, rtol=0):
                    raise ValueError("proposal decoder mismatch")
                actual = array(r["submitted_actions"], (n, 14), "actually submitted targets")
                if not np.allclose(actual, predicted[:n], atol=1e-6, rtol=0):
                    raise ValueError("actual actions differ; cannot relabel a clipped proposal as executed")
                slot = (*key[:2], epoch)
                claims.append((slot, set(range(begin, end))))
                if occupied[slot].intersection(claims[-1][1]):
                    raise ValueError("overlap with another attempt")
                terminal = index == len(attempt) - 1
                x = array(p["x"], (FEATURE_DIM,), "x")
                if not np.array_equal(x[-7:], scale):
                    raise ValueError("scale is not bound to policy observation")
                reconstructed = observation_vector(x[:1536], p["state"], p["base"], p["previous_command"],
                                                   arm, delay, p["elapsed_seconds"], scale,
                                                   snapshots[snapshot]["normalization"])
                if not np.allclose(x, reconstructed, atol=1e-6, rtol=0):
                    raise ValueError("causal observation vector mismatch")
                if index and not np.allclose(p["previous_command"], attempt[index - 1]["receipt"]["submitted_actions"][-1], atol=1e-6, rtol=0):
                    raise ValueError("command continuity anchor changed")
                nx = np.zeros_like(x) if terminal else array(attempt[index + 1]["proposal"]["x"], (FEATURE_DIM,), "nx")
                local.append({"x": x, "nx": nx, "scale": scale, "next_scale": nx[-7:], "action": action,
                              "reward": float(terminal and label["status"] == "held"), "terminal": float(terminal),
                              "discount": contract["discount_per_second"] ** dt, "dt": dt,
                              "online": mode == "collect", "train": label["split"] == "train",
                              "arm": arm == "right", "attempt": "/".join(key), "episode": "/".join(key[:2]),
                              "decision": p["decision_id"]})
                previous_end, previous_time = end, times[-1]
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
                  "excluded": excluded, "reward_location": "last fully executed decision of each reviewed attempt"}
