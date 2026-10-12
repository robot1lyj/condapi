"""Portable CPU Replay-Curator: native HIL provenance, physical IDs and rewards.

No client attempt IDs, fixed execution cadence, robot I/O or neural imports.
Raw rows and model replies remain immutable; all joins are derived copies.
"""

from collections import Counter
from collections import defaultdict
import copy
from itertools import pairwise

import numpy as np

from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.grasp_events import physical_events
from yam_pld_rl.grasp_events import relative_physical_events


def physical_candidates(rows, fk, episode_number=1, *, segmentation="relative_v3"):
    times = np.asarray([r["time"] for r in rows], dtype=np.float64)
    states = np.asarray([r["measured_state"] for r in rows], dtype=np.float64)
    valid = np.asarray([r["measured_state_valid"] for r in rows])
    epochs, ticks = (np.asarray([r[k] for r in rows]) for k in ("epoch", "tick"))
    positions = {arm: np.full((len(rows), 3), np.nan) for arm in ("left", "right")}
    for i in np.flatnonzero(valid & np.isfinite(states).all(axis=1)):
        for arm, offset in (("left", 0), ("right", 7)):
            positions[arm][i] = fk.position(states[i, offset:offset + 6])
    result = []
    for arm, axis, col in (("left", 6, 1), ("right", 13, 2)):
        xyz, g = positions[arm], states[:, axis]
        z = xyz[:, 2]
        if segmentation not in {"legacy_v2", "relative_v3"}:
            raise ValueError("unknown physical segmentation")
        events = (relative_physical_events if segmentation == "relative_v3" else physical_events)(times, g, z, valid, epochs, ticks)
        for serial, event in enumerate(events, 1):
            close, end = event["close"], event["end"]
            reference = event.get("reference", close)
            minimum = g[close]
            for i in range(close + 1, end + 1):
                minimum = min(minimum, g[i])
                if g[i] >= 0.5 and g[i] - minimum >= 0.18 and times[i] - times[close] >= 0.2:
                    end = i - 1
                    break
            fresh = [i for i in range(close, end + 1)
                     if rows[i]["video_indices"][col] >= 0 and rows[i]["camera_valid"][col]
                     and 0 <= times[i] - rows[i]["camera_times"][col] <= 0.3 and np.isfinite(z[i])]
            lifted = [i for i in fresh if i >= reference and z[i] - z[reference] >= 0.05]
            picks = [event["entry"], close, reference]
            if fresh:
                picks += [min(fresh, key=lambda i: abs(times[i] - times[close] - t)) for t in (0.2, 0.45, 0.75, 1.15)]
            if lifted:
                picks += [lifted[0], max(lifted, key=lambda i: z[i]), lifted[-1]]
            frames = []
            for i in sorted(set(picks)):
                views = {}
                for view, column in (("top", 0), ("left", 1), ("right", 2)):
                    r = rows[i]
                    if r["video_indices"][column] < 0 or not r["camera_valid"][column] or not 0 <= times[i] - r["camera_times"][column] <= 0.3:
                        continue
                    views[view] = {"source_video": r["videos"][view], "video_frame_index": r["video_indices"][column],
                                   "capture_time": r["camera_times"][column]}
                if arm not in views:
                    continue
                frames.append({"row": i, "raw_ref": rows[i]["raw_ref"], "tick": int(ticks[i]), "epoch": int(epochs[i]),
                               "time": float(times[i]), "since_close_s": float(times[i] - times[close]),
                               "openness": float(g[i]), "position_m": xyz[i].tolist(),
                               "lift_from_closure_m": float(z[i] - z[reference]) if i >= reference else 0.0,
                               "height_from_reference_m": float(z[i] - z[reference]),
                               "before_reference": i < reference, "views": views})
            version = "v3" if segmentation == "relative_v3" else "v2"
            result.append({"candidate_id": f"E{episode_number:02d}-{arm[0].upper()}{serial:03d}-{version}",
                           "episode_number": episode_number, "episode_id": rows[close]["episode_id"], "arm": arm,
                           "epoch": int(epochs[close]), "entry_row": event["entry"], "close_row": close,
                           "end_row": event["end"], "inspection_end_row": end, "end_reason": event["end_reason"],
                           "close_tick": int(ticks[close]), "close_time": float(times[close]),
                           "reference_base_z_m": float(z[reference]), "reference_row": reference,
                           "segmentation": segmentation, "frames": frames,
                           "lift_rows": lifted, "evidence_eligible": bool(lifted), "approved_reward": None})
    return result


def native_replies(rows, events=()):
    """Join HDF5 and sidecar by native token; never manufacture a control row."""
    replies, cancelled = {}, set()
    entries = [(row["details"].get("policy_reply"), row["raw_ref"]) for row in rows]
    for event in events:
        payload = event.get("details", {}).get("policy_reply") or {}
        token = payload.get("token") or {}
        if event.get("event") == "cancelled" and token:
            cancelled.add((token["epoch"], str(token["request_id"])))
        if event.get("event") == "reply":
            entries.append((payload, event["audit_ref"]))
    for reply, ref in entries:
        if not reply or not reply.get("token"):
            continue
        token = reply["token"]
        key = (token["epoch"], str(token["request_id"]))
        # Timing/context can be present only in one copy. Execution status and
        # model output must agree in every copy of a received reply.
        value = {k: reply.get(k) for k in ("token", "actions", "discarded", "error", "pld")}
        if key in replies:
            if fingerprint(replies[key]["value"]) != fingerprint(value):
                raise ValueError("same native reply identity has conflicting HDF5/sidecar copies")
            replies[key]["refs"].append(ref)
        else:
            replies[key] = {"value": value, "reply": reply, "refs": [ref]}
    return replies, cancelled


def reconstruct_intervals(rows, events=(), *, diagnostics=False, action_limits=None):
    """Actual [observation_i, observation_i+1) transitions at native cadence.

    Committed-prefix origins can belong to older replies. They must match both
    that reply and the current proposal's prefix, without rewriting their IDs.
    """
    by_tick, observations, origins, proposals = {}, {}, {}, {}
    audit = []
    declared_bounds = None
    if action_limits is not None:
        lo, hi = (np.asarray(action_limits[k], dtype=np.float64) for k in ("lower", "upper"))
        if (lo.shape != (14,) or hi.shape != (14,) or not np.isfinite(lo).all() or not np.isfinite(hi).all()
                or np.any(lo >= hi) or action_limits.get("joint_unit") != "radian"
                or action_limits.get("gripper_unit") != "0_closed_1_open"):
            raise ValueError("invalid native SDK bounds for diagnostic action attribution")
        declared_bounds = lo, hi
    for i, row in enumerate(rows):
        key = (row["epoch"], row["tick"])
        if key in by_tick:
            raise ValueError("duplicate epoch/tick in raw recording")
        by_tick[key] = i
        oid = row["details"].get("obs_id")
        if oid is not None and row.get("observation_valid", False):
            observations.setdefault((row["epoch"], oid), row["raw_ref"])
    replies, cancelled = native_replies(rows, events)
    barriers = set()
    for event in events:
        if event.get("event") == "cancelled":
            token = (event.get("details", {}).get("policy_reply") or {}).get("token") or {}
            if token.get("observation_policy_tick") is not None:
                barriers.add((token["epoch"], token["observation_policy_tick"]))
    for origin, entry in replies.items():
        reply = entry["reply"]
        token = reply.get("token") or {}
        value = {"token": token, "actions": reply["actions"], "discarded": reply.get("discarded"), "error": reply.get("error")}
        if origin in cancelled:
            value.update(discarded=True, error=value["error"] or "request cancelled")
        origins[origin] = value
        p = reply.get("pld")
        if p is None:
            continue
        try:
            if (p["epoch"] != token["epoch"] or str(p["request_id"]) != str(token["request_id"])
                    or p["start_tick"] != token["observation_policy_tick"] or p["observation_id"] != token["observation_id"]
                    or p["episode_id"] != rows[0]["episode_id"]):
                raise ValueError("PLD trace differs from native request token/episode")
            if value["discarded"] is False and not value["error"] and not np.allclose(reply["actions"], p.get("wire_actions", p.get("actions", p["base"])), atol=1e-6, rtol=0):
                raise ValueError("reply actions differ from PLD plan")
        except (ValueError, KeyError, TypeError) as exc:
            value.update(discarded=True, error=str(exc))
            barriers.add((token["epoch"], token["observation_policy_tick"]))
            audit.append({"request": list(origin), "reason": str(exc), "refs": entry["refs"]})
            continue
        proposals[origin] = p
    ordered = sorted(proposals.values(), key=lambda p: (p["epoch"], p["start_tick"]))
    result = []
    for p, nxt in pairwise(ordered):
        if not p["gate"]["active"]:
            continue
        try:
            if p.get("replay_eligible") is False and not diagnostics:
                raise ValueError("probe-only diagnostics lack the formal actor/physical contract; not SAC replay")
            if (p["epoch"] != nxt["epoch"] or p["run_id"] != nxt["run_id"] or p["episode_id"] != nxt["episode_id"]
                    or p["mode"] != nxt["mode"] or p["snapshot_sha256"] != nxt["snapshot_sha256"]):
                raise ValueError("identity changed before next observation")
            begin, end, epoch = p["start_tick"], nxt["start_tick"], p["epoch"]
            if not 0 < end - begin <= 50:
                raise ValueError("missing/reordered next observation or expired horizon")
            if any(e == epoch and begin < t <= end for e, t in barriers):
                raise ValueError("interval crosses a cancelled or malformed native request")
            indexes = [by_tick[(epoch, t)] for t in range(begin, end + 1)]
            selected = [rows[i] for i in indexes[:-1]]
            origin = origins[(epoch, str(p["request_id"]))]
            if origin["discarded"] is not False or origin["error"]:
                raise ValueError("proposal canceled/rejected")
            successor = origins[(epoch, str(nxt["request_id"]))]
            if successor["discarded"] is not False or successor["error"]:
                raise ValueError("next observation response canceled/rejected")
            limited_ticks = []
            for t, row in zip(range(begin, end), selected, strict=True):
                if not row["submitted_valid"] or row["is_intervention"] or not row["intervention_valid"]:
                    raise ValueError("invalid or human-controlled actual action")
                selection = row["details"].get("policy_selection") or {}
                token = selection.get("request") or {}
                source = origins[(token["epoch"], str(token["request_id"]))]
                k = selection["model_index"]
                if (selection.get("fusion") != "rtc" or selection.get("target_tick") != t or token["epoch"] != epoch
                        or k != t - token["observation_policy_tick"] or not 0 <= k < 50
                        or source["discarded"] is not False or source["error"]):
                    raise ValueError("native policy selection cannot prove action source")
                own = str(token["request_id"]) == str(p["request_id"])
                if (own and t - begin < p["delay_steps"]) or (not own and (t - begin >= p["delay_steps"]
                                                                          or token["observation_policy_tick"] >= begin)):
                    raise ValueError("RTC prefix source rewritten or new proposal never took over")
                target = np.asarray(source["actions"])[k]
                if not np.allclose(row["submitted_action"], target, atol=1e-6, rtol=0):
                    known_mapping = diagnostics or p.get("baseline_transform") == "sdk_gripper_endpoints_v1"
                    delta = np.flatnonzero(np.abs(target - row["submitted_action"]) > 1e-6)
                    if (not known_mapping or declared_bounds is None or not set(delta).issubset({6, 13})
                            or not np.allclose(row["submitted_action"], np.clip(target, *declared_bounds), atol=1e-6, rtol=0)):
                        raise ValueError("actual submission differs from native selected source")
                    limited_ticks.append({"tick": t, "dimensions": np.flatnonzero(np.abs(target - row["submitted_action"]) > 1e-6).tolist()})
            obs_ref = observations[(epoch, p["observation_id"])]
            next_ref = observations[(epoch, nxt["observation_id"])]
            ended = nxt["gate"].get("ended") or {}
            receipt = {"decision_id": p["decision_id"], "epoch": epoch, "status": "executed", "source_verified": True,
                       "start_tick": begin, "end_tick": end, "ticks": list(range(begin, end)),
                       "monotonic_times": [rows[i]["time"] for i in indexes],
                       "submitted_actions": [r["submitted_action"] for r in selected],
                       "human_intervention": False, "client_modified": False,
                       "observation_ref": obs_ref, "next_observation_ref": next_ref,
                       "raw_refs": [r["raw_ref"] for r in selected],
                       "window_end": ended.get("window_id") == p["window_id"], "window_end_reason": ended.get("reason"),
                       "diagnostic_only": p.get("replay_eligible") is False or (bool(limited_ticks) and diagnostics),
                       "known_sdk_limited_ticks": limited_ticks,
                       "reply_refs": replies[(epoch, str(p["request_id"]))]["refs"]}
            result.append({"proposal": {**copy.deepcopy(p), "end_tick": end}, "receipt": receipt})
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            # Preserve a rejected slot so a window cannot silently reconnect
            # around an invalid or canceled decision during reward association.
            result.append({"proposal": copy.deepcopy(p), "receipt": {"status": "rejected", "reason": str(exc)}})
            audit.append({"decision": p["decision_id"], "reason": str(exc)})
    if ordered and ordered[-1]["gate"]["active"]:
        p = ordered[-1]
        result.append({"proposal": copy.deepcopy(p), "receipt": {"status": "rejected", "reason": "no next observation"}})
        audit.append({"decision": p["decision_id"], "reason": "no next observation"})
    # A residual may first execute in the NEXT observation interval's RTC
    # prefix. Keep transition spans unchanged, but verify its eventual effect
    # against original native ownership before this window's handoff.
    window_ends = {}
    for record in result:
        p, receipt = record["proposal"], record["receipt"]
        if receipt.get("window_end"):
            window_ends[(p["epoch"], p["window_id"])] = receipt["end_tick"]
    selected_by_origin = defaultdict(list)
    for row in rows:
        selection = row["details"].get("policy_selection") or {}
        token = selection.get("request") or {}
        if token and row["submitted_valid"] and not row["is_intervention"]:
            selected_by_origin[(token["epoch"], str(token["request_id"]))].append((row, selection))
    for record in result:
        p, receipt = record["proposal"], record["receipt"]
        if receipt["status"] != "executed" or diagnostics:
            continue
        terminal = window_ends.get((p["epoch"], p["window_id"]), p.get("end_tick", p["start_tick"]))
        effects = []
        physical_actions = np.asarray(p.get("physical_actions", p["actions"]))
        physical_base = np.asarray(p.get("physical_base", p["base"]))
        for row, selection in selected_by_origin[(p["epoch"], str(p["request_id"]))]:
            k = selection["model_index"]
            if (p["start_tick"] <= row["tick"] < terminal and k == row["tick"] - p["start_tick"]
                    and selection.get("fusion") == "rtc" and selection.get("target_tick") == row["tick"]
                    and 0 <= k < 50 and np.allclose(row["submitted_action"], physical_actions[k], atol=1e-6, rtol=0)
                    and np.any(np.abs(physical_actions[k] - physical_base[k]) > 1e-6)):
                effects.append({"tick": row["tick"], "model_index": k, "submitted_action": row["submitted_action"]})
        receipt["actor_effects_before_handoff"] = effects
    return result, audit


def associate_rewards(records, candidates, rewards, rows, split):
    """One reviewed physical candidate per fully verified online window."""
    if split not in {"train", "val"}:
        raise ValueError("episode-level split required")
    reviewed = {r["candidate_id"]: r for r in rewards}
    if len(reviewed) != len(rewards):
        raise ValueError("duplicate reviewed candidate")
    windows = defaultdict(list)
    for r in records:
        p = r["proposal"]
        windows[(p["run_id"], p["episode_id"], p["epoch"], p["window_id"])].append(r)
    counts = Counter()
    matches = {}
    for key, window in windows.items():
        window.sort(key=lambda r: r["proposal"]["start_tick"])
        p, last = window[0]["proposal"], window[-1]
        if "end_tick" not in last["receipt"]:
            matches[key] = []
            continue
        matches[key] = [c for c in candidates if c["episode_id"] == p["episode_id"] and c["epoch"] == p["epoch"]
                        and c["arm"] == p["arm"] and p["start_tick"] <= c["close_tick"] < last["receipt"]["end_tick"]]
        counts.update(c["candidate_id"] for c in matches[key])
    output, labels, audit = [], [], []
    for key, window in windows.items():
        try:
            if any(r["proposal"].get("replay_eligible") is False or r["receipt"].get("diagnostic_only") for r in window):
                raise ValueError("diagnostic probe cannot publish training rewards")
            if any(r["receipt"]["status"] != "executed" for r in window):
                raise ValueError("window contains rejected execution intervals")
            last = window[-1]
            if not last["receipt"]["window_end"]:
                raise ValueError("missing Thor window handoff observation")
            if last["receipt"]["window_end_reason"] not in {"lift_guard", "reopened", "timeout"}:
                raise ValueError("window truncated by observation gap or invalid handoff")
            match = matches[key]
            if len(match) != 1 or counts[match[0]["candidate_id"]] != 1:
                raise ValueError("online window does not uniquely match one physical closure")
            c = match[0]
            reward = reviewed[c["candidate_id"]]
            if reward.get("candidate_sha256") != c.get("candidate_sha256") or not c.get("candidate_sha256"):
                raise ValueError("review evidence release changed")
            if reward.get("approved") is not True or reward.get("status") not in {"held", "empty"} or not reward.get("reviewer"):
                raise ValueError("binary visual outcome not reviewed/published")
            end = last["receipt"]["end_tick"]
            evidence = reward["evidence_rows"]
            lifted = set(c["lift_rows"])
            if not evidence or any(i not in lifted or rows[i]["tick"] > end for i in evidence):
                raise ValueError("visual outcome lacks fresh 5cm evidence inside expert window")
            outcome = max(rows[i]["tick"] for i in evidence)
            attempt_id = c["candidate_id"]  # assigned here, never sent by client
            output.extend({"proposal": {**r["proposal"], "attempt_id": attempt_id}, "receipt": r["receipt"]} for r in window)
            labels.append({"run_id": key[0], "episode_id": key[1], "attempt_id": attempt_id,
                           "final_decision_id": last["proposal"]["decision_id"], "terminal_tick": end,
                           "close_tick": c["close_tick"], "outcome_tick": outcome, "status": reward["status"],
                           "reviewer": reward["reviewer"], "evidence": [rows[i]["raw_ref"] for i in evidence],
                           "lift_m": min(next(f["lift_from_closure_m"] for f in c["frames"] if f["row"] == i) for i in evidence),
                           "held_after_lift": reward["status"] == "held", "view_valid": True, "split": split})
        except (ValueError, KeyError, TypeError, IndexError, StopIteration) as exc:
            audit.append({"window": key, "reason": str(exc)})
    return output, labels, audit
