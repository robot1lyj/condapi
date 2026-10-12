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


def physical_candidates(rows, fk, episode_number=1):
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
        events = physical_events(times, g, z, valid, epochs, ticks)
        for serial, event in enumerate(events, 1):
            close, end = event["close"], event["end"]
            minimum = g[close]
            for i in range(close + 1, end + 1):
                minimum = min(minimum, g[i])
                if g[i] >= 0.5 and g[i] - minimum >= 0.18 and times[i] - times[close] >= 0.2:
                    end = i - 1
                    break
            fresh = [i for i in range(close, end + 1)
                     if rows[i]["video_indices"][col] >= 0 and rows[i]["camera_valid"][col]
                     and 0 <= times[i] - rows[i]["camera_times"][col] <= 0.3 and np.isfinite(z[i])]
            lifted = [i for i in fresh if z[i] - z[close] >= 0.05]
            picks = [event["entry"], close]
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
                               "lift_from_closure_m": float(z[i] - z[close]), "views": views})
            result.append({"candidate_id": f"E{episode_number:02d}-{arm[0].upper()}{serial:03d}-v2",
                           "episode_number": episode_number, "episode_id": rows[close]["episode_id"], "arm": arm,
                           "epoch": int(epochs[close]), "entry_row": event["entry"], "close_row": close,
                           "end_row": event["end"], "inspection_end_row": end, "end_reason": event["end_reason"],
                           "close_tick": int(ticks[close]), "close_time": float(times[close]),
                           "reference_base_z_m": float(z[close]), "frames": frames,
                           "lift_rows": lifted, "evidence_eligible": bool(lifted), "approved_reward": None})
    return result


def reconstruct_intervals(rows):
    """Actual [observation_i, observation_i+1) transitions at native cadence.

    Committed-prefix origins can belong to older replies. They must match both
    that reply and the current proposal's prefix, without rewriting their IDs.
    """
    by_tick, observations, origins, proposals = {}, {}, {}, {}
    audit = []
    for i, row in enumerate(rows):
        key = (row["epoch"], row["tick"])
        if key in by_tick:
            raise ValueError("duplicate epoch/tick in raw recording")
        by_tick[key] = i
        oid = row["details"].get("obs_id")
        if oid is not None and row.get("observation_valid", False):
            observations.setdefault((row["epoch"], oid), row["raw_ref"])
        reply = row["details"].get("policy_reply") or {}
        token = reply.get("token") or {}
        if not token or "actions" not in reply:
            continue
        origin = (token["epoch"], str(token["request_id"]))
        value = {"token": token, "actions": reply["actions"], "discarded": reply.get("discarded"), "error": reply.get("error")}
        if origin in origins and fingerprint(origins[origin]) != fingerprint(value):
            raise ValueError("same native reply identity has conflicting actions/status")
        origins[origin] = value
        p = reply.get("pld")
        if p is None:
            continue
        if origin in proposals and fingerprint(proposals[origin]) != fingerprint(p):
            raise ValueError("same native reply identity has conflicting PLD trace")
        if (p["epoch"] != token["epoch"] or str(p["request_id"]) != str(token["request_id"])
                or p["start_tick"] != token["observation_policy_tick"] or p["observation_id"] != token["observation_id"]):
            raise ValueError("PLD trace differs from native request token")
        if not np.allclose(reply["actions"], p.get("actions", p["base"]), atol=1e-6, rtol=0):
            raise ValueError("reply actions differ from PLD plan")
        proposals[origin] = p
    ordered = sorted(proposals.values(), key=lambda p: (p["epoch"], p["start_tick"]))
    result = []
    for p, nxt in pairwise(ordered):
        if not p["gate"]["active"]:
            continue
        try:
            if (p["epoch"] != nxt["epoch"] or p["run_id"] != nxt["run_id"] or p["episode_id"] != nxt["episode_id"]
                    or p["mode"] != nxt["mode"] or p["snapshot_sha256"] != nxt["snapshot_sha256"]):
                raise ValueError("identity changed before next observation")
            begin, end, epoch = p["start_tick"], nxt["start_tick"], p["epoch"]
            if not 0 < end - begin <= 50:
                raise ValueError("missing/reordered next observation or expired horizon")
            indexes = [by_tick[(epoch, t)] for t in range(begin, end + 1)]
            selected = [rows[i] for i in indexes[:-1]]
            origin = origins[(epoch, str(p["request_id"]))]
            if origin["discarded"] is not False or origin["error"]:
                raise ValueError("proposal canceled/rejected")
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
                if not np.allclose(row["submitted_action"], np.asarray(source["actions"])[k], atol=1e-6, rtol=0):
                    raise ValueError("actual submission differs from native selected source")
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
                       "window_end": ended.get("window_id") == p["window_id"], "window_end_reason": ended.get("reason")}
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
