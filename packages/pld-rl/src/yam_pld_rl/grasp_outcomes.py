"""Offline, causal outcome evidence for drained grasp windows (no models).

No-lift is a failed *goal*, not a claim that an occluded gripper is empty.
All proposed labels still require review; missing data is never a zero reward.
"""

from collections import defaultdict

import numpy as np


def attach_outcomes(candidates, records, rows, fk, episode_number=1):
    windows = defaultdict(list)
    for record in records:
        p = record["proposal"]
        windows[(p["epoch"], p["window_id"])].append(record)
    by_tick = {(r["epoch"], r["tick"]): i for i, r in enumerate(rows)}
    for serial, ((epoch, wid), unordered) in enumerate(windows.items(), 1):
        records_ = sorted(unordered, key=lambda r: r["proposal"]["start_tick"])
        first, last = records_[0]["proposal"], records_[-1]["receipt"]
        end_info = last.get("window_end_metadata") or {}
        if (end_info.get("gate_version") != 3 or not last.get("window_end")
                or any(r["receipt"].get("status") != "executed" for r in records_)):
            continue
        begin, end, arm = first["start_tick"], last["end_tick"], first["arm"]
        matched = [c for c in candidates if c["epoch"] == epoch and c["arm"] == arm
                   and begin <= c["close_tick"] < end and c.get("kind") != "no_closure"]
        if len(matched) > 1:
            continue
        if not matched:
            if end_info.get("close_tick") is not None or end_info.get("reason") != "forecast_timeout":
                continue
            # Keep a forecast-only task attempt for review, without inventing a
            # physical closure or reusing an earlier physical candidate ID.
            c = {"candidate_id": f"E{episode_number:02d}-W{serial:03d}-v4", "kind": "no_closure",
                 "episode_number": episode_number, "episode_id": first["episode_id"], "arm": arm,
                 "epoch": epoch, "close_tick": begin, "close_confirmed": False,
                 "entry_row": by_tick[(epoch, begin)], "close_row": by_tick[(epoch, begin)],
                 "end_row": by_tick[(epoch, end)], "inspection_end_row": by_tick[(epoch, end)],
                 "reference_row": by_tick[(epoch, begin)], "reference_base_z_m": 0.,
                 "end_reason": "forecast_timeout", "frames": [], "lift_rows": [],
                 "evidence_eligible": False, "approved_reward": None, "segmentation": "window_forecast_v4"}
            candidates.append(c)
        else:
            c = matched[0]
        try:
            indexes = [by_tick[(epoch, tick)] for tick in range(begin, end+1)]
            settle = end_info["drain_until_tick"]
            settle_time = rows[by_tick[(epoch, settle)]]["time"]
        except KeyError:
            continue
        valid = all(rows[i]["measured_state_valid"] and rows[i]["observation_valid"]
                    and np.isfinite(rows[i]["measured_state"]).all() for i in indexes)
        axis, column = (0, 1) if arm == "left" else (7, 2)
        minimum, max_lift, fresh, lifted = float("inf"), 0., [], []
        frames = {f["row"]: f for f in c["frames"]}
        picks = {begin, c["close_tick"], end_info["stop_tick"], (begin+end_info["stop_tick"])//2}
        for i in indexes:
            r = rows[i]
            if not r["measured_state_valid"] or not np.isfinite(r["measured_state"]).all():
                continue
            xyz = fk.position(np.asarray(r["measured_state"])[axis:axis+6])
            if r["tick"] == c["close_tick"]:
                minimum = float(xyz[2])
            minimum = min(minimum, float(xyz[2]))
            lift = float(xyz[2])-minimum if r["tick"] >= c["close_tick"] else 0.
            max_lift = max(max_lift, lift)
            visible = (r["video_indices"][column] >= 0 and r["camera_valid"][column]
                       and 0 <= r["time"]-r["camera_times"][column] <= .3)
            settled = visible and r["tick"] >= settle and r["camera_times"][column] >= settle_time
            if not visible or (not settled and r["tick"] not in picks):
                continue
            views = {}
            for view, col in (("top", 0), ("left", 1), ("right", 2)):
                if (r["video_indices"][col] >= 0 and r["camera_valid"][col]
                        and 0 <= r["time"]-r["camera_times"][col] <= .3):
                    views[view] = {"source_video": r["videos"][view], "video_frame_index": r["video_indices"][col],
                                   "capture_time": r["camera_times"][col]}
            if settled:
                fresh.append(i)
            if settled and lift >= .05:
                lifted.append(i)
            frames[i] = {"row": i, "raw_ref": r["raw_ref"], "tick": r["tick"], "epoch": epoch,
                         "time": r["time"], "since_close_s": (r["tick"]-c["close_tick"])/30,
                         "openness": r["measured_state"][axis+6], "position_m": xyz.tolist(),
                         "lift_from_closure_m": lift, "height_from_reference_m": lift,
                         "before_reference": r["tick"] < c["close_tick"], "after_queue_drain": settled, "views": views}
        c["frames"] = [frames[i] for i in sorted(frames) if frames[i]["tick"] <= end]
        c.update(scoring_end_tick=end, scoring_start_tick=settle,
                 outcome={"schema": "drained_grasp_v1", "window_id": wid, "terminal_tick": end,
                          "stop_tick": end_info["stop_tick"], "drain_until_tick": settle,
                          "reason": end_info["reason"], "close_confirmed": c.get("kind") != "no_closure",
                          "kinematic_complete": valid, "max_lift_m": max_lift,
                          "fresh_rows": fresh, "lifted_rows": lifted})


def reviewed_outcome(candidate, review):
    """Validate goal evidence; return the immutable proof copied to replay."""
    p = candidate["outcome"]
    status, evidence = review["status"], review["evidence_rows"]
    fresh = p["fresh_rows"]
    if not evidence or not fresh or max(evidence) != fresh[-1] or any(i not in fresh for i in evidence):
        raise ValueError("outcome requires latest fresh evidence after queued corrections")
    if not p["kinematic_complete"]:
        raise ValueError("invalid measured trajectory; cannot turn missing data into task failure")
    if status in {"held", "empty"}:
        if not p["close_confirmed"] or any(i not in p["lifted_rows"] for i in evidence):
            raise ValueError("held/empty require post-drain lift evidence")
        if status == "held":
            frames = {f["row"]: f for f in candidate["frames"]}
            images = {(frames[i]["views"][candidate["arm"]]["source_video"],
                       frames[i]["views"][candidate["arm"]]["video_frame_index"]) for i in evidence}
            if len(images) < 2:
                raise ValueError("held requires two distinct post-drain frames, not duplicate captures")
    elif status in {"not_lifted", "no_closure"}:
        if review.get("grasp_intent_confirmed") is not True:
            raise ValueError("failure requires reviewed grasp intent, not an unverified gate trigger")
        if status == "not_lifted" and (not p["close_confirmed"] or p["max_lift_m"] >= .05
                or p["reason"] not in {"reopened", "closed_timeout", "timeout"}):
            raise ValueError("not_lifted contradicts measured goal trajectory or stop reason")
        if status == "no_closure" and (p["close_confirmed"] or p["reason"] != "forecast_timeout"):
            raise ValueError("no_closure contradicts measured closure")
    elif status == "dropped":
        if review.get("held_then_lost_confirmed") is not True:
            raise ValueError("dropped requires reviewed temporal visual evidence")
    else:
        raise ValueError("unsupported grasp outcome")
    return {**p, "status": status, "grasp_intent_confirmed": review.get("grasp_intent_confirmed", False),
            "held_then_lost_confirmed": review.get("held_then_lost_confirmed", False)}
