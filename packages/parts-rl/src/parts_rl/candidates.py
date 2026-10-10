"""Conservative offline grasp proposals. No reward is approved by this module."""
# ruff: noqa: RUF001 -- Human-facing Chinese punctuation is intentional.

import numpy as np


def closing_events(times, openness, valid, epochs, ticks):
    """Hysteretic measured closures, retaining interrupted and boundary events.

    These are review proposals, not a replacement for a calibrated selector.
    Requiring a preceding opening avoids counting every closed feedback row.
    """
    times, openness = np.asarray(times), np.asarray(openness)
    valid, epochs, ticks = np.asarray(valid), np.asarray(epochs), np.asarray(ticks)
    if not (times.shape == openness.shape == valid.shape == epochs.shape == ticks.shape):
        raise ValueError("aligned trace arrays required")
    n = len(times)
    events, active, peak, peak_row, armed = [], None, None, None, False
    minimum = 1.0
    for row in range(n):
        connected = row == 0 or (epochs[row] == epochs[row - 1] and ticks[row] == ticks[row - 1] + 1
                                and 0 < times[row] - times[row - 1] <= 0.15)
        if not valid[row] or not connected:
            if active is not None:
                events[active].update(end=max(events[active]["close"], row - 1), end_reason="trace_gap")
                active = None
            peak, peak_row, armed = None, None, False
            if not valid[row]:
                continue
        value = float(openness[row])
        if active is not None:
            minimum = min(minimum, value)
            if value >= 0.70 and value - minimum >= 0.18 and times[row] - times[events[active]["close"]] >= 0.20:
                events[active].update(end=row, end_reason="reopened")
                active = None
                peak, peak_row, armed = value, row, True
            elif times[row] - times[events[active]["close"]] >= 8.0:
                events[active].update(end=row, end_reason="review_window_limit")
                active = None
                peak, peak_row, armed = value, row, value >= 0.65
            continue
        if peak is None or value >= peak or times[row] - times[peak_row] > 0.8:
            peak, peak_row = value, row
        armed |= value >= 0.65
        if armed and value <= 0.62 and peak - value >= 0.18:
            events.append({"entry": peak_row, "close": row, "end": n - 1, "end_reason": "episode_end"})
            active = len(events) - 1
            minimum, armed = value, False
    return events


def longest_run(samples, predicate, max_gap=0.25):
    """Duration using distinct, advancing capture evidence, never query count."""
    best, start, previous, seen = 0.0, None, None, set()
    for sample in samples:
        key, time_s = sample["frame_key"], sample["visual_time"]
        if time_s is None or not np.isfinite(time_s):
            start = None
            continue
        if key in seen:
            continue
        seen.add(key)
        if previous is not None and time_s <= previous:
            start = None
            continue
        if previous is not None and time_s - previous > max_gap:
            start = None
        if predicate(sample):
            start = time_s if start is None else start
            best = max(best, time_s - start)
        else:
            start = None
        previous = time_s
    return float(best)


def suggest_outcome(candidate, predictions):
    """Suggest success/failure/uncertain; never create an approved RL reward."""
    samples = []
    previous_object = None
    for original in candidate["samples"]:
        sample = dict(original)
        vision = predictions.get(sample["frame_key"], {}).get("geometry", {})
        age = sample["time"] - sample["visual_time"] if sample["visual_time"] is not None else np.inf
        usable = sample["valid"] and 0 <= age <= 0.25 and sample["base_z_m"] is not None
        delta = sample["base_z_m"] - candidate["reference_base_z_m"] if usable else None
        held = usable and bool(vision.get("holding_candidate")) and sample["openness"] < 0.70
        objects = vision.get("held_objects", [])
        stable = True
        if held and objects:
            obj = max(objects, key=lambda x: x["score"])
            if previous_object is not None:
                distance = np.linalg.norm(np.asarray(obj["centroid"]) - previous_object["centroid"])
                ratio = obj["area"] / max(1, previous_object["area"])
                stable = distance <= 45 and 0.5 <= ratio <= 2
            previous_object = obj
        else:
            previous_object = None
        sample.update(lift_delta_m=delta,
                      held_lifted=bool(held and stable and delta >= 0.05),
                      empty_lifted=bool(usable and vision.get("empty_candidate")
                                        and sample["openness"] < 0.60 and delta >= 0.05),
                      vision=vision)
        samples.append(sample)
    held_s = longest_run(samples, lambda s: s["held_lifted"])
    empty_s = longest_run(samples, lambda s: s["empty_lifted"])
    reasons = []
    if held_s >= 1.0:
        label = "success"
        reasons.append("夹爪间持物候选稳定；相对抓取参考抬升≥5cm且新鲜视觉连续≥1秒")
    elif empty_s >= 0.4 and not any(s["held_lifted"] for s in samples):
        label = "failure"
        reasons.append("抬升≥5cm后，连续多帧显示清晰指间空隙、夹爪正检及无持物候选，待人工核实")
    else:
        label = "uncertain"
        if any(s["held_lifted"] for s in samples):
            reasons.append("有抬起持物迹象，但连续1秒条件未得到充分证实；可能快速放置或遮挡")
        else:
            reasons.append("抓取或空手证据不足；零检测不作为失败")
    if candidate["end_reason"] != "reopened":
        reasons.append("片段在" + {"trace_gap": "观测或时间轴断裂", "episode_end": "录像终点",
                                   "review_window_limit": "候选窗口上限"}.get(candidate["end_reason"], candidate["end_reason"]) + "处结束")
    reasons.append("使用各臂base-Z的暂定向上轴及候选进入行参考；尚无桌面标定，全部待审核")
    maximum = max((s["lift_delta_m"] for s in samples if s["lift_delta_m"] is not None), default=None)
    return {**candidate, "suggested_label": label, "review_status": "unreviewed", "review_label": None,
            "approved_reward": None, "reasons": reasons, "max_lift_delta_m": maximum,
            "visual_hold_s": held_s, "clear_empty_s": empty_s, "samples": samples}
