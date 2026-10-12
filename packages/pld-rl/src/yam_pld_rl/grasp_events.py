"""Server-side physical segmentation shared with the original video review.

Camera gaps do not erase an otherwise continuous physical attempt. A closure
event is a candidate, never a success label. NumPy only; safe for CPU imports.
"""

import numpy as np


def physical_events(times, g, z, state_valid, epochs, ticks):
    physical = state_valid & np.isfinite(g) & np.isfinite(z)
    events, active, peak, peak_row, armed, minimum = [], None, None, None, False, 1.0
    for row in range(len(times)):
        connected = row == 0 or (
            epochs[row] == epochs[row - 1] and ticks[row] == ticks[row - 1] + 1
            and 0 < times[row] - times[row - 1] <= 0.15
        )
        if not physical[row] or not connected:
            if active is not None:
                events[active].update(end=max(events[active]["close"], row - 1), end_reason="trace_gap")
            active, peak, peak_row, armed = None, None, None, False
            if not physical[row]:
                continue
        value = float(g[row])
        if active is not None:
            minimum = min(minimum, value)
            if value >= 0.65 and value - minimum >= 0.18 and times[row] - times[events[active]["close"]] >= 0.2:
                events[active].update(end=row, end_reason="reopened")
                active, peak, peak_row, armed = None, value, row, True
            continue
        if peak is None or value >= peak:
            peak, peak_row = value, row
        armed |= value >= 0.65
        if armed and value <= 0.62 and peak - value >= 0.18:
            events.append({"entry": max(peak_row, row - 15), "close": row,
                           "end": len(times) - 1, "end_reason": "episode_end"})
            active, minimum, armed = len(events) - 1, value, False
    return events


def relative_physical_events(times, g, z, state_valid, epochs, ticks):
    """Versioned offline identity: partial reopening separates repeated grasps.

    The measured low point is frozen once a 5cm rise is observed. This gives
    evidence selection a descent-aware reference, without declaring success.
    """
    events, active, peak, peak_row = [], None, None, None
    for row in range(len(times)):
        connected = row == 0 or (epochs[row] == epochs[row - 1] and ticks[row] == ticks[row - 1] + 1
                                and 0 < times[row] - times[row - 1] <= 0.15)
        valid = bool(state_valid[row]) and np.isfinite(g[row]) and np.isfinite(z[row])
        if not valid or not connected:
            if active is not None:
                active.update(end=max(active["close"], row - 1), end_reason="trace_gap")
            active, peak, peak_row = None, None, None
            if not valid:
                continue
        value = float(g[row])
        if active is not None:
            active["minimum"] = min(active["minimum"], value)
            if not active["lift_observed"]:
                if z[row] < z[active["reference"]]:
                    active["reference"] = row
                active["lift_observed"] = bool(z[row] - z[active["reference"]] >= .05)
            if value - active["minimum"] >= .12 and times[row] - times[active["close"]] >= .2:
                active.update(end=row, end_reason="reopened")
                active, peak, peak_row = None, value, row
            continue
        if peak is None or value >= peak:
            peak, peak_row = value, row
        contraction = peak - value
        if contraction >= .18 or (value <= .8 and contraction >= .10):
            active = {"entry": max(peak_row, row - 15), "close": row, "reference": row,
                      "minimum": value, "lift_observed": False,
                      "end": len(times) - 1, "end_reason": "episode_end"}
            events.append(active)
    return events
