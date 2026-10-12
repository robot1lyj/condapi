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
