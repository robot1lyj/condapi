"""Task-level efficiency, kept separate from local grasp rewards and Q loss."""

import math


def episode_metrics(*, initial_objects, completed_objects, attempts, duration_s):
    """A shorter failed rollout must never beat a completed one by stopping early.

    attempts must be reviewed physical attempts, not model chunks/windows. C/A
    measures end-to-end correct completions per attempt, NOT grasp success rate.
    Missing counts remain unknown; no inference from fixed initial object count.
    """
    n, c, a = initial_objects, completed_objects, attempts
    if not isinstance(n, int) or n <= 0:
        raise ValueError("positive integer initial object count required")
    if c is not None and (not isinstance(c, int) or not 0 <= c <= n):
        raise ValueError("invalid completed object count")
    if a is not None and (not isinstance(a, int) or a < 0):
        raise ValueError("invalid physical attempt count")
    if not math.isfinite(duration_s) or duration_s < 0:
        raise ValueError("invalid rollout duration")
    if c and a == 0:
        raise ValueError("positive completion with zero grasp attempts")
    full = None if c is None else c == n
    return {"initial_objects": n, "completed_objects": c, "physical_attempts": a,
            "duration_s": duration_s, "completion_fraction": None if c is None else c/n,
            "full_completion": full,
            "attempts_if_complete": a if full else None,
            "attempts_per_correct_completion": a/c if a is not None and c else None,
            "correct_completions_per_attempt": c/a if c is not None and a else None,
            "completion_weighted_attempt_efficiency": (
                None if full is None or a is None else min(1., n/a) if full else 0.),
            "correct_completions_per_minute": c*60/duration_s if c is not None and duration_s else None}
