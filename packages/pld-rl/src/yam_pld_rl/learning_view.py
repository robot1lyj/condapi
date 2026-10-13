"""CPU-only learning views of verified RTC replay, preserving discounted returns."""

from itertools import pairwise

import numpy as np


def learning_view(source, *, collapse_wait=True):
    """Collapse only mandatory zero-control suffixes within one complete attempt.

    The resulting last decision waits until the original verified reward endpoint.
    Its reward is the discounted sum over that exact interval, never an earlier
    success annotation. Original raw rows and replay remain immutable.
    """
    required = {"x", "nx", "action", "scale", "negative_scale", "next_scale", "next_negative_scale",
                "reward", "terminal", "discount", "dt", "attempt", "episode", "decision", "train", "arm"}
    if required - source.keys():
        raise ValueError("missing verified replay fields")
    n = len(source["reward"])
    if any(len(v) != n for v in source.values()):
        raise ValueError("unaligned replay arrays")
    control = (source["scale"] + source["negative_scale"]).sum(-1) > 0
    if not n:
        raise ValueError("empty replay")
    output, audit = [], []
    seen = set()
    for i in range(n):
        aid = str(source["attempt"][i])
        if aid in seen:
            continue
        seen.add(aid)
        ids = np.flatnonzero(source["attempt"] == aid)
        if not np.array_equal(ids, np.arange(ids[0], ids[-1]+1)):
            raise ValueError("attempt is not contiguous")
        if (not np.array_equal(source["terminal"][ids], [0.]*(len(ids)-1)+[1.])
                or len(np.unique(source["episode"][ids])) != 1
                or len(np.unique(source["train"][ids])) != 1
                or len(np.unique(source["arm"][ids])) != 1):
            raise ValueError("attempt has invalid terminal/split/identity")
        if np.any(source["reward"][ids[:-1]] != 0) or source["reward"][ids[-1]] not in (0, 1):
            raise ValueError("expected immutable terminal binary local reward")
        if np.any(source["dt"][ids] <= 0) or np.any((source["discount"][ids] <= 0) | (source["discount"][ids] > 1)):
            raise ValueError("invalid SMDP time/discount")
        for j, k in pairwise(ids):
            if not np.array_equal(source["nx"][j], source["x"][k]):
                raise ValueError("replay next-state chain broken")
        editable = ids[control[ids]]
        if not len(editable):
            raise ValueError("attempt has no editable action")
        cutoff = int(editable[-1]) if collapse_wait else int(ids[-1])
        tail = ids[ids > cutoff]
        if np.any(source["action"][tail] != 0) or np.any(control[tail]):
            raise ValueError("cannot merge an active control decision")
        spans = [[int(j)] for j in ids if j < cutoff] + [list(map(int, ids[ids >= cutoff]))]
        local = []
        for span in spans:
            first, last = span[0], span[-1]
            record = {k: np.array(v[first], copy=True) for k, v in source.items()}
            reward, discount = 0., 1.
            for j in span:
                reward += discount * float(source["reward"][j])
                discount *= float(source["discount"][j])
            record.update(reward=reward, discount=discount, dt=float(source["dt"][span].sum()),
                          terminal=float(source["terminal"][last]), nx=source["nx"][last].copy(),
                          next_scale=source["next_scale"][last].copy(),
                          next_negative_scale=source["next_negative_scale"][last].copy(),
                          attempt_outcome=float(source["reward"][ids[-1]]),
                          source_first=first, source_last=last, source_count=len(span))
            local.append(record)
        full_return = 0.
        for j in ids[::-1]:
            full_return = float(source["reward"][j]) + float(source["discount"][j]) * full_return
        mc = 0.
        for record in reversed(local):
            mc = record["reward"] + record["discount"] * mc
            record["mc_return"] = mc
        if not np.isclose(mc, full_return, atol=1e-10, rtol=0):
            raise ValueError("wait collapse changed the attempt return")
        for j, record in enumerate(local):
            record["is_entry"] = j == 0
            record["next_action"] = local[j+1]["action"].copy() if j+1 < len(local) else np.zeros_like(record["action"])
        output.extend(local)
        audit.append({"attempt": aid, "native_intervals": len(ids), "learning_intervals": len(local),
                      "merged_wait_intervals": len(tail), "wait_seconds": float(source["dt"][tail].sum()),
                      "return_preserved": True, "outcome": float(source["reward"][ids[-1]])})
    arrays = {k: np.asarray([r[k] for r in output]) for k in output[0]}
    return arrays, {"native_transitions": n, "learning_transitions": len(output), "attempts": audit,
                    "collapse_wait": collapse_wait, "reward_endpoint_unchanged": True}
