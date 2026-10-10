"""Validate per-episode final counts; no frame-level placement annotation needed.

Input JSON: [{episode_id, initial_remaining, newly_correct, wrong, unplaced,
             end_reason, assisted, base_fingerprint}, ...].
end_reason is task_end, task_budget, or interrupted. Interrupted episodes have
no training reward until an explicitly reviewed terminal observation exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/residual-rl/src"))
from yam_residual_rl.contracts import progress_reward


def validate(rows, *, base_fingerprint=None, initial_remaining=None):
    seen, results = set(), []
    fingerprints = set()
    for source_row in rows:
        row = dict(source_row)
        row.setdefault("assisted", False)
        row.setdefault("base_fingerprint", base_fingerprint)
        row.setdefault("initial_remaining", initial_remaining)
        row.setdefault("end_reason", "task_end")
        identity = row["episode_id"]
        if not identity or identity in seen:
            raise ValueError("empty/duplicate episode identity")
        seen.add(identity)
        if not isinstance(row["assisted"], bool):
            raise ValueError("explicit boolean assisted flag required")
        fingerprint = row["base_fingerprint"]
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("fixed checkpoint/norm/inference identity required")
        fingerprints.add(fingerprint)
        reason = row["end_reason"]
        if reason not in ("task_end", "task_budget", "interrupted", "pending_review"):
            raise ValueError("unknown terminal/interruption reason")
        reward = None
        if reason in ("task_end", "task_budget"):
            row.setdefault("unplaced", row["initial_remaining"] - row["newly_correct"] - row["wrong"])
            reward = progress_reward(row["initial_remaining"], row["newly_correct"], row["wrong"], row["unplaced"])
        results.append(
            {
                **row,
                "schema": "yam_rollout_label_v1",
                "reward": reward,
                "reward_kind": "whole_episode_progress",
                "reward_source": "operator_final_counts",
                "autonomous": not row["assisted"],
                "terminal_label_valid": reward is not None,
            }
        )
    if len(fingerprints) > 1:
        raise ValueError("one initialization batch must pin one base policy identity")
    if not results:
        raise ValueError("empty label batch")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-fingerprint", help="Same immutable checkpoint/norm/inference identity for this batch")
    parser.add_argument("--initial-remaining", type=int, help="Optional shared initial brick count")
    args = parser.parse_args()
    result = validate(
        json.loads(args.source.read_text()),
        base_fingerprint=args.base_fingerprint,
        initial_remaining=args.initial_remaining,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
