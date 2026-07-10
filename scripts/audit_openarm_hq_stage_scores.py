"""Audit complete HQ-Stage score shards before they can enter the formal KAI0 dataset."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.openarm_kai0_contract import HQ_FOLDING_ONLY_START
except ImportError:
    from openarm_kai0_contract import HQ_FOLDING_ONLY_START


SCORE_COLUMNS = ("relative_advantage", "absolute_value", "absolute_advantage")


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def _data_path(info: dict[str, Any], episode_index: int) -> pathlib.Path:
    chunk = episode_index // int(info["chunks_size"])
    return pathlib.Path(info["data_path"].format(episode_chunk=chunk, episode_index=episode_index))


def _write_json_atomic(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _percentiles(values: np.ndarray) -> dict[str, float]:
    points = np.percentile(values, [0, 1, 5, 10, 50, 90, 95, 99, 100])
    return {
        key: float(value)
        for key, value in zip(("min", "p01", "p05", "p10", "median", "p90", "p95", "p99", "max"), points, strict=True)
    }


def audit_curves(
    curves: list[tuple[int, np.ndarray, np.ndarray]],
    *,
    expected_episodes: int,
    folding_only_start: int,
) -> dict[str, Any]:
    actual_ids = {episode for episode, _, _ in curves}
    expected_ids = set(range(expected_episodes))
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    duplicate_count = len(curves) - len(actual_ids)
    finite = all(np.isfinite(value).all() and np.isfinite(relative).all() for _, value, relative in curves)
    in_range = all(
        np.all((value >= -1.0001) & (value <= 1.0001)) and np.all((relative >= -1.0001) & (relative <= 1.0001))
        for _, value, relative in curves
    )

    full = [(value, relative) for episode, value, relative in curves if episode < folding_only_start]
    folding = [(value, relative) for episode, value, relative in curves if episode >= folding_only_start]
    if not full or not folding:
        raise ValueError("Both full-task and folding-only HQ groups must contain episodes")
    full_peaks = np.asarray([np.percentile(value, 95) for value, _ in full], dtype=np.float32)
    folding_peaks = np.asarray([np.percentile(value, 95) for value, _ in folding], dtype=np.float32)
    relative = np.concatenate([values for _, _, values in curves]).astype(np.float32)
    gates = {
        "episode_ids_exact": not missing and not unexpected and duplicate_count == 0,
        "all_scores_finite": finite,
        "all_scores_in_range": in_range,
        "full_task_count_exact": len(full) == folding_only_start,
        "folding_only_count_exact": len(folding) == expected_episodes - folding_only_start,
        "full_task_peak_crossing_fraction>=0.95": float(np.mean(full_peaks >= 0.5)) >= 0.95,
        "full_task_peak_p10>=0.60": float(np.percentile(full_peaks, 10)) >= 0.60,
        "folding_only_peak_p10>=0.25": float(np.percentile(folding_peaks, 10)) >= 0.25,
        "relative_advantage_p90>=0.02": float(np.percentile(relative, 90)) >= 0.02,
        "relative_advantage_p10<=-0.005": float(np.percentile(relative, 10)) <= -0.005,
        "relative_near_zero_fraction<=0.60": float(np.mean(np.abs(relative) < 0.01)) <= 0.60,
    }
    return {
        "schema_version": "openarm_hq_stage_audit_v1",
        "passed": all(gates.values()),
        "expected_episodes": expected_episodes,
        "actual_episodes": len(curves),
        "folding_only_start": folding_only_start,
        "missing_episode_ids": missing,
        "unexpected_episode_ids": unexpected,
        "duplicate_count": duplicate_count,
        "full_task": {
            "episodes": len(full),
            "peak_percentiles": _percentiles(full_peaks),
            "peak_crossing_fraction": float(np.mean(full_peaks >= 0.5)),
        },
        "folding_only": {
            "episodes": len(folding),
            "episode_relative_peak_percentiles": _percentiles(folding_peaks),
            "peak_above_0.20_fraction": float(np.mean(folding_peaks >= 0.2)),
        },
        "relative_advantage": {
            "frames": len(relative),
            "percentiles": _percentiles(relative),
            "negative_fraction": float(np.mean(relative < 0)),
            "near_zero_fraction": float(np.mean(np.abs(relative) < 0.01)),
        },
        "gates": gates,
    }


def audit_score_roots(
    roots: list[pathlib.Path],
    *,
    expected_episodes: int,
    folding_only_start: int,
) -> dict[str, Any]:
    curves = []
    seen = set()
    for root_path in roots:
        root = root_path.resolve()
        info = _load_json(root / "meta/info.json")
        for row in _load_jsonl(root / "meta/episodes.jsonl"):
            local_episode = int(row["episode_index"])
            source_episode = int(row["source_episode_index"])
            if source_episode in seen:
                curves.append((source_episode, np.asarray([np.nan]), np.asarray([np.nan])))
                continue
            seen.add(source_episode)
            frame = pd.read_parquet(root / _data_path(info, local_episode), columns=list(SCORE_COLUMNS))
            value = frame["absolute_value"].to_numpy(dtype=np.float32)
            relative = frame["relative_advantage"].to_numpy(dtype=np.float32)
            if len(value) == 0:
                raise ValueError(f"Empty score episode: {root} local={local_episode}")
            curves.append((source_episode, value, relative))
    return audit_curves(curves, expected_episodes=expected_episodes, folding_only_start=folding_only_start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-root", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=999)
    parser.add_argument("--folding-only-start", type=int, default=HQ_FOLDING_ONLY_START)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    report = audit_score_roots(
        args.score_root,
        expected_episodes=args.expected_episodes,
        folding_only_start=args.folding_only_start,
    )
    _write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
