"""Read-only per-episode numeric distribution audit for published YAM data.

This deliberately operates on the published LeRobot v3 Parquet layout.  It
does not construct a trainer, read video, or write inside the dataset; the
caller chooses a new output path for the JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np
import pyarrow.parquet as pq


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_column(table, name: str, rows: int) -> np.ndarray:
    values = np.asarray(table[name].combine_chunks().values, dtype=np.float32)
    return values.reshape(rows, 14)


def episode_path(root: pathlib.Path, episode: int) -> pathlib.Path:
    chunk, file_id = divmod(episode, 1000)
    return root / f"data/chunk-{chunk:03d}/file-{file_id:03d}.parquet"


def extrema(values: np.ndarray) -> tuple[float, int, int]:
    if values.size == 0:
        return 0.0, 0, 0
    flat = np.abs(values)
    flat_index = int(np.argmax(flat))
    row, dim = np.unravel_index(flat_index, values.shape)
    return float(flat[row, dim]), int(row), int(dim)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "q99": float(np.quantile(array, 0.99)),
        "q999": float(np.quantile(array, 0.999)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--selection", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()

    manifest_path = args.dataset / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selected = set(json.loads(args.selection.read_text()))
    rows: list[dict] = []
    state_jumps: list[float] = []
    action_jumps: list[float] = []
    action_state: list[float] = []
    for item in manifest["episodes"]:
        episode = int(item["episode_index"])
        length = int(item["length"])
        table = pq.read_table(
            episode_path(args.dataset, episode),
            columns=["observation.state", "action", "frame_index", "timestamp", "episode_index"],
        )
        state = array_column(table, "observation.state", length)
        action = array_column(table, "action", length)
        frame_index = np.asarray(table["frame_index"])
        episode_index = np.asarray(table["episode_index"])
        timestamp = np.asarray(table["timestamp"], dtype=np.float64)
        state_diff = np.diff(state, axis=0)
        action_diff = np.diff(action, axis=0)
        state_max, state_row, state_dim = extrema(state_diff)
        action_max, action_row, action_dim = extrema(action_diff)
        action_state_max, action_state_row, action_state_dim = extrema(action - state)
        state_jumps.append(state_max)
        action_jumps.append(action_max)
        action_state.append(action_state_max)
        rows.append(
            {
                "episode_index": episode,
                "selected": episode in selected,
                "length": length,
                "finite": bool(np.isfinite(state).all() and np.isfinite(action).all()),
                "frame_ok": bool(np.array_equal(frame_index, np.arange(length))),
                "episode_ok": bool(np.array_equal(episode_index, np.full(length, episode))),
                "timestamp_ok": bool(np.allclose(timestamp, np.arange(length) / 30, atol=1e-4, rtol=1e-6)),
                "state_max_abs_jump": state_max,
                "state_max_jump_frame": state_row,
                "state_max_jump_dim": state_dim,
                "action_max_abs_jump": action_max,
                "action_max_jump_frame": action_row,
                "action_max_jump_dim": action_dim,
                "action_state_max_abs": action_state_max,
                "action_state_max_frame": action_state_row,
                "action_state_max_dim": action_state_dim,
                "gripper_state_outside_frames": int(
                    ((state[:, 6] < 0) | (state[:, 6] > 1) | (state[:, 13] < 0) | (state[:, 13] > 1)).sum()
                ),
                "gripper_action_outside_frames": int(
                    ((action[:, 6] < 0) | (action[:, 6] > 1) | (action[:, 13] < 0) | (action[:, 13] > 1)).sum()
                ),
            }
        )

    selected_rows = [row for row in rows if row["selected"]]
    complement_rows = [row for row in rows if not row["selected"]]

    def group_summary(group: list[dict]) -> dict:
        return {
            "episodes": len(group),
            "frames": int(sum(row["length"] for row in group)),
            "finite_failures": int(sum(not row["finite"] for row in group)),
            "frame_failures": int(sum(not row["frame_ok"] for row in group)),
            "episode_failures": int(sum(not row["episode_ok"] for row in group)),
            "timestamp_failures": int(sum(not row["timestamp_ok"] for row in group)),
            "gripper_state_outside_frames": int(sum(row["gripper_state_outside_frames"] for row in group)),
            "gripper_action_outside_frames": int(sum(row["gripper_action_outside_frames"] for row in group)),
            "state_max_abs_jump": summarize([row["state_max_abs_jump"] for row in group]),
            "action_max_abs_jump": summarize([row["action_max_abs_jump"] for row in group]),
            "action_state_max_abs": summarize([row["action_state_max_abs"] for row in group]),
        }

    payload = {
        "dataset": str(args.dataset),
        "dataset_manifest_sha256": sha256(manifest_path),
        "selection": str(args.selection),
        "selection_count": len(selected),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "all": group_summary(rows),
        "selected": group_summary(selected_rows),
        "complement": group_summary(complement_rows),
        "top_state_jumps": sorted(rows, key=lambda row: row["state_max_abs_jump"], reverse=True)[:20],
        "top_action_jumps": sorted(rows, key=lambda row: row["action_max_abs_jump"], reverse=True)[:20],
        "top_action_state": sorted(rows, key=lambda row: row["action_state_max_abs"], reverse=True)[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "episodes": len(rows), "elapsed_seconds": payload["elapsed_seconds"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
