"""Build recorded RTC prefix cases from the same YAM episode rows as observations.

This is an offline model test fixture. Its frame_index is a dataset tick, not
a claim about the physical camera-to-controller timing on the live IPC.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from prepare_checkpoint_suite import digest
from prepare_checkpoint_suite import prepare
from rtc_norm_identity import checkpoint_norm_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-suite", type=Path, required=True)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_suite = json.loads((args.source_suite / "suite.json").read_text())
    source_key = "train/data/source-file-000.parquet"
    if (
        source_suite.get("source_kind") != "real_yam_recording"
        or source_suite.get("source_files", {}).get(source_key) != digest(args.source_parquet)
    ):
        raise ValueError("Source Parquet is not the observation suite's recorded episode data")
    contract = json.loads((args.checkpoint / "training_contract.json").read_text())
    max_delay = contract["model"]["rtc_training_max_delay"]
    norm = args.checkpoint / "assets/yam/norm_stats.json"
    if not 0 < max_delay < 50:
        raise ValueError("RTC training delay does not match the checkpoint")
    norm_identity = checkpoint_norm_identity(norm, contract["norm_sha256"])
    # Prepare also copies the real observation files and rebinds provenance to
    # this checkpoint's norm without altering RGB/state arrays.
    prepare(args.source_suite, args.checkpoint, args.output)
    suite = json.loads((args.output / "suite.json").read_text())
    if len(suite["samples"]) < 3:
        raise ValueError("Need at least three recorded RTC cases")
    cases = []
    by_episode = {}
    for index, entry in enumerate(suite["samples"]):
        provenance = json.loads((args.output / entry["provenance"]).read_text())
        episode = provenance["episode"]["source_episode_index"]
        frame = provenance["frame_index"]
        if episode not in by_episode:
            table = pq.read_table(args.source_parquet, filters=[("episode_index", "=", episode)])
            table = table.sort_by([("frame_index", "ascending")])
            frames = np.asarray(table["frame_index"], dtype=np.int64)
            states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            timestamps = np.asarray(table["timestamp"], dtype=np.float64)
            if not np.array_equal(frames, np.arange(len(frames))) or states.shape[1:] != (14,) or actions.shape[1:] != (14,):
                raise ValueError("RTC source episode has invalid frame/state/action layout")
            by_episode[episode] = states, actions, timestamps
        states, actions, timestamps = by_episode[episode]
        delay = (0, 1, max_delay)[index % 3]
        if frame + 50 > len(actions):
            raise ValueError("RTC case is too near episode end for a real H50 target")
        with np.load(args.output / entry["sample"], allow_pickle=False) as sample:
            state = np.asarray(sample["observation.state"], dtype=np.float32)
        if not np.array_equal(state, states[frame]):
            raise ValueError("RTC observation state is not the selected episode/frame row")
        if not np.isclose(timestamps[frame + 1] - timestamps[frame], 1 / 30, atol=1e-5):
            raise ValueError("RTC source action ticks are not 30 Hz")
        prefix = actions[frame : frame + delay].copy()
        path = args.output / f"rtc-prefix-{index:03d}.npy"
        np.save(path, prefix, allow_pickle=False)
        cases.append({
            "sample": entry["sample"],
            "provenance": entry["provenance"],
            "source_episode": source_key,
            "committed_actions": path.name,
            "committed_actions_sha256": digest(path),
            "delay_steps": delay,
            "observation_policy_tick": int(frame),
            "target_start_tick": int(frame),
            "committed_start_tick": int(frame),
            "observation_dataset_timestamp_s": float(timestamps[frame]),
            "action_0_dataset_timestamp_s": float(timestamps[frame]),
            "timing_scope": "offline LeRobot row index; not live camera/control offset",
        })
    (args.output / "cases.json").write_text(json.dumps({
        "source_kind": "real_yam_recording",
        "norm_stats_sha256": digest(norm),
        "source_parquet_sha256": digest(args.source_parquet),
        "norm_identity": norm_identity,
        "cases": cases,
        "timing_scope": "same LeRobot row t has observation/state/action[0]; physical IPC offset unknown",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
