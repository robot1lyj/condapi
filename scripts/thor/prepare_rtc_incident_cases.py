"""Extract read-only RTC incident requests from a saved YAM episode for Thor replay.

The original request payload and random denoising noise are not recorded. The
returned prefix is therefore a reconstructed (float32) committed prefix, and
the replay is a same-input/same-noise backend comparison, not bitwise history.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def video_frame(path, frame_index):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index)):
            raise RuntimeError(f"Cannot seek {path} to frame {frame_index}")
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot decode {path} frame {frame_index}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--norm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must not already exist")
    manifest = json.loads((args.episode / "manifest.json").read_text())
    segment = args.episode / manifest["segments"][0]["path"]
    sample_path = segment / "samples.h5"
    source_files = {str(sample_path): sha256(sample_path)}
    for role in ("top", "left", "right"):
        path = segment / f"{role}.mp4"
        source_files[str(path)] = sha256(path)
    args.output.mkdir(parents=True)
    cases = []
    with h5py.File(sample_path) as data:
        details = [json.loads(value) for value in data["details"][:]]
        obs_rows = {row["obs_id"]: i for i, row in enumerate(details)}
        for reply_row, detail in enumerate(details):
            reply = detail.get("policy_reply")
            if not reply or not reply.get("server_timing", {}).get("rtc_used"):
                continue
            token = reply["token"]
            obs_row = obs_rows[token["observation_id"]]
            delay = int(token["rtc_delay_steps"])
            recorded = np.asarray(reply["actions"], dtype=np.float32)
            if recorded.shape != (50, 14) or not np.isfinite(recorded).all():
                raise ValueError("Invalid recorded reply")
            observation = {
                "observation.state": np.asarray(data["observation_state"][obs_row], dtype=np.float32),
                "prompt": np.asarray(manifest["task"]),
            }
            for camera_index, role in enumerate(("top", "left", "right")):
                frame_index = int(data["video_indices"][obs_row, camera_index])
                observation[f"observation.images.{role}_rgb"] = video_frame(segment / f"{role}.mp4", frame_index)
            index = len(cases)
            sample_name = f"incident-{index:03d}.npz"
            prefix_name = f"prefix-{index:03d}.npy"
            provenance_name = f"incident-{index:03d}.json"
            np.savez_compressed(args.output / sample_name, **observation)
            np.save(args.output / prefix_name, recorded[:delay], allow_pickle=False)
            provenance = {
                "source_kind": "real_yam_recording",
                "source_files": source_files,
                "source_episode": str(args.episode),
                "source_reply_row": reply_row,
                "source_observation_row": obs_row,
                "original_request_payload_recorded": False,
                "prefix_reconstruction": "float32 cast of recorded reply.actions[:delay]",
                "original_denoising_noise_recorded": False,
                "sample_sha256": sha256(args.output / sample_name),
                "norm_stats_sha256": sha256(args.norm),
            }
            (args.output / provenance_name).write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
            tick = int(token["observation_policy_tick"])
            cases.append(
                {
                    "sample": sample_name,
                    "provenance": provenance_name,
                    "source_episode": str(sample_path),
                    "committed_actions": prefix_name,
                    "committed_actions_sha256": sha256(args.output / prefix_name),
                    "delay_steps": delay,
                    "observation_policy_tick": tick,
                    "target_start_tick": tick,
                    "committed_start_tick": tick,
                    "request_id": token["request_id"],
                    "reply_row": reply_row,
                    "observation_row": obs_row,
                    "recorded_reply_actions": f"recorded-{index:03d}.npy",
                }
            )
            np.save(args.output / f"recorded-{index:03d}.npy", recorded, allow_pickle=False)
    if not cases:
        raise ValueError("No RTC replies in episode")
    (args.output / "cases.json").write_text(
        json.dumps(
            {
                "source_kind": "real_yam_recording",
                "norm_stats_sha256": sha256(args.norm),
                "cases": cases,
                "limitations": [
                    "Original request prefix and random noise were not logged; this is a reconstructed replay.",
                    "The episode manifest RTC mode may be stale; per-frame reply rtc_used is authoritative.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"cases": len(cases), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
