"""Read-only YAM HIL audit and physical-action replay preparation (no training).

Run with the existing YAM data environment. Raw files are never rewritten.
Rewards are episode sidecars, not rewards for pressing the intervention key.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import av
import h5py
import numpy as np
from PIL import Image
from PIL import ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/residual-rl/src"))
from yam_residual_rl.contracts import progress_reward


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def transitions(valid, ticks, epochs, times):
    """Never stitch across a bad observation, missing tick, or clock/epoch reset."""
    return np.flatnonzero(
        valid[:-1] & valid[1:] & (np.diff(ticks) == 1) & (np.diff(epochs) == 0) & (np.diff(times) > 0)
    )


def episode_audit(manifest_path, destination, reward, label_override=None):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("error") or manifest.get("mock"):
        raise ValueError(f"not a closed real-robot episode: {manifest_path}")
    episode_id = manifest["episode_id"]
    arrays, details, videos, pictures = {}, [], [], {}
    offset = 0
    for segment in manifest["segments"]:
        if segment["state"] != "committed" or segment["start_frame"] != offset:
            raise ValueError(f"uncommitted/noncontiguous segment: {manifest_path}")
        folder = manifest_path.parent / segment["path"]
        with h5py.File(folder / "samples.h5", "r") as handle:
            size = int(handle["committed_rows"][()])
            if size != segment["steps"]:
                raise ValueError("manifest/HDF5 committed row mismatch")
            for key in handle:
                if key not in ("details", "committed_rows"):
                    arrays.setdefault(key, []).append(handle[key][:size])
            details.extend(json.loads(row) for row in handle["details"][:size])
            references = handle["video_indices"][:size]
        for camera_index, role in enumerate(("top", "left", "right")):
            path = folder / f"{role}.mp4"
            pts, shapes = [], Counter()
            picks = {0, size // 2, max(0, size - 15), size - 1}
            with av.open(str(path)) as container:
                container.streams.video[0].codec_context.thread_count = 1
                for index, frame in enumerate(container.decode(video=0)):
                    pts.append(float(frame.pts * frame.time_base))
                    shapes[(frame.width, frame.height)] += 1
                    if index in picks:
                        pictures[(role, offset + index)] = frame.to_image().resize((256, 192))
            if len(pts) != segment["video_frames"][role]:
                raise ValueError(f"video decode count mismatch: {path}")
            if np.any(np.diff(pts) <= 0):
                raise ValueError(f"nonmonotonic video timestamps: {path}")
            if np.any(references[:, camera_index] < 0) or np.any(references[:, camera_index] >= len(pts)):
                raise ValueError(f"video reference out of range: {path}")
            videos.append(
                {
                    "path": str(path.relative_to(manifest_path.parent)),
                    "frames": len(pts),
                    "dimensions": {f"{w}x{h}": n for (w, h), n in shapes.items()},
                    "sequential_references": bool(np.array_equal(references[:, camera_index], np.arange(size))),
                }
            )
        offset += size
    a = {key: np.concatenate(parts) for key, parts in arrays.items()}
    n = len(details)
    if n != manifest["steps"] or n < 2:
        raise ValueError("episode length mismatch/too short")
    action, state = a["submitted_action"], a["observation_state"]
    if action.shape != (n, 14) or state.shape != (n, 14):
        raise ValueError("expected YAM 14D physical actions and states")
    source = np.array([row.get("source", "unknown") for row in details])
    finite = np.isfinite(action).all(1) & np.isfinite(state).all(1) & np.isfinite(a["time"])
    valid = (
        finite
        & a["observation_valid"]
        & a["observation_state__valid"]
        & a["submitted_action__valid"]
        & np.isin(source, ["human", "policy", "hold"])
        & a["tick__valid"]
        & a["epoch__valid"]
        & a["time__valid"]
    )
    # Unknown/slow HIL frames remain on the real timeline, but are not BC targets.
    expert = valid & (source == "human") & a["expert_valid"]
    for row_index, row in enumerate(details):
        if row.get("wait_boundary"):
            valid[row_index] = False
    expert &= valid
    idx = transitions(valid, a["tick"], a["epoch"], a["time"])
    terminal = idx + 1 == n - 1
    terminal_observed = bool(terminal.any())
    # A rescued trajectory's earlier return is not evidence of autonomous
    # completion. Keep it as off-policy data, with calibration provenance.
    future_human = np.cumsum((source == "human")[::-1])[::-1] > 0
    label = {
        "schema": "yam_episode_reward_v1",
        "episode_id": episode_id,
        "reward": reward,
        "reward_kind": "whole_episode_progress",
        "initial_remaining": None,
        "newly_correct": None,
        "assisted": bool((source == "human").any()),
        "label_source": "user_completed_saved_hil_success_convention" if reward is not None else "unlabeled",
        "terminal_observation_frame": n - 1,
        "terminal_observed": terminal_observed,
        "note": "R=1 labels the completed trajectory; it does not label every action optimal.",
    }
    if label_override:
        if label_override["assisted"] != label["assisted"]:
            raise ValueError("operator assisted flag disagrees with recorded human control")
        label.update(label_override)
        label["terminal_observed"] = terminal_observed
    rewards = np.zeros(len(idx), dtype=np.float32)
    if reward is not None:
        rewards[terminal] = reward
    np.savez_compressed(
        destination / f"{episode_id}.npz",
        observation=state[idx].astype(np.float32),
        next_observation=state[idx + 1].astype(np.float32),
        action=action[idx].astype(np.float32),
        frame_index=idx,
        next_frame_index=idx + 1,
        tick=a["tick"][idx],
        time=a["time"][idx],
        next_time=a["time"][idx + 1],
        remaining_seconds=(a["time"][-1] - a["time"][idx]).astype(np.float64),
        dt=(a["time"][idx + 1] - a["time"][idx]).astype(np.float64),
        reward=rewards,
        terminated=terminal,
        reward_valid=np.full(len(idx), reward is not None and terminal_observed),
        calibration_valid=(~future_human[idx]) & (reward is not None) & terminal_observed,
        action_source=np.array([{"human": 0, "policy": 1, "hold": 2}[v] for v in source[idx]]),
        expert_valid=expert[idx],
    )
    # Check actual policy rows against the original reply, without inventing base
    # actions for human rows or calling measured next state an action.
    replies, matched, missing, mismatches = {}, 0, 0, 0
    for i, row in enumerate(details):
        reply = row.get("policy_reply")
        if reply and not reply.get("error") and reply.get("actions"):
            token = reply["token"]
            replies[(token["epoch"], token["request_id"])] = reply["actions"]
        if source[i] != "policy":
            continue
        selection = row.get("policy_selection") or {}
        token = selection.get("request") or {}
        reply_actions = replies.get((token.get("epoch"), token.get("request_id")))
        model_index = selection.get("model_index")
        if reply_actions is None or not isinstance(model_index, int) or not 0 <= model_index < len(reply_actions):
            missing += 1
        else:
            matched += 1
            if not np.allclose(a["policy_action"][i], reply_actions[model_index], atol=1e-7, rtol=0):
                mismatches += 1
            if selection.get("target_tick") != int(a["tick"][i]):
                mismatches += 1
    picture_ticks = sorted({i for _, i in pictures})
    sheet = Image.new("RGB", (3 * 256, len(picture_ticks) * 216), "#202020")
    draw = ImageDraw.Draw(sheet)
    for y, i in enumerate(picture_ticks):
        for x, role in enumerate(("top", "left", "right")):
            if (role, i) in pictures:
                sheet.paste(pictures[(role, i)], (x * 256, y * 216))
                draw.text((x * 256 + 4, y * 216 + 194), f"{role} frame={i}", fill="white")
    sheet.save(destination / f"{episode_id}.jpg", quality=85)
    human_ids = {int(v) for v in a["intervention_id"][source == "human"] if v >= 0}
    commands = {
        row.get("human_input", {}).get("command", {}).get("command_id")
        for row in details
        if row.get("human_input", {}).get("command")
    }
    action_finite = action[np.isfinite(action).all(1)]
    return {
        "episode_id": episode_id,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "frames": n,
        "sources": dict(Counter(source.tolist())),
        "interventions": len(human_ids),
        "keyboard_commands": len(commands),
        "expert_frames": int(expert.sum()),
        "invalid_observation_frames": np.flatnonzero(~valid).tolist(),
        "tick_gaps": int((np.diff(a["tick"]) != 1).sum()),
        "epoch_changes": int((np.diff(a["epoch"]) != 0).sum()),
        "max_step_seconds": float(np.diff(a["time"]).max()),
        "valid_transitions": len(idx),
        "reward_label": label,
        "autonomous_tail_calibration_transitions": int(((~future_human[idx]) & terminal_observed).sum()),
        "action_min": action_finite.min(0).tolist(),
        "action_max": action_finite.max(0).tolist(),
        "policy_reply_check": {"matched": matched, "missing": missing, "mismatches": mismatches},
        "videos": videos,
        "prepared_for": "physical_action_critic_initialization",
        "actor_replay_ready": False,
        "pending": [
            "frozen visual features",
            "versioned base policy actions at the same observations",
            "online decision/RTC queue contract; raw 30Hz pairs are not H50 decisions",
        ],
    }


def audit(raw, snapshot_path, output, *, completed_successes=False, labels_path=None):
    if output.exists() or output.resolve().is_relative_to(raw.resolve()):
        raise ValueError("audit output must be a new independent directory")
    snapshot = json.loads(snapshot_path.read_text())
    labels = {}
    if labels_path:
        for row in json.loads(labels_path.read_text()):
            if row["schema"] != "yam_rollout_label_v1" or row["episode_id"] in labels:
                raise ValueError("expected validated unique rollout labels")
            if row["reward"] is not None:
                expected = progress_reward(
                    row["initial_remaining"], row["newly_correct"], row["wrong"], row["unplaced"]
                )
                if row["reward"] != expected:
                    raise ValueError("reward/count mismatch")
            labels[row["episode_id"]] = row
    for relative, expected in snapshot["files"].items():
        path = raw / relative
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"source checksum mismatch: {relative}")
    output.mkdir(parents=True)
    report = {
        "schema": "yam_hil_rl_audit_v1",
        "source_snapshot_sha256": sha256(snapshot_path),
        "hashes_verified": len(snapshot["files"]),
        "episodes": [],
        "failures": [],
    }
    for path in sorted(raw.rglob("manifest.json")):
        try:
            episode_id = json.loads(path.read_text())["episode_id"]
            if labels_path and episode_id not in labels:
                raise ValueError("missing per-episode result label")
            label = labels.get(episode_id)
            reward = label["reward"] if label else (1.0 if completed_successes else None)
            result = episode_audit(path, output, reward, label)
            report["episodes"].append(result)
            print(
                json.dumps(
                    {
                        "episode": result["episode_id"],
                        "frames": result["frames"],
                        "transitions": result["valid_transitions"],
                    }
                ),
                flush=True,
            )
        except Exception as error:
            report["failures"].append({"manifest": str(path), "error": f"{type(error).__name__}: {error}"})
    # Episode-level split, never mix neighboring frames into train and validation.
    episodes = sorted(report["episodes"], key=lambda row: hashlib.sha256(row["episode_id"].encode()).hexdigest())
    holdout_count = max(1, round(len(episodes) * 0.2)) if len(episodes) > 1 else 0
    for i, episode in enumerate(episodes):
        episode["split"] = "holdout" if i < holdout_count else "train"
    report["totals"] = {
        key: sum(row[key] for row in episodes)
        for key in ("frames", "valid_transitions", "expert_frames", "interventions", "keyboard_commands")
    }
    report["training_ready"] = False
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "rewards.json").write_text(
        json.dumps([row["reward_label"] for row in episodes], ensure_ascii=False, indent=2) + "\n"
    )
    if report["failures"]:
        raise ValueError(f"{len(report['failures'])} episodes failed audit; see audit.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    labels = parser.add_mutually_exclusive_group()
    labels.add_argument("--labels", type=Path, help="Output of label_rollouts.py with per-episode final counts")
    labels.add_argument(
        "--completed-successes",
        action="store_true",
        help="Only use with the operator's completed saved episode success convention",
    )
    args = parser.parse_args()
    audit(args.raw, args.snapshot, args.output, completed_successes=args.completed_successes, labels_path=args.labels)


if __name__ == "__main__":
    main()
