"""Publish reviewed grasp rewards and physical single-step behavior replay.

CPU data processing only. No model forward or training on the workstation.
This is offline value initialization; not an accepted online selector or RTC policy.
"""

import argparse
from collections import Counter
from collections import defaultdict
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src"))
from yam_pld_rl.data import action_coordinates
from yam_pld_rl.data import fit_normalization
from yam_pld_rl.data import observed_edges
from yam_pld_rl.data import physical_edges
from yam_pld_rl.data import sha256
from yam_pld_rl.data import suffix_return
from yam_pld_rl.data import write_json

KEYS = [
    "time",
    "tick",
    "epoch",
    "observation_state",
    "observation_state__valid",
    "observation_valid",
    "submitted_action",
    "submitted_action__valid",
    "policy_action",
    "policy_action__valid",
    "video_indices",
    "camera_host_received_at",
    "camera_host_received_at__valid",
    "tick__valid",
    "epoch__valid",
    "time__valid",
]


def read_episode(root, label):
    folder = root / "raw" / label["source_path"]
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("mock") or manifest.get("error") or manifest["episode_id"] != label["episode_id"]:
        raise ValueError("raw manifest identity/error")
    arrays, details, segments, hashes = defaultdict(list), [], [], {}
    offset = 0
    for seg in manifest["segments"]:
        if seg["state"] != "committed" or seg["start_frame"] != offset:
            raise ValueError("uncommitted/noncontiguous storage segment")
        path = folder / seg["path"] / "samples.h5"
        hashes[str(path.relative_to(root))] = sha256(path)
        with h5py.File(path) as h:
            n = int(h["committed_rows"][()])
            if n != seg["steps"]:
                raise ValueError("committed row count")
            for key in KEYS:
                arrays[key].append(h[key][:n])
            details.extend(json.loads(x) for x in h["details"][:n])
            segments.extend([seg["path"]] * n)
            offset += n
    a = {k: np.concatenate(v) for k, v in arrays.items()}
    a["segments"] = segments
    a["details"] = details
    if offset != manifest["steps"]:
        raise ValueError("episode frame count")
    a["source"] = np.array([d.get("source") for d in details])
    a["base_target"] = np.array(
        [(d.get("policy_selection") or {}).get("parts", {}).get("base_target", [np.nan] * 14) for d in details],
        dtype=np.float64,
    )
    # Prove zero residual from recorded runtime values, not rerunning Pi today.
    base_matches = np.isclose(a["base_target"], a["submitted_action"], atol=1e-7, rtol=0).all(1)
    a["physical_valid"] = (
        a["submitted_action__valid"]
        & a["tick__valid"]
        & a["epoch__valid"]
        & a["time__valid"]
        & np.isfinite(a["submitted_action"]).all(1)
        & (a["source"] == "policy")
        & base_matches
        & np.array([not d.get("wait_boundary") for d in details])
    )
    capture_age = a["time"][:, None] - a["camera_host_received_at"]
    fresh = (
        (capture_age >= 0) & (capture_age <= 0.3) & a["camera_host_received_at__valid"] & (a["video_indices"] >= 0)
    ).all(1)
    a["observed_valid"] = (
        a["observation_valid"]
        & a["observation_state__valid"]
        & fresh
        & np.isfinite(a["observation_state"]).all(1)
        & a["physical_valid"]
    )
    a["physical_edges"] = physical_edges(a["time"], a["tick"], a["epoch"], a["physical_valid"])
    a["observed_edges"] = observed_edges(a["physical_edges"], a["observed_valid"])
    return a, hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-bundle", type=Path, required=True)
    p.add_argument("--classification", type=Path, required=True)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--discount-per-second", type=float, default=0.99)
    p.add_argument("--preclose-context-seconds", type=float, default=2.0)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    summary_path = args.qualification / "summary.json"
    qualification = json.loads(summary_path.read_text())
    candidate_path = args.classification / "candidates.json"
    if qualification["source_candidates_sha256"] != sha256(candidate_path):
        raise ValueError("qualification/candidate fingerprint differs")
    candidates = {c["candidate_id"]: c for c in json.loads(candidate_path.read_text())}
    proposals = json.loads((args.qualification / "lift-qualified-proposals.json").read_text())
    labels = json.loads((args.raw_bundle / "operator_labels_v2.json").read_text())
    approved = []
    for q in proposals:
        r = dict(q)
        r["approved_reward"] = q["proposed_reward"] if q["evidence_eligible"] else None
        r["approval_basis"] = "2026-10-12 user accepts reviewed rewards and explicitly starts offline experiments"
        approved.append(r)
    write_json(args.output / "rewards.json", approved)
    write_json(args.output / "global-rewards.json", labels)
    observations, observation_index, frame_jobs = [], {}, {}
    storage, attempts, hashes = defaultdict(list), [], {}
    excluded = Counter()

    for label in labels:
        ep = label["operator_row"]
        a, hs = read_episode(args.raw_bundle, label)
        hashes.update(hs)
        n = len(a["time"])

        def observation(row, ep=ep, a=a, label=label):
            key = (ep, row)
            if key not in observation_index:
                keys = []
                for col, role in enumerate(("top", "left", "right")):
                    frame = int(a["video_indices"][row, col])
                    seg = a["segments"][row]
                    frame_key = f"E{ep:02d}-{seg}-{role}-{frame:06d}"
                    keys.append(frame_key)
                    frame_jobs[frame_key] = {
                        "key": frame_key,
                        "frame": frame,
                        "video": f"{label['source_path']}/{seg}/{role}.mp4",
                    }
                observation_index[key] = len(observations)
                observations.append({"episode": ep, "row": int(row), "frames": keys})
            return observation_index[key]

        def context(row, arm, a=a):
            previous_valid = row > 0 and a["observed_edges"][row - 1]
            prev = row - 1 if previous_valid else row
            sel = a["details"][row].get("policy_selection") or {}
            queue = [
                float(previous_valid),
                float(sel.get("model_index", 0)) / 50,
                float(sel.get("rtc_delay_steps", 0)) / 50,
            ]
            return {
                "state": a["observation_state"][row],
                "previous_state": a["observation_state"][prev],
                "previous_action": a["submitted_action"][prev] if previous_valid else a["observation_state"][row],
                "base": a["base_target"][row],
                "aux": [float(arm == "left"), float(arm == "right"), *queue],
            }

        for q in [r for r in approved if r["episode_number"] == ep and r["approved_reward"] is not None]:
            c = candidates[q["candidate_id"]]
            close, end = c["close_row"], c["inspection_end_row"]
            # Hindsight segmentation is allowed offline; never feed close-time,
            # outcome frame or phase inferred from the future as policy inputs.
            entry = int(np.searchsorted(a["time"], a["time"][close] - args.preclose_context_seconds))
            earlier = [
                x["inspection_end_row"]
                for x in candidates.values()
                if x["episode_number"] == ep and x["arm"] == c["arm"] and x["close_row"] < close
            ]
            entry = max(entry, max(earlier, default=-1) + 1)
            if not 0 <= entry < close <= end < n:
                excluded["invalid_window"] += 1
                continue
            # Keep the labeled endpoint exact; no shifting across a release or gap.
            if end == 0 or not a["observed_edges"][end - 1]:
                excluded["missing_exact_outcome_transition"] += 1
                continue
            mc = suffix_return(a["time"], a["physical_edges"], end, q["approved_reward"], args.discount_per_second)
            rows = np.flatnonzero(a["observed_edges"][entry:end]) + entry
            if not len(rows):
                excluded["no_valid_transitions"] += 1
                continue
            aid = len(attempts)
            lo = sum(len(x) for x in storage["reward"])
            current, nxt = [context(r, c["arm"]) for r in rows], [context(r + 1, c["arm"]) for r in rows]
            block = {
                "obs_index": np.array([observation(r) for r in rows]),
                "next_obs_index": np.array([observation(r + 1) for r in rows]),
                "action": a["submitted_action"][rows],
                "next_action": a["submitted_action"][rows + 1],
                "reward": np.where(rows + 1 == end, q["approved_reward"], 0.0).astype(np.float32),
                "terminal": rows + 1 == end,
                "discount": args.discount_per_second ** np.diff(a["time"])[rows],
                "mc_return": np.nan_to_num(mc[rows], nan=0.0),
                "mc_valid": np.isfinite(mc[rows]),
                "episode": np.full(len(rows), ep),
                "arm": np.full(len(rows), c["arm"] == "right", dtype=int),
                "attempt": np.full(len(rows), aid),
                "row": rows,
                "outcome": np.full(len(rows), q["approved_reward"]),
                "dt": np.diff(a["time"])[rows],
                "time": a["time"][rows],
            }
            for name in current[0]:
                block[name] = np.array([x[name] for x in current])
                block["next_" + name] = np.array([x[name] for x in nxt])
            for key, value in block.items():
                storage[key].append(value)
            attempts.append(
                {
                    "candidate_id": c["candidate_id"],
                    "episode": ep,
                    "arm": c["arm"],
                    "reward": q["approved_reward"],
                    "entry_row": entry,
                    "close_row": close,
                    "terminal_row": end,
                    "transition_start": lo,
                    "transitions": len(rows),
                    "entry_transition_row": int(rows[0]),
                    "entry_precedes_closure": bool(rows[0] < close),
                    "calibration_transitions": int(np.isfinite(mc[rows]).sum()),
                }
            )
    arrays = {k: np.concatenate(v) for k, v in storage.items()}
    if not arrays or not np.any(arrays["episode"] == 1):
        raise ValueError("empty replay or validation absent")
    arrays["action_coordinates"] = action_coordinates(arrays["action"], arrays["state"])
    arrays["next_action_coordinates"] = action_coordinates(arrays["next_action"], arrays["next_state"])
    train = arrays["episode"] != 1
    stats = fit_normalization(arrays["state"], arrays["action_coordinates"], train)
    np.savez_compressed(args.output / "replay.npz", **arrays)
    write_json(args.output / "normalization.json", stats)
    write_json(args.output / "observations.json", observations)
    write_json(args.output / "frame-jobs.json", list(frame_jobs.values()))
    write_json(args.output / "attempts.json", attempts)
    videos = {j["video"] for j in frame_jobs.values()}
    for video in sorted(videos):
        hashes["raw/" + video] = sha256(args.raw_bundle / "raw" / video)
    result = {
        "schema": "pld_behavior_replay_v1",
        "phase": "offline_single_step_behavior_value_initialization",
        "input_candidates": len(proposals),
        "approved_rewards": sum(r["approved_reward"] is not None for r in approved),
        "replay_attempts": len(attempts),
        "replay_excluded": dict(excluded),
        "train_attempts": sum(x["episode"] != 1 for x in attempts),
        "validation_attempts": sum(x["episode"] == 1 for x in attempts),
        "attempt_outcomes": dict(
            Counter(f"{x['arm']}/{'val' if x['episode'] == 1 else 'train'}/{x['reward']}" for x in attempts)
        ),
        "transitions": len(arrays["reward"]),
        "train_transitions": int(train.sum()),
        "validation_transitions": int((~train).sum()),
        "observations": len(observations),
        "unique_image_frames": len(frame_jobs),
        "calibration_valid_fraction": float(arrays["mc_valid"].mean()),
        "discount_per_second": args.discount_per_second,
        "action_horizon": 1,
        "preclose_context_seconds": args.preclose_context_seconds,
        "behavior_residual": "recorded base_target equals submitted_action on every retained row",
        "state_inputs": "current RGB/state, past state/action, current logged base target, arm, queue scalars; no outcome/phase/timing labels",
        "physical_units": "original raw target units preserved; no conversion; deployment bounds not defined by normalization",
        "global_reward_role": "baseline evaluation only, not broadcast into local rewards",
        "reward_sha256": sha256(args.output / "rewards.json"),
        "source_candidates_sha256": sha256(candidate_path),
        "qualification_sha256": sha256(summary_path),
        "preparation_code_sha256": sha256(Path(__file__)),
        "raw_sha256": hashes,
        "raw_root_hint": str(args.raw_bundle / "raw"),
        "normalization_sha256": sha256(args.output / "normalization.json"),
        "replay_sha256": sha256(args.output / "replay.npz"),
        "not_online_ready": [
            "causal entry/exit",
            "physical residual bounds",
            "RTC chunk alignment",
            "nonzero-action rollouts",
        ],
    }
    write_json(args.output / "manifest.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "raw_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
