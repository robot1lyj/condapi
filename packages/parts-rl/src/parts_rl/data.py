"""Audited decision replay. No torch import and no training execution here."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .state import NON_VISUAL_DIM
from .state import STATE_SCHEMA


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def discounted_return(rewards, gamma):
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all() or not 0 < gamma <= 1:
        raise ValueError("Finite nonempty tick rewards and explicit gamma required")
    return float(np.dot(np.power(gamma, np.arange(values.size)), values))


def curate_attempts(attempts, failure_fraction, seed):
    if not 0 <= failure_fraction <= 1:
        raise ValueError("failure_fraction must be in [0,1]")
    success, failure = [], []
    for attempt in attempts:
        if attempt.get("split_role") != "train" or attempt.get("mock") is not False:
            continue
        if attempt.get("result") == "success":
            success.append(attempt["attempt_id"])
        elif attempt.get("result") == "failure":
            failure.append(attempt["attempt_id"])
    failure.sort()
    rng = np.random.default_rng(seed)
    n = int(np.floor(len(failure) * failure_fraction))
    selected = [] if n == 0 else rng.choice(failure, n, replace=False).tolist()
    return sorted(success + selected)


class Replay:
    """Immutable NPZ shards whose decision provenance has already been audited."""

    def __init__(self, ready_path, arm, *, expected_contract=None, selected_attempts=None):
        ready_path = Path(ready_path).resolve()
        metadata = json.loads(ready_path.read_text())
        if (
            metadata.get("schema") != "yam_parts_replay_v1"
            or metadata.get("status") != "READY"
            or metadata.get("state_schema") != STATE_SCHEMA
            or metadata.get("arm") != arm
            or metadata.get("split_role") != "train"
            or metadata.get("mock") is not False
            or metadata.get("action_semantics") != "accepted_candidate_plan_v1"
        ):
            raise ValueError("Replay is not audited autonomous train data")
        if expected_contract is not None and metadata.get("contract_sha") != expected_contract:
            raise ValueError("Replay contract mismatch")
        feature_dim = metadata["feature_dim"]
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("Replay feature_dim invalid")
        self.state_dim = feature_dim + NON_VISUAL_DIM
        if metadata.get("state_dim") != self.state_dim:
            raise ValueError("Replay state schema dimension mismatch")
        self.metadata = metadata
        rows = {}
        origins = []
        groups = set()
        keys = (
            "state",
            "next_state",
            "action",
            "action_mask",
            "executed_mask",
            "next_action_mask",
            "reward",
            "elapsed_steps",
            "bootstrap",
            "success",
            "attempt_id",
            "group_id",
        )
        for member in metadata.get("shards", []):
            path = (ready_path.parent / member["path"]).resolve()
            if not path.is_relative_to(ready_path.parent) or digest(path) != member["sha256"]:
                raise ValueError("Replay path/hash mismatch")
            with np.load(path, allow_pickle=False) as shard:
                if set(shard.files) != set(keys):
                    raise ValueError("Replay shard fields mismatch")
                arrays = {key: shard[key].copy() for key in keys}
            n = arrays["state"].shape[0]
            shapes = {
                "state": (n, self.state_dim),
                "next_state": (n, self.state_dim),
                "action": (n, 300),
                "action_mask": (n, 300),
                "executed_mask": (n, 300),
                "next_action_mask": (n, 300),
            }
            for key in keys:
                if arrays[key].shape != shapes.get(key, (n,)):
                    raise ValueError(f"Replay {key} shape mismatch")
                if key not in ("attempt_id", "group_id") and not np.isfinite(arrays[key]).all():
                    raise ValueError(f"Replay {key} nonfinite")
            for key in ("action_mask", "executed_mask", "next_action_mask", "bootstrap", "success"):
                if arrays[key].dtype != np.bool_:
                    raise ValueError(f"Replay {key} must be bool")
            if any(arrays[key].dtype.kind not in "US" for key in ("attempt_id", "group_id")):
                raise ValueError("Replay IDs must be text")
            if (
                np.any(abs(arrays["action"]) > 1)
                or np.any(arrays["executed_mask"] & ~arrays["action_mask"])
                or arrays["elapsed_steps"].dtype.kind not in "iu"
                or np.any(arrays["elapsed_steps"] <= 0)
            ):
                raise ValueError("Replay action/masks/timing/terminal mismatch")
            if np.any(arrays["next_action_mask"][~arrays["bootstrap"]]):
                raise ValueError("Terminal transitions cannot have a bootstrap action mask")
            labels = {value["attempt_id"]: value for value in metadata["attempts"]}
            if len(labels) != len(metadata["attempts"]):
                raise ValueError("Duplicate replay attempt labels")
            for attempt_id in np.unique(arrays["attempt_id"]):
                label = labels.get(attempt_id)
                selected = arrays["attempt_id"] == attempt_id
                if (
                    not label
                    or label.get("split_role") != "train"
                    or label.get("mock") is not False
                    or label.get("result") not in ("success", "failure")
                    or np.any(arrays["success"][selected] != (label["result"] == "success"))
                ):
                    raise ValueError("Replay success labels disagree with audited attempt result")
            keep = (
                np.ones(n, np.bool_) if selected_attempts is None else np.isin(arrays["attempt_id"], selected_attempts)
            )
            for key in keys:
                rows.setdefault(key, []).append(arrays[key][keep])
            groups.update(arrays["group_id"][keep].tolist())
            origins.append({"path": member["path"], "sha256": member["sha256"]})
        if not rows:
            raise ValueError("Replay has no shards")
        self.arrays = {key: np.concatenate(value) for key, value in rows.items()}
        if self.arrays["state"].shape[0] == 0:
            raise ValueError("Replay has no selected transitions")
        held_out = set(metadata.get("holdout_groups", []))
        if not held_out or groups & held_out:
            raise ValueError("Replay needs disjoint explicit holdout groups")
        self.origins = origins

    def __len__(self):
        return len(self.arrays["reward"])

    def normalization(self):
        values = self.arrays["state"].astype(np.float64)
        return values.mean(axis=0).astype(np.float32), np.maximum(values.std(axis=0), 1e-3).astype(np.float32)

    def batch(self, size, rng):
        indices = rng.integers(0, len(self), size=size)
        return {key: value[indices] for key, value in self.arrays.items() if key not in ("attempt_id", "group_id")}
