"""PARTS wire contracts and publication checks; standard library only."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

PROTOCOL = "yam-parts-v1"
STATE_SCHEMA = "yam-parts-state-v1"
INDICES = {"left": list(range(6)), "right": list(range(7, 13))}
MODES = ("off", "shadow", "collect", "eval")
PHASES = ("READY", "ACTIVE_DESCENT", "ACTIVE_CLOSURE", "EXIT_PENDING", "WAIT_REARM")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def object_value(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    return float(value)


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def numeric_array(value: Any, shape: tuple[int, ...], name: str) -> list:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != shape[0]:
        raise ValueError(f"{name} must have shape {shape}")
    if len(shape) == 1:
        return [finite(v, name) for v in value]
    return [numeric_array(v, shape[1:], name) for v in value]


def bool_array(value: Any, length: int, name: str) -> list[bool]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != length or any(type(v) is not bool for v in value):
        raise ValueError(f"{name} must be {length} booleans")
    return list(value)


def contract(value: Any) -> dict:
    declaration = object_value(value, "contract")
    if declaration.get("protocol") != PROTOCOL or declaration.get("residual_space") != "joint_delta_rad":
        raise ValueError("Unsupported PARTS protocol/residual space")
    if declaration.get("horizon") != 50 or declaration.get("state_dim") != 14:
        raise ValueError("PARTS requires H50/14D")
    if not math.isclose(finite(declaration.get("action_dt"), "action_dt"), 1 / 30, rel_tol=0, abs_tol=1e-8):
        raise ValueError("PARTS requires action_dt=1/30")
    modes = declaration.get("supported_modes")
    if not isinstance(modes, list) or not modes or any(mode not in MODES for mode in modes):
        raise ValueError("Invalid supported_modes")
    text(declaration.get("contract_sha"), "contract_sha")
    text(declaration.get("feature_schema_id"), "feature_schema_id")
    text(declaration.get("behavior_manifest_ref"), "behavior_manifest_ref")
    per_arm = object_value(declaration.get("per_arm"), "per_arm")
    for arm, indices in INDICES.items():
        spec = object_value(per_arm.get(arm), arm)
        if spec.get("indices") != indices:
            raise ValueError(f"Wrong {arm} residual indices")
        bounds = numeric_array(spec.get("B_rad"), (6,), f"{arm}.B_rad")
        if any(v < 0 for v in bounds):
            raise ValueError("Residual bounds cannot be negative")
    return declaration


def request(value: Any, declaration: dict, *, rtc: dict | None = None) -> dict:
    value = object_value(value, "parts")
    if value.get("protocol") != PROTOCOL or value.get("contract_sha") != declaration["contract_sha"]:
        raise ValueError("PARTS request protocol/contract mismatch")
    mode = value.get("mode")
    if mode not in declaration["supported_modes"]:
        raise ValueError("Unsupported PARTS mode")
    context = object_value(value.get("context"), "context")
    for key in ("run_id", "session_id"):
        text(context.get(key), key)
    for key in ("epoch", "request_id", "observation_id", "observation_policy_tick"):
        integer(context.get(key), key)
    if value.get("active_arm") not in (None, "left", "right"):
        raise ValueError("Invalid active_arm")
    if rtc is not None:
        if context["observation_policy_tick"] != rtc.get("observation_policy_tick"):
            raise ValueError("PARTS and RTC observation ticks differ")
        delay = integer(rtc.get("delay_steps"), "delay_steps")
        if delay >= 50:
            raise ValueError("RTC delay must be <50")
    arms = object_value(value.get("arms"), "arms")
    for arm in INDICES:
        state = object_value(arms.get(arm), arm)
        if state.get("phase") not in PHASES:
            raise ValueError(f"Unknown {arm} phase")
        if type(state.get("eligible")) is not bool:
            raise ValueError(f"{arm}.eligible must be bool")
        for key in ("height_m", "h_goal_m", "error_m"):
            if state.get(key) is not None:
                finite(state[key], f"{arm}.{key}")
        if all(state.get(key) is not None for key in ("height_m", "h_goal_m", "error_m")) and not math.isclose(
            state["error_m"], state["height_m"] - state["h_goal_m"], abs_tol=1e-7
        ):
            raise ValueError(f"{arm} height error inconsistent")
    return value


def reply(value: Any, sent: dict, declaration: dict, *, delay: int = 0) -> dict:
    value = object_value(value, "parts reply")
    for key in ("protocol", "contract_sha", "mode", "context"):
        if value.get(key) != sent.get(key):
            raise ValueError(f"PARTS reply {key} mismatch")
    text(value.get("behavior_snapshot_id"), "behavior_snapshot_id")
    for arm in INDICES:
        item = object_value(value.get("candidates", {}).get(arm), f"{arm} candidate")
        u = numeric_array(item.get("u"), (50, 6), "u")
        if any(abs(v) > 1 for row in u for v in row):
            raise ValueError("Candidate is outside [-1,1]")
        bounds = numeric_array(item.get("B_rad"), (6,), "B_rad")
        if bounds != numeric_array(declaration["per_arm"][arm]["B_rad"], (6,), "B_rad"):
            raise ValueError("Candidate bounds mismatch")
        mask = bool_array(item.get("editable_mask"), 50, "editable_mask")
        if any(mask[:delay]):
            raise ValueError("Committed prefix cannot be editable")
        text(item.get("actor_snapshot_id"), "actor_snapshot_id")
        if type(item.get("exploration_applied")) is not bool:
            raise ValueError("exploration_applied must be bool")
        if value["mode"] in ("shadow", "eval") and item["exploration_applied"]:
            raise ValueError("Exploration must be disabled in shadow/eval")
    features = object_value(value.get("features"), "features")
    if features.get("feature_schema_id") != declaration["feature_schema_id"]:
        raise ValueError("Feature schema mismatch")
    if value["mode"] in ("collect", "eval") and "z" not in features:
        text(features.get("feature_ref"), "feature_ref")
        text(features.get("sha256"), "feature sha256")
    return value


def height_reward(recipe: dict, error_m: float, grasp_reward: int = 0, *, phase: str = "ACTIVE_DESCENT") -> dict:
    if recipe.get("formula") != "negative_absolute_height_error_v1":
        raise ValueError("Unsupported height reward formula")
    text(recipe.get("schema"), "reward schema")
    height_weight = finite(recipe.get("height_weight"), "height_weight")
    grasp_weight = finite(recipe.get("grasp_weight"), "grasp_weight")
    if height_weight < 0 or grasp_weight <= 0 or grasp_reward not in (0, 1):
        raise ValueError("Invalid reward weights/grasp result")
    if recipe.get("height_scope") not in ("active_descent", "whole_attempt"):
        raise ValueError("Explicit height_scope required")
    if phase not in PHASES:
        raise ValueError("Unknown reward phase")
    height = -abs(finite(error_m, "error_m"))
    if recipe["height_scope"] == "active_descent" and phase != "ACTIVE_DESCENT":
        height = 0.0
    return {
        "height_reward": height,
        "grasp_reward": grasp_reward,
        "total_reward": height_weight * height + grasp_weight * grasp_reward,
        "valid": True,
        "reward_schema": recipe["schema"],
    }


def audit_publication(root: Path, *, allow_mock: bool = False) -> dict:
    """Verify immutable producer files, without claiming RL transition readiness."""
    root = root.resolve()
    publication = json.loads((root / "publication.json").read_text())
    run = json.loads((root / "run.json").read_text())
    if run.get("schema") != "yam_parts_raw_v1" or publication.get("schema") != "yam_parts_raw_v1":
        raise ValueError("Unknown publication schema")
    if publication.get("run_id") != run.get("run_id") or publication.get("client_complete") is not True:
        raise ValueError("Incomplete/mismatched publication")
    if run.get("mock") is not False and not allow_mock:
        raise ValueError("Mock/unknown data cannot enter training")
    members = publication.get("files")
    if not isinstance(members, dict) or not members:
        raise ValueError("Publication files must be a nonempty path-to-hash object")
    if publication.get("training_ready") is not False or publication.get("gaps"):
        raise ValueError("Producer cannot grant READY or hide recording gaps")
    seen = set()
    for relative, item in members.items():
        text(relative, "file path")
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root) or relative in seen:
            raise ValueError("Unsafe/duplicate publication file path")
        seen.add(relative)
        if not path.is_file() or path.stat().st_size != integer(item.get("bytes"), "bytes"):
            raise ValueError(f"Missing/size mismatch: {relative}")
        if sha256(path) != item.get("sha256"):
            raise ValueError(f"SHA256 mismatch: {relative}")
    required = {"run.json", "requests.jsonl", "requests.h5", "events.jsonl", "attempts.jsonl"}
    if not required.issubset(seen):
        raise ValueError(f"Publication lacks required files: {sorted(required - seen)}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "publication.json"
    }
    if seen != actual:
        raise ValueError("Publication file index is incomplete")
    return {
        "schema": "yam_parts_file_audit_v1",
        "status": "files_verified_not_training_ready",
        "run_id": run["run_id"],
        "publication_sha256": sha256(root / "publication.json"),
        "files_verified": len(seen),
        "mock": run.get("mock"),
        "training_ready": False,
    }
