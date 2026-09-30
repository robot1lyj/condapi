"""Standard-library recipe checks for server-only RLT experiments."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/parts-rl/src"))
sys.path.insert(0, str(ROOT / "packages/vla-platform/src"))

from parts_rl.rlt_contract import ALGORITHM  # noqa: E402
from parts_rl.rlt_contract import architecture  # noqa: E402
from parts_rl.rlt_contract import load_token_manifest  # noqa: E402
from vla_platform import parts as wire  # noqa: E402


def load_recipe(path):
    value = json.loads(Path(path).read_text())
    keys = {
        "schema_version",
        "algorithm",
        "allowed_training_hosts",
        "replay",
        "token_manifest",
        "hidden_dims",
        "actor_lr",
        "critic_lr",
        "gamma",
        "tau",
        "policy_frequency",
        "q_weight",
        "reference_weight",
        "reference_dropout",
        "fixed_std",
        "gradient_clip",
        "batch_size",
        "updates",
        "save_every",
        "seed",
    }
    if set(value) != keys or value["schema_version"] != 1 or value["algorithm"] != ALGORITHM:
        raise ValueError("Unknown/incomplete RLT experiment recipe")
    if not isinstance(value["allowed_training_hosts"], list) or not all(
        isinstance(v, str) and v for v in value["allowed_training_hosts"]
    ):
        raise ValueError("Explicit server hostnames required")
    if not isinstance(value["replay"], dict) or set(value["replay"]) != {"left", "right"}:
        raise ValueError("Left/right audited READY paths required")
    for v in value["replay"].values():
        wire.text(v, "READY path")
    wire.text(value["token_manifest"], "token_manifest")
    if not isinstance(value["hidden_dims"], list) or not value["hidden_dims"]:
        raise ValueError("Explicit hidden dimensions required")
    for v in value["hidden_dims"]:
        wire.integer(v, "hidden width", 1)
    for key in ("batch_size", "updates", "save_every", "policy_frequency"):
        wire.integer(value[key], key, 1)
    wire.integer(value["seed"], "seed")
    for key in ("actor_lr", "critic_lr", "q_weight", "fixed_std", "gradient_clip"):
        if wire.finite(value[key], key) <= 0:
            raise ValueError(f"{key} must be positive")
    if wire.finite(value["reference_weight"], "reference_weight") < 0:
        raise ValueError("reference_weight cannot be negative")
    if not 0 < wire.finite(value["gamma"], "gamma") <= 1 or not 0 < wire.finite(value["tau"], "tau") <= 1:
        raise ValueError("Invalid gamma/tau")
    if not 0 <= wire.finite(value["reference_dropout"], "reference_dropout") < 1:
        raise ValueError("Invalid reference dropout")
    return value


def bind_token(recipe, replay):
    manifest, _ = load_token_manifest(recipe["token_manifest"])
    if (
        replay.metadata["feature_schema_id"] != manifest["feature_schema_id"]
        or replay.metadata["feature_dim"] != manifest["architecture"]["embed_dim"]
    ):
        raise ValueError("Replay must contain this frozen RL token; pooled features are incompatible")
    if set(replay.metadata["holdout_groups"]) != set(manifest["data_split"]["holdout_groups"]):
        raise ValueError("Token pretraining and RL must reserve the same holdout layout groups")
    return manifest


def load_token_recipe(path):
    value = json.loads(Path(path).read_text())
    keys = {
        "schema_version",
        "algorithm",
        "allowed_training_hosts",
        "prefix_ready",
        "architecture",
        "learning_rate",
        "weight_decay",
        "gradient_clip",
        "batch_size",
        "updates",
        "save_every",
        "seed",
    }
    if set(value) != keys or value["schema_version"] != 1 or value["algorithm"] != "rlinf_rlt_token_reconstruction_v1":
        raise ValueError("Unknown/incomplete frozen-prefix token recipe")
    architecture(value["architecture"])
    wire.text(value["prefix_ready"], "prefix_ready")
    if not isinstance(value["allowed_training_hosts"], list) or not all(
        isinstance(v, str) and v for v in value["allowed_training_hosts"]
    ):
        raise ValueError("Explicit server hostnames required")
    for key in ("batch_size", "updates", "save_every"):
        wire.integer(value[key], key, 1)
    wire.integer(value["seed"], "seed")
    for key in ("learning_rate", "gradient_clip"):
        if wire.finite(value[key], key) <= 0:
            raise ValueError(f"{key} must be positive")
    if wire.finite(value["weight_decay"], "weight_decay") < 0:
        raise ValueError("weight_decay cannot be negative")
    return value
