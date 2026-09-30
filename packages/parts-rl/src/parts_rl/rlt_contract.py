"""RLT experiment identities and token assets; standard library only."""

import json
from pathlib import Path

from vla_platform import parts as wire

UPSTREAM_COMMIT = "d34d4c320d08cb982de034aa9a011f08dc0fa217"
ALGORITHM = "yam_rlt_gaussian_reference_v1"
FEATURE_EXTRACTOR = "pi_eager_rlt_final_prefix_v1"
PREFIX_SCHEMA = "pi05_yam_final_image_prefix_v1"


def architecture(value):
    keys = {"input_dim", "embed_dim", "prefix_seq_len", "num_layers", "num_heads", "mlp_ratio", "dropout_rate"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Explicit RLinf token architecture required")
    for key in keys - {"mlp_ratio", "dropout_rate"}:
        wire.integer(value[key], key, 1)
    if value["input_dim"] != 2048 or value["embed_dim"] % value["num_heads"] or value["embed_dim"] % 2:
        raise ValueError("Pi05 2048D prefix and an even, head-divisible token dimension required")
    if (
        wire.finite(value["mlp_ratio"], "mlp_ratio") <= 0
        or int(value["embed_dim"] * value["mlp_ratio"]) < 1
        or not 0 <= wire.finite(value["dropout_rate"], "dropout_rate") < 1
    ):
        raise ValueError("Invalid token MLP ratio/dropout")
    return value


def identity(value):
    if not isinstance(value, dict) or set(value) != {"checkpoint_weights_sha256", "norm_stats_sha256"}:
        raise ValueError("Explicit frozen Pi checkpoint/norm identity required")
    for digest in value.values():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid frozen Pi asset SHA256")
    return value


def token_identity(manifest):
    return "rlt-" + wire.canonical_sha(
        {key: manifest[key] for key in ("upstream_commit", "architecture", "base_identity", "prefix_schema", "weights")}
    )


def training_groups(value):
    train, holdout = value.get("train_groups"), value.get("holdout_groups")
    if (
        not isinstance(train, list)
        or not isinstance(holdout, list)
        or not train
        or not holdout
        or not all(isinstance(g, str) and g for g in train + holdout)
        or len(set(train)) != len(train)
        or len(set(holdout)) != len(holdout)
        or set(train) & set(holdout)
    ):
        raise ValueError("Explicit disjoint train/holdout layout groups required")
    return set(train), set(holdout)


def load_token_manifest(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (
        value.get("schema") != "yam_rlt_token_v1"
        or value.get("upstream_commit") != UPSTREAM_COMMIT
        or value.get("prefix_schema") != PREFIX_SCHEMA
        or value.get("feature_extractor") != FEATURE_EXTRACTOR
    ):
        raise ValueError("Unknown RLT token/source/prefix schema")
    if value.get("status") != "trained_not_task_evaluated":
        raise ValueError("A trained token snapshot is required")
    architecture(value["architecture"])
    identity(value["base_identity"])
    training_groups(value["data_split"])
    wire.text(value["prefix_ready_sha256"], "prefix_ready_sha256")
    member = value["weights"]
    if not isinstance(member, dict) or set(member) != {"path", "sha256"}:
        raise ValueError("Explicit token weights path/hash required")
    relative = Path(wire.text(member["path"], "token weights path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe token weights path")
    asset = (path.parent / relative).resolve()
    if not asset.is_relative_to(path.parent) or wire.sha256(asset) != value["weights"]["sha256"]:
        raise ValueError("Token path/hash mismatch")
    if value.get("feature_schema_id") != token_identity(value):
        raise ValueError("Token feature identity mismatch")
    return value, asset
