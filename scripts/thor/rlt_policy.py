# ruff: noqa: PLC0415, SLF001
"""Pinned RLT token/actor loading for the opt-in eager Pi extension."""

import json
from pathlib import Path

from parts_policy import PartsPolicyExtension
from parts_policy import wire
from parts_rl.rlt_contract import ALGORITHM
from parts_rl.rlt_contract import FEATURE_EXTRACTOR
from parts_rl.rlt_contract import load_token_manifest
from parts_rl.rlt_features import FrozenRltFeatures
from parts_rl.state import NON_VISUAL_DIM


def verify_manifest(path, base_metadata):
    """Asset/provenance checks are usable without importing any model library."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    snapshot_id = manifest.get("behavior_snapshot_id")
    if snapshot_id != wire.canonical_sha({k: v for k, v in manifest.items() if k != "behavior_snapshot_id"}):
        raise ValueError("RLT behavior identity mismatch")
    wire.contract(manifest["contract"])
    if (
        manifest.get("schema") != "yam_parts_behavior_v1"
        or manifest.get("algorithm") != ALGORITHM
        or manifest.get("feature_extractor") != FEATURE_EXTRACTOR
        or manifest.get("exploration_space") != "pre_tanh_gaussian"
    ):
        raise ValueError("Unknown RLT behavior schema")
    spec = manifest["token_manifest"]
    token_path = (path.parent / spec["path"]).resolve()
    if not token_path.is_relative_to(path.parent) or wire.sha256(token_path) != spec["sha256"]:
        raise ValueError("RLT token manifest path/hash mismatch")
    token, _ = load_token_manifest(token_path)
    if (
        token["base_identity"] != manifest["base_identity"]
        or any(base_metadata.get(k) != v for k, v in token["base_identity"].items())
        or token["feature_schema_id"] != manifest["contract"]["feature_schema_id"]
        or token["architecture"]["embed_dim"] != manifest["feature_dim"]
    ):
        raise ValueError("RLT token/base/feature binding mismatch")
    if set(manifest["actors"]) != set(wire.INDICES):
        raise ValueError("Explicit left/right actors required")
    for actor in manifest["actors"].values():
        if actor.get("initialization") == "zero_residual":
            continue
        asset = (path.parent / actor["path"]).resolve()
        if not asset.is_relative_to(path.parent) or wire.sha256(asset) != actor["sha256"]:
            raise ValueError("RLT actor path/hash mismatch")
    return manifest, token_path


def load_rlt_extension(path, policy, base_metadata):
    manifest, token_path = verify_manifest(path, base_metadata)
    if not getattr(policy, "_is_pytorch_model", False):
        raise ValueError("RLT requires eager PyTorch Pi final-prefix features")
    from parts_rl.rlt import ActorRunner
    from parts_rl.rlt import TokenRunner
    import torch

    model = policy._model.requires_grad_(requires_grad=False).eval()
    token = TokenRunner(token_path, device=policy._pytorch_device)
    features = FrozenRltFeatures(model, token)
    actors = {}
    for arm, spec in manifest["actors"].items():
        if spec.get("initialization") == "zero_residual":
            continue
        bundle = torch.load(Path(path).resolve().parent / spec["path"], map_location="cpu", weights_only=True)
        expected = {
            "schema": "yam_rlt_actor_v1",
            "algorithm": ALGORITHM,
            "arm": arm,
            "contract_sha": manifest["contract"]["contract_sha"],
            "state_schema": wire.STATE_SCHEMA,
            "feature_schema_id": manifest["contract"]["feature_schema_id"],
            "feature_dim": manifest["feature_dim"],
            "state_dim": manifest["feature_dim"] + NON_VISUAL_DIM,
        }
        if (
            any(bundle.get(k) != v for k, v in expected.items())
            or bundle.get("token_manifest", {}).get("sha256") != manifest["token_manifest"]["sha256"]
        ):
            raise ValueError("RLT actor bundle binding mismatch")
        if wire.finite(bundle.get("fixed_std"), "fixed_std") <= 0 or (
            "collect" in manifest["contract"]["supported_modes"] and bundle["fixed_std"] != manifest["exploration_std"]
        ):
            raise ValueError("RLT Gaussian std mismatch")
        if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in bundle["weights"].values()):
            raise ValueError("Finite FP32 RLT actor weights required")
        actors[arm] = ActorRunner(bundle, device=policy._pytorch_device)
    return PartsPolicyExtension(manifest, actors=actors, features=features)
