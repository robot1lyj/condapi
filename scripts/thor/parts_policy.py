# ruff: noqa: E402, PLC0415, SLF001
# Source checkout imports and opt-in OpenPI feature hook.
"""PARTS response extension. Never changes the base policy's physical actions."""

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "packages/vla-platform/src"))
sys.path.insert(0, str(_ROOT / "packages/parts-rl/src"))

from parts_rl.state import NON_VISUAL_DIM
from parts_rl.state import encode_state
from vla_platform import parts as wire


class FrozenPiFeatures:
    """Capture mean image embeddings from the same eager frozen Pi invocation.

    Opt-in eager path only. An actions-only TRT engine cannot provide this data.
    The original embed_prefix method and inference output remain authoritative.
    """

    def __init__(self, model):
        if not callable(getattr(model, "embed_prefix", None)):
            raise ValueError("Feature capture requires an eager Pi embed_prefix")
        self.model = model
        self.z = None

    @contextmanager
    def capture(self):
        original = self.model.embed_prefix
        had_instance_method = "embed_prefix" in self.model.__dict__
        self.z = None

        def embed(images, masks, tokens, token_masks):
            result = original(images, masks, tokens, token_masks)
            prefix = result[0]
            count = prefix.shape[1] - tokens.shape[1]
            if len(images) != 3 or count <= 0 or count % 3 or prefix.shape[0] != 1:
                raise ValueError("Unexpected Pi three-view prefix feature shape")
            pooled = prefix[:, :count].reshape(1, 3, count // 3, -1).float().mean(dim=2)
            self.z = pooled[0].detach().cpu().numpy().reshape(-1)
            if not np.isfinite(self.z).all():
                raise ValueError("Nonfinite visual features")
            return result

        self.model.embed_prefix = embed
        try:
            yield self
        finally:
            if had_instance_method:
                self.model.embed_prefix = original
            else:
                del self.model.embed_prefix


class _MissingFeatures:
    z = None

    @contextmanager
    def capture(self):
        yield self


class PartsPolicyExtension:
    def __init__(self, manifest, *, actors=None, features=None):
        self.manifest = manifest
        self.metadata = wire.contract(manifest["contract"])
        self.actors = actors or {}
        self.features = features or _MissingFeatures()
        self.snapshot = wire.text(manifest.get("behavior_snapshot_id"), "behavior_snapshot_id")
        self._served = OrderedDict()
        self.feature_dim = wire.integer(manifest.get("feature_dim", 0), "feature_dim")
        self.metadata = {
            **self.metadata,
            "behavior_snapshot_id": self.snapshot,
            "feature_dim": self.feature_dim,
            "state_schema": wire.STATE_SCHEMA,
        }
        if any(mode in self.metadata["supported_modes"] for mode in ("collect", "eval")):
            if features is None or self.feature_dim == 0:
                raise ValueError("collect/eval requires a real frozen visual feature provider")
            if manifest.get("state_schema") != wire.STATE_SCHEMA:
                raise ValueError("Unknown actor state schema")
            for arm in wire.INDICES:
                spec = manifest["actors"][arm]
                if arm not in self.actors and spec.get("initialization") != "zero_residual":
                    raise ValueError(f"Missing {arm} actor")
                if "eval" in self.metadata["supported_modes"] and arm not in self.actors:
                    raise ValueError("eval requires learned actor weights")
        self.exploration_std = manifest.get("exploration_std")
        if "collect" in self.metadata["supported_modes"]:
            if wire.finite(self.exploration_std, "exploration_std") <= 0:
                raise ValueError("collect exploration_std must be explicitly positive")
            wire.integer(manifest.get("exploration_seed"), "exploration_seed")
            for arm in wire.INDICES:
                if not any(self.metadata["per_arm"][arm]["B_rad"]):
                    raise ValueError("collect needs a nonzero reviewed residual bound")

    def infer(self, obs, rtc, payload, base_infer):
        sent = wire.request(payload, self.metadata, rtc=rtc)
        if sent["mode"] == "off":
            return base_infer()
        context = sent["context"]
        key = tuple(context[k] for k in ("run_id", "session_id", "epoch", "request_id"))
        if key in self._served:
            raise ValueError("PARTS request id already served; allocate a new id")
        # Inference is synchronous in WebsocketPolicyServer; this scoped hook
        # captures this observation only, never reuses an earlier request's z.
        with self.features.capture() as capture:
            response, rtc_used, warnings, error = base_infer()
        actions = wire.numeric_array(response.get("actions"), (50, 14), "actions")
        delay = rtc["delay_steps"] if rtc is not None and rtc_used else 0
        z = capture.z
        if z is not None:
            z = np.asarray(z, dtype=np.float32)
            if z.shape != (self.feature_dim,) or not np.isfinite(z).all():
                raise ValueError("Feature manifest dimension mismatch")
        if sent["mode"] in ("collect", "eval") and z is None:
            raise ValueError("Frozen visual features missing; candidates unavailable")
        candidates = {}
        for arm in wire.INDICES:
            u = np.zeros((50, 6), np.float32)
            spec = self.manifest.get("actors", {}).get(arm, {})
            if sent["mode"] != "shadow":
                state = encode_state(z, obs["observation.state"], actions, sent, arm, feature_dim=self.feature_dim)
                if arm in self.actors and not (
                    sent["mode"] == "collect" and self.manifest.get("exploration_space") == "pre_tanh_gaussian"
                ):
                    u = np.asarray(self.actors[arm](state), dtype=np.float32)
                if sent["mode"] == "collect":
                    seed = int(
                        hashlib.sha256(
                            json.dumps(
                                [self.manifest["exploration_seed"], self.snapshot, context, arm],
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest()[:16],
                        16,
                    )
                    rng = np.random.default_rng(seed)
                    if self.manifest.get("exploration_space") == "pre_tanh_gaussian":
                        actor = self.actors.get(arm)
                        u = (
                            actor.sample(state, rng)
                            if actor is not None
                            else np.tanh(rng.normal(0, self.exploration_std, (50, 6))).astype(np.float32)
                        )
                    else:
                        u = np.clip(u + rng.normal(0, self.exploration_std, (50, 6)), -1, 1).astype(np.float32)
            u[:delay] = 0
            candidates[arm] = {
                "u": u,
                "B_rad": self.metadata["per_arm"][arm]["B_rad"],
                "editable_mask": np.arange(50) >= delay,
                "actor_snapshot_id": spec.get("actor_snapshot_id", "zero-shadow"),
                "exploration_applied": sent["mode"] == "collect",
            }
        features = {
            "feature_schema_id": self.metadata["feature_schema_id"],
            "feature_ref": None,
            "status": "missing" if z is None else "available",
        }
        if z is not None:
            features.update(
                z=z, shape=list(z.shape), dtype=str(z.dtype), sha256=hashlib.sha256(z.tobytes()).hexdigest()
            )
        extension = {
            "protocol": wire.PROTOCOL,
            "contract_sha": self.metadata["contract_sha"],
            "context": context.copy(),
            "mode": sent["mode"],
            "behavior_snapshot_id": self.snapshot,
            "candidates": candidates,
            "features": features,
        }
        wire.reply(extension, sent, self.metadata, delay=delay)
        self._served[key] = None
        if len(self._served) > 4096:
            self._served.popitem(last=False)
        return {**response, "parts": extension}, rtc_used, warnings, error


def load_shadow_extension(path, base_metadata):
    """Attach only explicit shadow/off declarations to existing actions-only TRT."""
    manifest = json.loads(Path(path).read_text())
    for key, expected in manifest.get("base_identity", {}).items():
        if base_metadata.get(key) != expected:
            raise ValueError(f"PARTS base identity mismatch: {key}")
    if set(manifest["contract"]["supported_modes"]) - {"off", "shadow"}:
        raise ValueError("Current TRT export has no visual features; shadow/off only")
    return PartsPolicyExtension(manifest)


def load_eager_extension(path, policy, base_metadata):
    """Load pinned FP32 actors against the same eager Pi feature schema."""
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("feature_extractor") == "pi_eager_rlt_final_prefix_v1":
        from rlt_policy import load_rlt_extension

        return load_rlt_extension(manifest_path, policy, base_metadata)
    if manifest.get("feature_extractor") != "pi_eager_image_prefix_mean_three_views_v1":
        raise ValueError("Unknown eager feature extraction contract")
    identity = manifest.get("base_identity", {})
    for key in ("checkpoint_weights_sha256", "norm_stats_sha256"):
        wire.text(identity.get(key), key)
    for key, expected in identity.items():
        if base_metadata.get(key) != expected:
            raise ValueError(f"PARTS base identity mismatch: {key}")
    if not getattr(policy, "_is_pytorch_model", False):
        raise ValueError("PARTS features require eager PyTorch Pi; JAX/TRT need a separate export")
    model = policy._model
    model.requires_grad_(requires_grad=False).eval()
    features = FrozenPiFeatures(model)
    actors = {}
    for arm, spec in manifest.get("actors", {}).items():
        if arm not in wire.INDICES:
            raise ValueError("Unknown actor arm")
        wire.text(spec.get("actor_snapshot_id"), "actor_snapshot_id")
        if spec.get("initialization") == "zero_residual":
            continue
        asset = (manifest_path.parent / spec["path"]).resolve()
        if not asset.is_relative_to(manifest_path.parent) or wire.sha256(asset) != spec["sha256"]:
            raise ValueError("Actor path/hash mismatch")
        from parts_rl.learner import ActorRunner
        import torch

        bundle = torch.load(asset, map_location="cpu", weights_only=True)
        expected = {
            "schema": "yam_parts_actor_v1",
            "arm": arm,
            "contract_sha": manifest["contract"]["contract_sha"],
            "state_schema": wire.STATE_SCHEMA,
            "feature_schema_id": manifest["contract"]["feature_schema_id"],
            "feature_dim": manifest["feature_dim"],
            "state_dim": manifest["feature_dim"] + NON_VISUAL_DIM,
        }
        if any(bundle.get(key) != value for key, value in expected.items()):
            raise ValueError("Actor bundle contract/schema mismatch")
        if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in bundle["weights"].values()):
            raise ValueError("Actor bundle must contain finite FP32 weights")
        actors[arm] = ActorRunner(bundle, device=policy._pytorch_device)
    return PartsPolicyExtension(manifest, actors=actors, features=features)
