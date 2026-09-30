# ruff: noqa: E402
# Standalone stdlib entry point: no model imports or training.
"""Build a new immutable behavior directory from a snapshot or explicit zero init."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/vla-platform/src"))

from vla_platform import parts as wire


def build(output, declaration, identity, *, snapshot=None, feature_dim=None, exploration_std=None, seed=None):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Behavior output must be a new directory")
    wire.contract(declaration)
    for key in ("checkpoint_weights_sha256", "norm_stats_sha256"):
        value = wire.text(identity.get(key), key)
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Actual base asset SHA256 required")
    trained = None
    if snapshot:
        snapshot = Path(snapshot).resolve()
        if (snapshot / "LATEST.json").is_file():
            pointer = json.loads((snapshot / "LATEST.json").read_text())
            path = (snapshot / pointer["path"]).resolve()
            if not path.is_relative_to(snapshot) or wire.sha256(path / "training.json") != pointer["training_sha256"]:
                raise ValueError("Unsafe/changed snapshot pointer")
            snapshot = path
        trained = json.loads((snapshot / "training.json").read_text())
        if (
            trained.get("schema") != "yam_parts_training_v1"
            or trained.get("contract_sha") != declaration["contract_sha"]
            or trained.get("feature_schema_id") != declaration["feature_schema_id"]
            or trained.get("state_schema") != wire.STATE_SCHEMA
        ):
            raise ValueError("Training snapshot contract/feature mismatch")
        feature_dim = trained["feature_dim"]
    if "eval" in declaration["supported_modes"] and trained is None:
        raise ValueError("eval requires learned actors")
    wire.integer(feature_dim, "feature_dim", 1)
    if "collect" in declaration["supported_modes"]:
        if wire.finite(exploration_std, "exploration_std") <= 0:
            raise ValueError("Positive explicit exploration required for collect")
        wire.integer(seed, "seed")
        if any(not any(declaration["per_arm"][arm]["B_rad"]) for arm in wire.INDICES):
            raise ValueError("Reviewed nonzero bounds required for collect")
    actors, assets = {}, []
    for arm in wire.INDICES:
        if trained is None:
            actors[arm] = {"initialization": "zero_residual", "actor_snapshot_id": "zero-" + arm}
            continue
        member = trained["actors"][arm]
        path = (snapshot / member["path"]).resolve()
        if not path.is_relative_to(snapshot) or wire.sha256(path) != member["sha256"]:
            raise ValueError("Actor path/hash mismatch")
        actors[arm] = {"path": arm + "_actor.pt", "sha256": member["sha256"], "actor_snapshot_id": member["sha256"]}
        assets.append((path, actors[arm]["path"]))
    manifest = {
        "schema": "yam_parts_behavior_v1",
        "contract": declaration,
        "state_schema": wire.STATE_SCHEMA,
        "base_identity": identity,
        "feature_dim": feature_dim,
        "feature_extractor": "pi_eager_image_prefix_mean_three_views_v1",
        "actors": actors,
        "exploration_std": exploration_std,
        "exploration_seed": seed,
        "training_manifest_sha256": None if trained is None else wire.sha256(snapshot / "training.json"),
        "status": "candidate_not_robot_or_latency_validated",
    }
    manifest["behavior_snapshot_id"] = wire.canonical_sha(manifest)
    output.mkdir(parents=True, exist_ok=False)
    for source, relative in assets:
        shutil.copy2(source, output / relative)
    path = output / "behavior.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return {"path": str(path), "behavior_snapshot_id": manifest["behavior_snapshot_id"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--base-identity", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path)
    source.add_argument("--zero-residual", action="store_true")
    parser.add_argument("--feature-dim", type=int, help="explicit dimension for initial zero actor")
    parser.add_argument("--exploration-std", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    result = build(
        args.output,
        json.loads(args.contract.read_text()),
        json.loads(args.base_identity.read_text()),
        snapshot=args.snapshot,
        feature_dim=args.feature_dim,
        exploration_std=args.exploration_std,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
