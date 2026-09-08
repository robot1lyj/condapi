"""Portable model manifest: hash real files, not an opaque checkpoint directory."""

import json
from pathlib import Path
import re

from vla_platform.contracts import require
from vla_platform.contracts import validate_contract
from vla_platform.project import digest
from vla_platform.project import identifier
from vla_platform.project import inside
from vla_platform.project import read_toml
from vla_platform.project import write_json

REQUIRED_ROLES = {"weights", "model_config", "preprocessing", "normalization", "contract", "reference"}


def validate_bundle(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    validate_manifest(manifest, path.parent)
    return {
        "status": "integrity_verified_not_accuracy_or_deployment_accepted",
        "model": manifest["model"],
        "model_version": manifest["model_version"],
        "files": len(manifest["files"]),
        "manifest_sha256": digest(path),
    }


def seal_bundle(recipe_path, output):
    """Create a manifest beside a user-prepared model package; never move/cast weights."""
    recipe_path = Path(recipe_path).resolve()
    output = Path(output).resolve()
    require(output.parent == recipe_path.parent, "Manifest must stay beside its recipe and files")
    require(not output.exists(), "Refusing to overwrite an existing bundle manifest")
    manifest = json.loads(recipe_path.read_text())
    manifest["acceptance"] = "unverified"
    for item in manifest["files"]:
        item["sha256"] = digest(inside(recipe_path.parent, item["path"]))
    # Validate before publishing. The validator can also be used without a temporary file.
    validate_manifest(manifest, output.parent)
    write_json(output, manifest)
    return validate_bundle(output)


def validate_manifest(manifest, root):
    require(manifest.get("schema_version") == 1, "Unsupported bundle version")
    require(manifest.get("acceptance") == "unverified", "Bundle integrity cannot grant deployment acceptance")
    identifier(manifest.get("model"))
    for field in ("model_version", "code_revision", "contract_id"):
        require(isinstance(manifest.get(field), str) and bool(manifest[field]), f"Missing {field}")
    require(manifest.get("format") in ("jax", "pytorch", "tensorrt", "adapter"), "Unknown weight format")
    precision = manifest.get("precision", {})
    require(
        all(precision.get(k) in ("float32", "bfloat16", "mixed") for k in ("storage", "compute")),
        "Declare storage and compute precision",
    )
    if manifest["format"] == "adapter":
        require(
            re.fullmatch(r"[0-9a-f]{64}", manifest.get("base_model_sha256", "")),
            "LoRA bundle requires base model identity",
        )
    files = manifest.get("files", [])
    require(REQUIRED_ROLES.issubset({item["role"] for item in files}), "Incomplete model handoff roles")
    paths = [inside(root, item["path"]) for item in files]
    require(len(set(paths)) == len(paths), "Duplicate bundle file")
    for path, item in zip(paths, files, strict=True):
        require(path.is_file() and digest(path) == item["sha256"], "Bundle hash mismatch")
    contracts = [path for path, item in zip(paths, files, strict=True) if item["role"] == "contract"]
    require(len(contracts) == 1, "Bundle requires exactly one robot contract")
    contract = validate_contract(read_toml(contracts[0]))
    require(contract["id"] == manifest["contract_id"], "Manifest/robot contract mismatch")
