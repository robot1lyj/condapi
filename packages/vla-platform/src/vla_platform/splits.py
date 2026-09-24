"""Immutable, group-disjoint dataset splits. This module never opens robot data files."""

import hashlib
import json
import math
import os
from pathlib import Path

from vla_platform.contracts import require
from vla_platform.inventory import source_files_digest
from vla_platform.inventory import label
from vla_platform.project import digest
from vla_platform.project import identifier
from vla_platform.project import inside
from vla_platform.project import read_toml
from vla_platform.project import write_json

PARTITIONS = ("train", "val", "test")


def _strict_json(line):
    return json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON {value}")))


def dataset_config(root, path):
    path = inside(Path(root), str(path))
    data = read_toml(path)
    require(data.get("schema_version") == 1, "Unsupported dataset component version")
    allowed = {"schema_version", "id", "source_format", "source_root", "episode_manifest",
               "source_revision", "contract_id", "state_space", "action_space", "cameras"}
    require(set(data).issubset(allowed), "Unknown dataset component fields")
    identifier(data.get("id"))
    require(data.get("source_format") in ("lerobot", "xr1-json"), "Unsupported dataset source format")
    for field in ("contract_id", "state_space", "action_space", "source_revision"):
        require(isinstance(data.get(field), str) and bool(data[field]), f"Missing dataset {field}")
    cameras = data.get("cameras")
    require(isinstance(cameras, list) and bool(cameras), "Missing dataset cameras")
    require(all(isinstance(value, str) and value for value in cameras), "Invalid camera name")
    require(len(cameras) == len(set(cameras)), "Duplicate dataset camera")
    root_path = Path(data.get("source_root", ""))
    manifest = Path(data.get("episode_manifest", ""))
    require(root_path.is_absolute() and manifest.is_absolute(), "Dataset source paths must be absolute")
    require(root_path.is_dir() and manifest.is_file(), "Dataset source root/episode manifest is missing")
    return path, data


def episodes_from_manifest(path):
    """Read a lightweight source inventory with stable episode and group identities."""
    found = {}
    groups = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = _strict_json(line)
            require(isinstance(item, dict), f"Episode line {line_number} is not an object")
            require(set(item).issubset({"episode_id", "group_id", "task_id", "frames", "source_sha256",
                                        "asset_uri", "source_files"}), f"Unknown episode fields at line {line_number}")
            episode_id = identifier(item.get("episode_id"))
            group_id = label(item.get("group_id"), "group_id")
            label(item.get("task_id"), "task_id")
            frames = item.get("frames")
            require(type(frames) is int and frames > 0, f"Invalid frames at line {line_number}")
            source_digest = item.get("source_sha256")
            require(
                isinstance(source_digest, str) and len(source_digest) == 64
                and all(char in "0123456789abcdef" for char in source_digest),
                f"Missing source SHA256 at line {line_number}",
            )
            require(episode_id not in found, f"Duplicate episode id: {episode_id}")
            found[episode_id] = item
            groups.setdefault(group_id, []).append(episode_id)
    require(bool(found), "Episode manifest is empty")
    return found, groups


def _validate_asset_identity(dataset, episodes):
    root = Path(dataset["source_root"]).resolve()
    cache = {}
    for episode_id, item in episodes.items():
        asset = item.get("asset_uri")
        if dataset["source_format"] == "xr1-json":
            require(isinstance(asset, str) and asset, f"XR-1 episode {episode_id} needs an asset URI")
        if asset is not None:
            path = Path(asset)
            require(path.is_absolute() and path.is_file(), f"Missing episode asset: {episode_id}")
            require(path.resolve().is_relative_to(root), f"Episode asset escapes source root: {episode_id}")
        files = item.get("source_files")
        require(isinstance(files, list) and bool(files), f"Missing source files: {episode_id}")
        if asset is not None:
            require(str(path.resolve()) in files, f"Episode asset missing from source files: {episode_id}")
        require(source_files_digest(root, files, cache) == item["source_sha256"],
                f"Episode source changed: {episode_id}")


def split_recipe(root, path):
    path = inside(Path(root), str(path))
    recipe = read_toml(path)
    require(recipe.get("schema_version") == 1, "Unsupported split recipe version")
    require(set(recipe).issubset({"schema_version", "id", "dataset", "seed", "ratios"}),
            "Unknown split recipe fields")
    identifier(recipe.get("id"))
    require(type(recipe.get("seed")) is int and recipe["seed"] >= 0, "Split seed must be a nonnegative integer")
    dataset_path, dataset = dataset_config(root, recipe.get("dataset"))
    ratios = recipe.get("ratios")
    require(isinstance(ratios, dict) and set(ratios) == set(PARTITIONS), "Declare train/val/test ratios")
    require(all(type(ratios[name]) in (int, float) and math.isfinite(ratios[name])
                and 0 <= ratios[name] <= 1 for name in PARTITIONS), "Invalid split ratios")
    require(abs(sum(ratios.values()) - 1) < 1e-9 and ratios["train"] > 0, "Ratios must sum to one with train > 0")
    return path, recipe, dataset_path, dataset


def _assign_groups(groups, ratios, seed):
    active = [name for name in PARTITIONS if ratios[name] > 0]
    require(len(groups) >= len(active), "Too few groups for nonempty partitions")
    ordered = sorted(groups, key=lambda group: (hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), group))
    assigned = {name: [] for name in PARTITIONS}
    counts = dict.fromkeys(PARTITIONS, 0)
    total = sum(len(members) for members in groups.values())
    for index, group in enumerate(ordered):
        if index < len(active):
            choice = active[index]
        else:
            choice = max(active, key=lambda name: (ratios[name] * total - counts[name], -PARTITIONS.index(name)))
        assigned[choice].extend(groups[group])
        counts[choice] += len(groups[group])
    return {name: sorted(ids) for name, ids in assigned.items()}


def create_split(root, recipe_path, output):
    """Publish only a membership manifest in a new directory; preserve all source assets."""
    root = Path(root).resolve()
    recipe_path, recipe, dataset_path, dataset = split_recipe(root, recipe_path)
    source_root = Path(dataset["source_root"]).resolve()
    source_manifest = Path(dataset["episode_manifest"]).resolve()
    output = Path(output).resolve()
    require(not output.is_relative_to(source_root), "Split output must be outside the source dataset")
    episodes, groups = episodes_from_manifest(source_manifest)
    _validate_asset_identity(dataset, episodes)
    partitions = _assign_groups(groups, recipe["ratios"], recipe["seed"])
    manifest = {
        "schema_version": 1,
        "split_id": recipe["id"],
        "dataset_id": dataset["id"],
        "dataset_config": str(dataset_path.relative_to(root)),
        "dataset_config_sha256": digest(dataset_path),
        "recipe": str(recipe_path.relative_to(root)),
        "recipe_sha256": digest(recipe_path),
        "episode_manifest_sha256": digest(source_manifest),
        "source_revision": dataset["source_revision"],
        "seed": recipe["seed"],
        "ratios": recipe["ratios"],
        "partitions": partitions,
        "episode_count": len(episodes),
        "group_count": len(groups),
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest)
    with (output / "READY").open("x") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    return {"status": "split_published_source_unchanged", "path": str(output / "manifest.json"),
            "counts": {name: len(ids) for name, ids in partitions.items()}}


def inspect_split(root, path):
    """Recompute all source and membership checks before a split enters a training plan."""
    root = Path(root).resolve()
    path = Path(path).resolve()
    require(path.name == "manifest.json" and (path.parent / "READY").is_file(), "Split is incomplete")
    manifest = _strict_json(path.read_text())
    require(manifest.get("schema_version") == 1, "Unsupported split artifact version")
    dataset_path, dataset = dataset_config(root, manifest["dataset_config"])
    recipe_path, recipe, expected_dataset_path, _ = split_recipe(root, manifest["recipe"])
    require(dataset_path == expected_dataset_path, "Split recipe/dataset mismatch")
    source_manifest = Path(dataset["episode_manifest"]).resolve()
    require(manifest["dataset_id"] == dataset["id"], "Split dataset identity mismatch")
    require(manifest["source_revision"] == dataset["source_revision"], "Split source revision mismatch")
    for key, file in (("dataset_config_sha256", dataset_path), ("recipe_sha256", recipe_path),
                      ("episode_manifest_sha256", source_manifest)):
        require(manifest[key] == digest(file), f"Split source changed: {key}")
    episodes, groups = episodes_from_manifest(source_manifest)
    _validate_asset_identity(dataset, episodes)
    expected = _assign_groups(groups, recipe["ratios"], recipe["seed"])
    require(manifest["partitions"] == expected, "Split membership or group separation changed")
    require(manifest["episode_count"] == len(episodes), "Split episode count changed")
    require(manifest["group_count"] == len(groups), "Split group count changed")
    return manifest, dataset, episodes
