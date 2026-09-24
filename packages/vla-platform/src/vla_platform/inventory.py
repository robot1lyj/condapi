"""Create a provenance-bound episode inventory from an audited file list."""

import hashlib
import json
from pathlib import Path

from vla_platform.contracts import require
from vla_platform.project import digest
from vla_platform.project import identifier


def label(value, field):
    require(isinstance(value, str) and 0 < len(value) <= 256
            and all(ord(char) >= 32 for char in value), f"Invalid {field}")
    return value


def source_files_digest(source_root, files, cache=None):
    root = Path(source_root).resolve()
    require(isinstance(files, list) and files, "Each episode needs source files")
    cache = {} if cache is None else cache
    hashes = {}
    for name in files:
        require(isinstance(name, str), "Invalid source file path")
        path = Path(name)
        require(path.is_absolute() and path.is_file(), f"Missing source file: {name}")
        resolved = path.resolve()
        require(resolved.is_relative_to(root), f"Source file escapes dataset: {name}")
        relative = resolved.relative_to(root)
        require(str(relative) not in hashes, "Duplicate source file")
        if resolved not in cache:
            cache[resolved] = digest(resolved)
        hashes[str(relative)] = cache[resolved]
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def create_inventory(source_root, listing_path, output):
    """Compute file hashes and publish JSONL once; source files remain read-only."""
    source_root = Path(source_root).resolve()
    listing_path = Path(listing_path).resolve()
    output = Path(output).resolve()
    require(source_root.is_dir() and listing_path.is_file(), "Missing inventory source/listing")
    require(not output.is_relative_to(source_root), "Write inventory outside the source dataset")
    require(not output.exists(), "Inventory output already exists")
    items = []
    ids = set()
    cache = {}
    with listing_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            require(isinstance(item, dict), f"Listing line {line_number} is not an object")
            allowed = {"episode_id", "group_id", "task_id", "frames", "asset_uri", "source_files"}
            require(set(item).issubset(allowed), f"Unknown listing fields at line {line_number}")
            episode_id = identifier(item.get("episode_id"))
            label(item.get("group_id"), "group_id")
            label(item.get("task_id"), "task_id")
            require(episode_id not in ids, f"Duplicate episode: {episode_id}")
            require(type(item.get("frames")) is int and item["frames"] > 0, "Invalid episode frames")
            raw_files = item.get("source_files")
            require(isinstance(raw_files, list) and raw_files and all(isinstance(name, str) for name in raw_files),
                    "Listing requires source_files")
            files = [str((Path(name) if Path(name).is_absolute() else source_root / name).resolve())
                     for name in raw_files]
            item["source_files"] = sorted(files)
            item["source_sha256"] = source_files_digest(source_root, item["source_files"], cache)
            if item.get("asset_uri"):
                require(isinstance(item["asset_uri"], str), "Invalid asset URI")
                name = Path(item["asset_uri"])
                asset = str((name if name.is_absolute() else source_root / name).resolve())
                require(asset in item["source_files"], "Episode asset must be included in source files")
                item["asset_uri"] = asset
            ids.add(episode_id)
            items.append(item)
    require(bool(items), "Listing is empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for item in sorted(items, key=lambda value: value["episode_id"]):
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    return {"status": "inventory_published_source_unchanged", "path": str(output),
            "episodes": len(items), "sha256": digest(output)}
