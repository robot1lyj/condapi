"""Commit receipts for native DeepSpeed checkpoints, without model imports."""

import json
import os
from pathlib import Path
import re
import tempfile


def checkpoint_step(path):
    match = re.fullmatch(r"step_(\d+)\.ckpt", Path(path).name)
    return int(match[1]) if match else None


def receipt_path(path):
    path = Path(path).resolve()
    return path.with_name(path.name + ".commit.json")


def inventory(path):
    path = Path(path).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("Expected a native DeepSpeed checkpoint directory")
    result = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("Unexpected symlink inside checkpoint shards")
        if item.is_file():
            size = item.stat().st_size
            if size == 0:
                raise ValueError(f"Empty checkpoint file: {item}")
            result[str(item.relative_to(path))] = size
    return result


def commit_checkpoint(path, step, world_size):
    if type(world_size) is not int or world_size < 1 or type(step) is not int:
        raise ValueError("Invalid checkpoint step or world size")
    path = Path(path).resolve(strict=True)
    files = inventory(path)
    models = sum(name.endswith("model_states.pt") for name in files)
    optimizers = sum(name.endswith("optim_states.pt") for name in files)
    if checkpoint_step(path) != step or step < 1 or models < 1 or optimizers < world_size:
        raise ValueError("Incomplete model/optimizer shards or inconsistent checkpoint step")
    payload = {"schema_version": 1, "global_step": step, "world_size": world_size, "files": files}
    receipt = receipt_path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=receipt.parent, prefix=".commit-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, receipt)
    return payload


def validate_checkpoint(path, world_size):
    if type(world_size) is not int or world_size < 1:
        raise ValueError("Invalid checkpoint world size")
    path = Path(path).resolve(strict=True)
    payload = json.loads(receipt_path(path).read_text())
    if (checkpoint_step(path) is None or checkpoint_step(path) < 1
            or payload.get("schema_version") != 1 or payload.get("world_size") != world_size
            or payload.get("global_step") != checkpoint_step(path)
            or payload.get("files") != inventory(path)):
        raise ValueError("Checkpoint receipt, shards, world size or step changed")
    if (sum(name.endswith("model_states.pt") for name in payload["files"]) < 1
            or sum(name.endswith("optim_states.pt") for name in payload["files"]) < world_size):
        raise ValueError("Checkpoint is missing model or optimizer shards")
    return payload
