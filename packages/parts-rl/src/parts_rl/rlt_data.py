"""Immutable final-prefix cache validation and sampling; no model imports."""

import json
from pathlib import Path

import numpy as np
from vla_platform import parts as wire

from .rlt_contract import PREFIX_SCHEMA
from .rlt_contract import identity
from .rlt_contract import training_groups


def member_path(root, member):
    relative = Path(wire.text(member["path"], "member path"))
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or wire.sha256(path) != member["sha256"]:
        raise ValueError("RLT cache member path/hash mismatch")
    return path


def load_observations(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (
        value.get("schema") != "yam_rlt_observations_v1"
        or value.get("mock") is not False
        or value.get("split_role") != "train"
    ):
        raise ValueError("Real training observations required")
    train, _ = training_groups(value)
    if not value.get("observations") or not value.get("sources"):
        raise ValueError("Observations and raw publication provenance required")
    seen = set()
    for member in value["observations"]:
        key = wire.text(member["observation_key"], "observation_key")
        if key in seen or member.get("group_id") not in train:
            raise ValueError("Observation identity/split mismatch")
        seen.add(key)
        member_path(path.parent, member)
    return value


class PrefixReplay:
    def __init__(self, ready_path, *, prefix_seq_len):
        self.path = Path(ready_path).resolve()
        value = json.loads(self.path.read_text())
        if (
            value.get("schema") != "yam_rlt_prefixes_v1"
            or value.get("status") != "READY"
            or value.get("prefix_schema") != PREFIX_SCHEMA
            or value.get("mock") is not False
            or value.get("split_role") != "train"
        ):
            raise ValueError("Audited frozen-prefix training cache required")
        identity(value["base_identity"])
        train, _ = training_groups(value)
        if not value.get("members") or not value.get("observations_manifest_sha256"):
            raise ValueError("Empty/unbound final-prefix cache")
        self.metadata, self.members, self.sequence_length = value, value["members"], None
        seen = set()
        for member in self.members:
            if member["observation_key"] in seen or member["group_id"] not in train:
                raise ValueError("Prefix observation/split mismatch")
            seen.add(member["observation_key"])
            prefix, _ = self.read(member)
            length = prefix.shape[0]
            if length > prefix_seq_len or (self.sequence_length is not None and length != self.sequence_length):
                raise ValueError("Prefix sequence length mismatch")
            self.sequence_length = length

    def read(self, member):
        with np.load(member_path(self.path.parent, member), allow_pickle=False) as file:
            if set(file.files) != {"prefix", "mask"}:
                raise ValueError("Prefix cache fields mismatch")
            prefix, mask = file["prefix"], file["mask"]
        if (
            prefix.ndim != 2
            or prefix.shape[1] != 2048
            or prefix.shape[0] <= 0
            or prefix.dtype != np.float32
            or not np.isfinite(prefix).all()
            or mask.dtype != np.bool_
            or mask.shape != prefix.shape[:1]
            or not mask.any()
        ):
            raise ValueError("Finite FP32 final image prefix and nonempty bool mask required")
        return prefix, mask

    def batch(self, size, rng):
        rows = [self.read(self.members[i]) for i in rng.integers(0, len(self.members), size=size)]
        return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])
