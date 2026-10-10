"""Verify native YAM LeRobot full export against immutable source arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def packet_digest(paths):
    digest, count = hashlib.sha256(), 0
    for path in paths:
        with av.open(str(path)) as container:
            for packet in container.demux(video=0):
                if packet.size:
                    digest.update(bytes(packet))
                    count += 1
    return digest.hexdigest(), count


def verify(root):
    conversion = json.loads((root / "conversion_report.json").read_text())
    if conversion["failed"]:
        raise ValueError("native conversion contains failed episodes")
    checks = []
    for dataset in conversion["datasets"]:
        if dataset["state"] != "complete":
            raise ValueError("native conversion contains incomplete/empty episodes")
        raw, output = Path(dataset["source"]), Path(dataset["output"])
        manifest = json.loads((raw / "manifest.json").read_text())
        arrays, sources = {}, []
        for segment in manifest["segments"]:
            with h5py.File(raw / segment["path"] / "samples.h5") as handle:
                size = int(handle["committed_rows"][()])
                for key in (
                    "submitted_action",
                    "observation_state",
                    "observation_valid",
                    "expert_valid",
                    "tick",
                    "time",
                ):
                    arrays.setdefault(key, []).append(handle[key][:size])
                sources.extend(json.loads(row)["source"] for row in handle["details"][:size])
        a = {key: np.concatenate(parts) for key, parts in arrays.items()}
        table = pa.concat_tables([pq.read_table(path) for path in sorted((output / "data").rglob("*.parquet"))])
        if len(table) != manifest["steps"]:
            raise ValueError("converted trajectory length mismatch")

        def column(key, table=table):
            return np.asarray(table[key].to_pylist())

        if not np.array_equal(column("action"), a["submitted_action"].astype(np.float32)):
            raise ValueError("submitted physical actions changed in conversion")
        valid = a["observation_valid"]
        if not np.array_equal(column("observation.state")[valid], a["observation_state"][valid].astype(np.float32)):
            raise ValueError("valid observed states changed in conversion")
        for converted, original in (
            ("source_tick", "tick"),
            ("control_time", "time"),
            ("observation_valid", "observation_valid"),
            ("expert_valid", "expert_valid"),
        ):
            if not np.array_equal(column("complementary_info." + converted), a[original]):
                raise ValueError(f"source metadata changed: {original}")
        source_codes = np.array([{"human": 0, "policy": 1, "hold": 2}[v] for v in sources])
        if not np.array_equal(column("complementary_info.action_source"), source_codes):
            raise ValueError("action sources changed")
        packets = {}
        for role in ("top", "left", "right"):
            original = packet_digest([raw / segment["path"] / f"{role}.mp4" for segment in manifest["segments"]])
            converted = packet_digest(sorted((output / "videos" / f"observation.images.{role}_rgb").rglob("*.mp4")))
            if original != converted:
                raise ValueError(f"packet-copy payload changed: {manifest['episode_id']}/{role}")
            packets[role] = {"sha256": original[0], "packets": original[1]}
        checks.append(
            {
                "episode_id": manifest["episode_id"],
                "frames": len(table),
                "action_state_source_time_exact": True,
                "video_packet_payloads": packets,
            }
        )
    result = {
        "schema": "yam_conversion_verification_v1",
        "episodes": checks,
        "frames": sum(item["frames"] for item in checks),
        "passed": True,
    }
    path = root.parent / "conversion_verification.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = verify(args.output)
    print(json.dumps({"episodes": len(result["episodes"]), "frames": result["frames"], "passed": result["passed"]}))
