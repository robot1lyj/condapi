"""CPU-only integrity audit for a verified OpenWAM foundation checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np


def scan_safetensors(path: Path) -> dict:
    with path.open("rb") as stream:
        header_size_bytes = stream.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"Truncated safetensors header: {path}")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        if not 0 < header_size <= 64 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header size: {path}")
        header = json.loads(stream.read(header_size))
        nonfinite = {}
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if tensor["dtype"] != "BF16":
                raise ValueError(f"Expected BF16 foundation weights, got {tensor['dtype']} in {name}")
            start, end = tensor["data_offsets"]
            if start < 0 or end < start or (end - start) % 2:
                raise ValueError(f"Invalid tensor offsets in {name}")
            stream.seek(8 + header_size + start)
            remaining = end - start
            count = 0
            while remaining:
                block = stream.read(min(remaining, 8 * 1024 * 1024))
                if not block:
                    raise ValueError(f"Truncated tensor data in {name}")
                bits = np.frombuffer(block, dtype="<u2")
                count += int(np.count_nonzero((bits & 0x7F80) == 0x7F80))
                remaining -= len(block)
            if count:
                nonfinite[name] = count
    return {"tensor_count": len(header) - int("__metadata__" in header), "nonfinite_by_tensor": nonfinite}


def audit(root: Path, output: Path, receipt_name: str = "foundation-download-receipt.json") -> dict:
    root, output = root.resolve(), output.resolve()
    receipt_path = root / receipt_name
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    if output.exists():
        raise FileExistsError(output)
    receipt = json.loads(receipt_path.read_text())
    reports = {}
    for entry in receipt["files"]:
        if entry["file"].endswith(".safetensors"):
            path = root / entry["file"]
            if path.stat().st_size != entry["size"]:
                raise ValueError(f"File size changed after download receipt: {path}")
            reports[entry["file"]] = scan_safetensors(path)
    if not reports:
        raise ValueError("The receipt contains no safetensors weights")
    result = {
        "repo": receipt["repo"],
        "revision": receipt["revision"],
        "finite": all(not item["nonfinite_by_tensor"] for item in reports.values()),
        "files": reports,
        "scope": "cpu_safetensors_exponent_scan_only",
        "training_executed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", default="foundation-download-receipt.json")
    args = parser.parse_args(argv)
    result = audit(args.root, args.output, args.receipt)
    print("FOUNDATION_FINITE_PASS" if result["finite"] else "FOUNDATION_FINITE_FAIL", flush=True)
    if not result["finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
