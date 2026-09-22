"""Wait for a verified download, then scan BF16 weights on CPU only."""

import argparse
import json
from pathlib import Path
import struct
import time

import numpy as np


def scan(path):
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < header_size <= 64 * 1024 * 1024:
            raise ValueError("Invalid header size")
        header = json.loads(stream.read(header_size))
        results = {}
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if tensor["dtype"] != "BF16":
                raise ValueError(f"Unsupported dtype: {name}: {tensor['dtype']}")
            start, end = tensor["data_offsets"]
            if start < 0 or end < start or (end - start) % 2:
                raise ValueError(f"Invalid offsets: {name}")
            stream.seek(8 + header_size + start)
            remaining = end - start
            count = 0
            while remaining:
                block = stream.read(min(remaining, 8 * 1024 * 1024))
                if not block:
                    raise ValueError("Truncated weights")
                bits = np.frombuffer(block, dtype="<u2")
                count += int(np.count_nonzero((bits & 0x7F80) == 0x7F80))
                remaining -= len(block)
            if count:
                results[name] = count
    return {"tensor_count": len(header) - int("__metadata__" in header), "nonfinite_by_tensor": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    receipt = args.root / "thor-download-receipt.json"
    deadline = time.monotonic() + args.wait_seconds
    while not receipt.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Verified download receipt not yet available")
        time.sleep(30)
    data = json.loads(receipt.read_text())
    reports = {}
    for entry in data["files"]:
        if entry["file"].endswith(".safetensors"):
            path = args.root / entry["file"]
            if path.stat().st_size != entry["size"]:
                raise ValueError("File size changed after download verification")
            reports[entry["file"]] = scan(path)
    if not reports:
        raise ValueError("No safetensors weights in receipt")
    passed = all(not report["nonfinite_by_tensor"] for report in reports.values())
    args.output.write_text(
        json.dumps({"revision": data["revision"], "finite": passed, "files": reports}, indent=2) + "\n"
    )
    print("FINITE_AUDIT_PASS" if passed else "FINITE_AUDIT_FAIL", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
