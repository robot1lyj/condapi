"""Inspect a safetensors header without loading weights or using CUDA."""

import argparse
from collections import Counter
import json
from pathlib import Path
import struct


def inspect(path):
    path = Path(path)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Incomplete header length")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 0 < header_size <= 64 * 1024 * 1024:
            raise ValueError("Invalid or excessive header size")
        header = json.loads(stream.read(header_size))
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    end = max(value["data_offsets"][1] for value in tensors.values())
    expected = 8 + header_size + end
    actual = path.stat().st_size
    return {
        "path": str(path),
        "tensor_count": len(tensors),
        "dtype_tensor_counts": dict(Counter(value["dtype"] for value in tensors.values())),
        "actual_bytes": actual,
        "expected_bytes": expected,
        "size_complete": actual == expected,
        "finite_values_checked": False,
        "sha256_checked": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    print(json.dumps(inspect(parser.parse_args().path), indent=2))
