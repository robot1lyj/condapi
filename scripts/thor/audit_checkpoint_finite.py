"""Read original JAX parameters without casting and report finite-value counts."""

import argparse
import hashlib
import json
from pathlib import Path

from flax.nnx import traversals
import numpy as np

from openpi.models.model import restore_params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    params = restore_params(str(args.checkpoint / "params"), restore_type=np.ndarray, dtype=None)
    rows = []
    for name, leaf in traversals.flatten_mapping(params, sep="/").items():
        value = np.asarray(leaf)
        row = {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "elements": value.size,
            "nan": int(np.isnan(value).sum()),
            "inf": int(np.isinf(value).sum()),
        }
        rows.append(row)
        if row["nan"] or row["inf"] or "input_embedding" in name:
            print(json.dumps(row), flush=True)
    with (args.checkpoint / "params/_METADATA").open("rb") as stream:
        identity = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "checkpoint": str(args.checkpoint),
        "metadata_sha256": identity,
        "tensors": rows,
        "nonfinite_elements": sum(row["nan"] + row["inf"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print("NONFINITE_ELEMENTS", report["nonfinite_elements"], flush=True)


if __name__ == "__main__":
    main()
