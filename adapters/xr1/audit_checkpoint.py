"""Read-only finite-value audit of trusted native XR-1/DeepSpeed checkpoints."""

import argparse
import json
from pathlib import Path

import torch


def tensors(value, name=""):
    if isinstance(value, torch.Tensor):
        yield name, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, f"{name}/{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensors(item, f"{name}/{index}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-only", action="store_true")
    args = parser.parse_args()
    files = sorted(args.checkpoint.rglob("*model_states.pt"))
    if not args.model_only:
        files += sorted(args.checkpoint.rglob("*optim_states.pt"))
    if not files or args.output.exists():
        raise ValueError("Missing checkpoint or output already exists")
    report = {"checkpoint": str(args.checkpoint.resolve()), "model_only": args.model_only, "files": []}
    for path in files:
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        row = {"file": str(path), "bad_tensors": [], "checked_elements": 0,
               "global_step": state.get("global_step"), "engine_global_steps": state.get("global_steps")}
        for name, value in tensors(state):
            if not value.is_floating_point():
                continue
            count = 0
            for chunk in value.detach().reshape(-1).split(32 * 1024 * 1024):
                count += int((~torch.isfinite(chunk)).sum())
            row["checked_elements"] += value.numel()
            if count:
                row["bad_tensors"].append({"name": name, "dtype": str(value.dtype), "shape": list(value.shape), "nonfinite": count})
        report["files"].append(row)
        print(json.dumps(row), flush=True)
        del state
    report["nonfinite_elements"] = sum(t["nonfinite"] for f in report["files"] for t in f["bad_tensors"])
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    if report["nonfinite_elements"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
