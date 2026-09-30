# ruff: noqa: E402
# Standalone source checkout entry point.
"""Assemble queue-aware autonomous replay; no training or model inference."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/vla-platform/src"))
sys.path.insert(0, str(ROOT / "packages/parts-rl/src"))

from parts_rl.prepare import prepare_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    args = parser.parse_args()
    result = prepare_runs(
        args.run,
        args.output,
        json.loads(args.contract.read_text()),
        json.loads(args.reward.read_text()),
        json.loads(args.split.read_text()),
        args.gamma,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
