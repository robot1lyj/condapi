"""Publish a frozen RLT token plus zero/learned actors; no ML imports."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.parts.build_behavior_manifest import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--base-identity", type=Path, required=True)
    parser.add_argument("--token-manifest", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path)
    source.add_argument("--zero-residual", action="store_true")
    parser.add_argument("--exploration-std", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.output,
                json.loads(args.contract.read_text()),
                json.loads(args.base_identity.read_text()),
                snapshot=args.snapshot,
                token_manifest=args.token_manifest,
                exploration_std=args.exploration_std,
                seed=args.seed,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
