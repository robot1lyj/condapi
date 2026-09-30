"""Check client package hashes; this does not create training READY."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/vla-platform/src"))
from vla_platform.parts import audit_publication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--allow-mock", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit_publication(args.run, allow_mock=args.allow_mock), ensure_ascii=False))


if __name__ == "__main__":
    main()
