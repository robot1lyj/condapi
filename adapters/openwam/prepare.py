"""Audit selected YAM training episodes and write native normalization inputs to a new path."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adapters.openwam.common import activate_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True, help="JSON list of training episode ids")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    activate_source()
    from adapters.openwam.data import prepare_stats  # noqa: PLC0415

    prepare_stats(args.dataset, json.loads(args.episodes.read_text()), args.output)


if __name__ == "__main__":
    main()
