"""Prepare the pinned upstream conversion inside the Thor Pi container."""

import argparse
from pathlib import Path
import runpy
import sys

from transformers import GemmaTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer_dir = args.output / "tokenizer"
    tokenizer = GemmaTokenizer(vocab_file=str(args.tokenizer_model), add_bos_token=True, add_eos_token=False)
    tokenizer.save_pretrained(tokenizer_dir)
    sys.argv = [
        str(args.source / "convert_from_jax_pi05.py"),
        "--jax_path",
        str(args.checkpoint),
        "--output",
        str(args.output / "converted.pkl"),
        "--prompt",
        "pick up the object",
        "--tokenizer_path",
        str(tokenizer_dir),
    ]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
