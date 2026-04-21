#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import shutil
from pathlib import Path


def copy_tree(src: Path, dst: Path) -> None:
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay openpi transformers patches into site-packages.")
    parser.add_argument("--openpi-dir", type=Path, required=True, help="Path to the local openpi repository.")
    args = parser.parse_args()

    openpi_dir = args.openpi_dir.resolve()
    patch_dir = openpi_dir / "src" / "openpi" / "models_pytorch" / "transformers_replace"
    if not patch_dir.is_dir():
        raise FileNotFoundError(f"patch directory not found: {patch_dir}")

    transformers = importlib.import_module("transformers")
    target_dir = Path(transformers.__file__).resolve().parent
    copy_tree(patch_dir, target_dir)
    print(f"[ok] patched transformers in {target_dir}")


if __name__ == "__main__":
    main()
