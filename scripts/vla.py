"""Source-checkout bootstrap; installing yam-vla-platform provides `vla` instead."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/vla-platform/src"))

from vla_platform.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
