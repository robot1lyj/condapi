from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT,
    ROOT / "src",
    ROOT / "scripts/thor",
    ROOT / "packages/parts-rl/src",
    ROOT / "packages/vla-platform/src",
):
    sys.path.insert(0, str(path))
