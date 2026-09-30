"""RLT actor/critic entry: reuse the server replay/snapshot training orchestrator."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adapters.parts.train import main

if __name__ == "__main__":
    main(backend="rlt")
