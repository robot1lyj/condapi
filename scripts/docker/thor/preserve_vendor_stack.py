"""Generate pip constraints for the known-working NVIDIA GPU stack."""

import importlib.metadata
from pathlib import Path
import sys

PACKAGES = (
    "jax",
    "jaxlib",
    "jax-cuda13-pjrt",
    "jax-cuda13-plugin",
    "flax",
    "orbax-checkpoint",
    "numpy",
    "scipy",
    "optax",
    "tensorstore",
    "ml_dtypes",
)


if __name__ == "__main__":
    Path(sys.argv[1]).write_text("\n".join(f"{name}=={importlib.metadata.version(name)}" for name in PACKAGES) + "\n")
