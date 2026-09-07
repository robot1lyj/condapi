"""Generate pip constraints for the known-working NVIDIA GPU stack."""

import argparse
import importlib.metadata
from pathlib import Path

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("jax", "pytorch"), default="jax")
    args = parser.parse_args()
    if args.backend == "jax":
        constraints = [f"{name}=={importlib.metadata.version(name)}" for name in PACKAGES]
    else:
        constraints = []
        for package in importlib.metadata.distributions():
            name = package.metadata["Name"].lower().replace("_", "-")
            if name in {"torch", "torchvision", "torchaudio", "triton", "numpy", "scipy"} or name.startswith(
                ("nvidia-", "tensorrt", "torch-tensorrt", "transformer-engine")
            ):
                constraints.append(f"{name}=={package.version}")
        if not any(value.startswith("torch==") for value in constraints):
            raise RuntimeError("NVIDIA PyTorch must already be installed")
    args.output.write_text("\n".join(sorted(constraints)) + "\n")
