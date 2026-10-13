"""Currently selected training locations; no implicit device fallback."""

import os
from pathlib import Path
import platform


def require_training_runtime(execution, *, cuda_available, capability=None):
    if not cuda_available:
        raise RuntimeError("Training requires the explicitly allocated CUDA device")
    if execution == "slurm":
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Slurm allocation required for this execution target")
        return {"target": execution, "job": os.environ["SLURM_JOB_ID"]}
    if execution != "thor-offline":
        raise ValueError("unknown training execution target")
    if (platform.machine() != "aarch64" or not Path("/.dockerenv").is_file()
            or os.environ.get("YAM_RL_TRAINING_TARGET") != "thor-offline"
            or tuple(capability or ()) != (11, 0)):
        raise RuntimeError("Thor training requires the dedicated ARM64 SM110 offline RL container")
    return {"target": execution, "job": None, "container": os.environ.get("HOSTNAME"),
            "machine": platform.machine(), "device_capability": list(capability)}
