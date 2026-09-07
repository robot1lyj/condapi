"""Experiment-only relaxation of Inductor's desktop SM-count heuristic.

The NVIDIA 26.05 PyTorch utils.is_big_gpu heuristic excludes CUDA GPUs below
68 SMs. A single SM110 Thor can still compile Triton GEMMs. Keep ATen choices
available and let measured autotuning select; do not spoof hardware properties.
This is a private API experiment, not a production runtime default.
"""

import hashlib
import inspect

import torch


def enable_thor_triton_autotune():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Experiment requires one CUDA Thor GPU")
    if torch.cuda.get_device_capability() != (11, 0) or "Thor" not in torch.cuda.get_device_name():
        raise RuntimeError("Do not apply the Thor heuristic experiment to other GPUs")
    import torch._inductor.config as config  # noqa: PLC0415
    import torch._inductor.utils as utils  # noqa: PLC0415

    if not hasattr(utils, "is_big_gpu"):
        raise RuntimeError("Inductor heuristic changed; review the experiment first")
    original = utils.is_big_gpu
    record = {
        "original_eligible": original(torch.device("cuda", 0)),
        "actual_sms": torch.cuda.get_device_properties(0).multi_processor_count,
        "original_heuristic_sha256": hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
        "original_backends": config.max_autotune_gemm_backends,
        "candidate_backends": "ATEN,TRITON",
        "scope": "single SM110 Thor; benchmark process only; hardware properties unchanged",
    }
    utils.is_big_gpu = lambda index_or_device=0: True
    config.max_autotune_gemm_backends = "ATEN,TRITON"
    return record
