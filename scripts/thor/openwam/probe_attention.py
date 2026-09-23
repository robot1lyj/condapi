"""Thor GPU probe for Alpha-style masked joint attention; no checkpoint loaded.

This checks backend support and single-operator latency. It is not a full-model
benchmark or an accuracy gate for robot actions.
"""

import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend
from torch.nn.attention import sdpa_kernel


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    # Representative 513-token joint sequence from the r7 profiler, 24 heads.
    # This synthetic mask tests custom-mask support; it is not the trained mask.
    q = torch.randn((1, 24, 513, 128), device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    mask = torch.ones((513, 513), device=device, dtype=torch.bool)
    mask[:64, 64:] = False
    modes = {
        "auto": None,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "flash": SDPBackend.FLASH_ATTENTION,
    }
    result = {"scope": "synthetic_mask_operator_probe_not_model_accuracy", "shape": list(q.shape), "results": []}
    reference = None
    for name, backend in modes.items():
        try:

            def call(selected_backend=backend) -> torch.Tensor:
                if selected_backend is None:
                    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                with sdpa_kernel(selected_backend):
                    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

            for _ in range(5):
                output = call()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            times = []
            for _ in range(30):
                start.record()
                output = call()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            output = output.float()
            if not torch.isfinite(output).all():
                raise ValueError("nonfinite output")
            if reference is None:
                reference = output.clone()
            difference = (output - reference).abs()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as profiler:
                call()
                torch.cuda.synchronize()
            kernels = sorted({event.key for event in profiler.key_averages() if "attention" in event.key.lower()})
            row = {
                "backend": name,
                "p50_ms": float(torch.tensor(times).quantile(0.50)),
                "p95_ms": float(torch.tensor(times).quantile(0.95)),
                "max_abs_vs_auto": float(difference.max()),
                "mae_vs_auto": float(difference.mean()),
                "attention_ops": kernels,
            }
        except Exception as exc:
            row = {"backend": name, "error": str(exc)}
        result["results"].append(row)
        print(json.dumps(row), flush=True)
    Path("/reports/attention-probe.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
