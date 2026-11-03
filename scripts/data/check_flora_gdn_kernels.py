"""Thor-only inference microchecks: native HF GDN vs isolated FLA/causal-conv.

No model weights, training, global monkeypatches or precision changes. Run in a
fresh candidate process under the shared Flora GPU lock. Draft BF16 tolerances
are diagnostic only; passing cannot certify complete Flora semantic parity.
"""

# Every loop closure is consumed synchronously by check()/benchmark() before the
# iteration advances; no callable escapes or runs asynchronously.
# ruff: noqa: PLC0415, B023
import argparse
from datetime import UTC
from datetime import datetime
import hashlib
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

EXPECTED_TORCH = "2.12.0a0+5aff3928d8.nv26.05"
EXPECTED_TRITON_DIST = "3.7.0+gitb4e20bb.nv26.5"
MEMORY_LIMIT = 1024**3


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_report(path, report):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def tensor_identity(tensor):
    import torch

    value = tensor.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def compare_tensors(reference, candidate, atol, rtol):
    import torch

    result = {
        "reference": tensor_identity(reference),
        "candidate": tensor_identity(candidate),
        "shape_equal": reference.shape == candidate.shape,
        "dtype_equal": reference.dtype == candidate.dtype,
    }
    if not result["shape_equal"]:
        return dict(result, finite=False, draft_allclose=False)
    ref, cand = reference.detach().float().cpu(), candidate.detach().float().cpu()
    finite = bool(torch.isfinite(ref).all() and torch.isfinite(cand).all())
    result.update(
        finite=finite,
        bitwise_equal=bool(torch.equal(reference, candidate)),
        draft_allclose=finite and result["dtype_equal"] and bool(torch.allclose(ref, cand, atol=atol, rtol=rtol)),
    )
    if finite:
        difference = cand.double() - ref.double()
        rms = float(difference.square().mean().sqrt())
        reference_rms = float(ref.double().square().mean().sqrt())
        result.update(
            max_abs_diff=float(difference.abs().max()),
            rms_diff=rms,
            reference_rms=reference_rms,
            relative_rms_diff=rms / max(reference_rms, 1e-12),
        )
    return result


def benchmark(function, warmup, repeats):
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    first = function()
    torch.cuda.synchronize()
    first_wall = time.perf_counter() - started
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    walls, gpu_ms = [], []
    for _ in range(repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        begin.record()
        function()
        end.record()
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - started)
        gpu_ms.append(begin.elapsed_time(end))
    return first, {
        "first_call_wall_s_includes_possible_jit": first_wall,
        "warm_wall_s": walls,
        "warm_cuda_event_ms": gpu_ms,
        "warm_wall_median_s": statistics.median(walls),
        "warm_cuda_event_median_ms": statistics.median(gpu_ms),
    }


def require_signature(function, names):
    signature = inspect.signature(function)
    if not set(names).issubset(signature.parameters):
        raise RuntimeError(f"Unexpected API: {function.__module__}.{function.__name__}{signature}")
    source = inspect.getsourcefile(inspect.unwrap(function))
    return {
        "signature": str(signature),
        "module": function.__module__,
        "source_path": source,
        "source_sha256": sha_file(source) if source else None,
    }


def run_checks(args, report):
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d import causal_conv1d_update
    import causal_conv1d_cuda
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    import torch
    import torch.nn.functional as functional
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    import triton

    versions = {
        name: metadata.version(name)
        for name in ("torch", "triton", "transformers", "fla-core", "flash-linear-attention", "causal-conv1d")
    }
    report["runtime"] = {
        "versions": versions,
        "torch_module_version": torch.__version__,
        "triton_module_version": triton.__version__,
        "torch_file": torch.__file__,
        "triton_file": triton.__file__,
        "cuda": torch.version.cuda,
        "cxx11_abi": torch.compiled_with_cxx11_abi(),
        "causal_cuda_so": causal_conv1d_cuda.__file__,
        "causal_cuda_so_sha256": sha_file(causal_conv1d_cuda.__file__),
        "torch_num_threads": torch.get_num_threads(),
    }
    for package, version in [
        ("transformers", "5.8.1"),
        ("fla-core", "0.6.0"),
        ("flash-linear-attention", "0.6.0"),
        ("causal-conv1d", "1.7.0"),
    ]:
        if versions[package].split("+")[0] != version:
            raise RuntimeError(f"Expected {package}=={version}; found {versions[package]}")
    if torch.__version__ != EXPECTED_TORCH or versions["triton"] != EXPECTED_TRITON_DIST:
        raise RuntimeError("This check is bound to the recorded NGC Torch/Triton stack")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("Requires the Thor sm110 CUDA device")
    if not native.is_fast_path_available:
        raise RuntimeError("HF fast-path imports incomplete; candidate overlay is not active")
    report["runtime"].update(
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        allow_tf32_cudnn=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )
    functions = {
        "hf_chunk": (
            native.torch_chunk_gated_delta_rule,
            ["query", "key", "value", "g", "beta", "initial_state", "output_final_state"],
        ),
        "fla_chunk": (chunk_gated_delta_rule, ["q", "k", "v", "g", "beta", "initial_state", "output_final_state"]),
        "hf_recurrent": (
            native.torch_recurrent_gated_delta_rule,
            ["query", "key", "value", "g", "beta", "initial_state"],
        ),
        "fla_recurrent": (fused_recurrent_gated_delta_rule, ["q", "k", "v", "g", "beta", "initial_state"]),
        "hf_conv_update": (native.torch_causal_conv1d_update, ["hidden_states", "conv_state", "weight", "activation"]),
        "cuda_conv_full": (causal_conv1d_fn, ["x", "weight", "activation", "seq_idx"]),
        "cuda_conv_update": (causal_conv1d_update, ["x", "conv_state", "weight", "activation"]),
        "hf_norm": (native.Qwen3_5RMSNormGated.forward, ["hidden_states", "gate"]),
        "fused_norm_init": (FusedRMSNormGated.__init__, ["hidden_size", "eps", "activation", "device", "dtype"]),
        "fused_norm": (FusedRMSNormGated.forward, ["x", "g"]),
    }
    report["api_sources"] = {name: require_signature(function, names) for name, (function, names) in functions.items()}
    save_report(args.output, report)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    dtype, batch, heads, dimension = torch.bfloat16, 1, 2, 128
    report["input_generator"] = {
        "device": "cpu",
        "seed": args.seed,
        "random_base_dtype": "torch.float32",
        "compute_dtype": "torch.bfloat16",
        "g_state_parameter_dtype": "torch.float32",
        "batch": batch,
        "heads": heads,
        "key_dim": dimension,
        "value_dim": dimension,
    }
    torch.cuda.reset_peak_memory_stats()

    def cpu_random(shape, *, target_dtype=dtype, scale=1.0):
        return (torch.randn(shape, generator=generator, dtype=torch.float32) * scale).to(target_dtype)

    def inputs_to_cuda(values):
        identities = {name: tensor_identity(value) for name, value in values.items()}
        return {name: value.cuda() for name, value in values.items()}, identities

    def check(name, reference, candidate, inputs, *, extra=None):
        reference_result, reference_timing = benchmark(reference, args.warmup, args.repeats)
        candidate_result, candidate_timing = benchmark(candidate, args.warmup, args.repeats)
        ref = reference_result if isinstance(reference_result, tuple) else (reference_result,)
        cand = candidate_result if isinstance(candidate_result, tuple) else (candidate_result,)
        if len(ref) != len(cand):
            raise RuntimeError("Output arity mismatch: " + name)
        metrics = [compare_tensors(a, b, args.atol, args.rtol) for a, b in zip(ref, cand, strict=True)]
        peak = {
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        record = {
            "name": name,
            "inputs": inputs,
            "outputs": metrics,
            "reference_timing": reference_timing,
            "candidate_timing": candidate_timing,
            "draft_pass": all(m["draft_allclose"] for m in metrics),
            "memory": peak,
            "extra": extra or {},
        }
        report["cases"].append(record)
        report["memory"] = peak
        save_report(args.output, report)
        print(json.dumps({"case": name, "draft_pass": record["draft_pass"], "memory": peak}), flush=True)
        if max(peak.values()) >= MEMORY_LIMIT:
            raise RuntimeError("Microcheck exceeded the 1GiB PyTorch allocator memory budget")
        return reference_result, candidate_result

    with torch.inference_mode():
        for length in args.lengths:
            alog = cpu_random((heads,), target_dtype=torch.float32, scale=0.2)
            dtbias = cpu_random((heads,), target_dtype=torch.float32, scale=0.1)
            a = cpu_random((batch, length, heads), target_dtype=torch.float32, scale=0.2)
            b = cpu_random((batch, length, heads))
            cpu = {
                "q": cpu_random((batch, length, heads, dimension)),
                "k": cpu_random((batch, length, heads, dimension)),
                "v": cpu_random((batch, length, heads, dimension)),
                "beta": b.sigmoid(),
                "g": -alog.exp() * functional.softplus(a + dtbias),
                "initial_state": cpu_random(
                    (batch, heads, dimension, dimension), target_dtype=torch.float32, scale=0.05
                ),
                "A_log": alog,
                "dt_bias": dtbias,
            }
            values, identities = inputs_to_cuda(cpu)
            q, k, v, g, beta = (values[key] for key in ("q", "k", "v", "g", "beta"))
            kwargs = {"g": g, "beta": beta, "output_final_state": True, "use_qk_l2norm_in_kernel": True}
            check(
                f"gdn_prefill_zero_state_T{length}",
                lambda: native.torch_chunk_gated_delta_rule(q, k, v, initial_state=None, **kwargs),
                lambda: chunk_gated_delta_rule(q, k, v, initial_state=None, **kwargs),
                identities,
            )
            ref_prefill, cand_prefill = check(
                f"gdn_prefill_fp32_state_T{length}",
                lambda: native.torch_chunk_gated_delta_rule(q, k, v, initial_state=values["initial_state"], **kwargs),
                lambda: chunk_gated_delta_rule(q, k, v, initial_state=values["initial_state"], **kwargs),
                identities,
            )
            decode_cpu = {
                "q": cpu_random((batch, 1, heads, dimension)),
                "k": cpu_random((batch, 1, heads, dimension)),
                "v": cpu_random((batch, 1, heads, dimension)),
                "beta": cpu_random((batch, 1, heads)).sigmoid(),
                "g": -functional.softplus(cpu_random((batch, 1, heads), target_dtype=torch.float32)),
            }
            decode, decode_ids = inputs_to_cuda(decode_cpu)
            dq, dk, dv = (decode[key] for key in ("q", "k", "v"))
            decode_kwargs = {
                "g": decode["g"],
                "beta": decode["beta"],
                "output_final_state": True,
                "use_qk_l2norm_in_kernel": True,
            }
            check(
                f"gdn_recurrent_common_fp32_state_T{length}",
                lambda: native.torch_recurrent_gated_delta_rule(
                    dq, dk, dv, initial_state=values["initial_state"], **decode_kwargs
                ),
                lambda: fused_recurrent_gated_delta_rule(
                    dq, dk, dv, initial_state=values["initial_state"], **decode_kwargs
                ),
                dict(decode_ids, initial_state=identities["initial_state"]),
            )
            check(
                f"gdn_recurrent_after_prefill_T{length}",
                lambda: native.torch_recurrent_gated_delta_rule(
                    dq, dk, dv, initial_state=ref_prefill[1], **decode_kwargs
                ),
                lambda: fused_recurrent_gated_delta_rule(dq, dk, dv, initial_state=cand_prefill[1], **decode_kwargs),
                decode_ids,
                extra={"state_origin": "each branch's own immediately preceding prefill final state"},
            )

            channels, width = heads * dimension * 3, 4
            convolution, conv_ids = inputs_to_cuda(
                {
                    "x": cpu_random((batch, channels, length)),
                    "weight": cpu_random((channels, width), scale=0.15),
                    "decode_x": cpu_random((batch, channels, 1)),
                }
            )
            x, weight = convolution["x"], convolution["weight"]
            check(
                f"causal_conv_full_T{length}",
                lambda: functional.silu(
                    functional.conv1d(x, weight.unsqueeze(1), padding=width - 1, groups=channels)[:, :, :length]
                ),
                lambda: causal_conv1d_fn(x=x, weight=weight, bias=None, activation="silu", seq_idx=None),
                conv_ids,
                extra={"reference": "exact HF 5.8.1 full-prefill expression; no bias"},
            )
            state = functional.pad(x, (width - length, 0)).contiguous()

            def ref_update():
                copied = state.clone()
                output = native.torch_causal_conv1d_update(
                    convolution["decode_x"], copied, weight, bias=None, activation="silu"
                )
                return output, copied

            def cand_update():
                copied = state.clone()
                output = causal_conv1d_update(convolution["decode_x"], copied, weight, bias=None, activation="silu")
                return output, copied

            check(
                f"causal_conv_update_after_prefill_T{length}",
                ref_update,
                cand_update,
                dict(conv_ids, conv_state=tensor_identity(state)),
                extra={"state_clone_per_timed_call_included": True},
            )

            normalization, norm_ids = inputs_to_cuda(
                {
                    "x": cpu_random((batch, length, heads, dimension)),
                    "gate": cpu_random((batch, length, heads, dimension)),
                    "weight": (1 + cpu_random((dimension,), target_dtype=torch.float32, scale=0.1)).to(dtype),
                }
            )
            native_norm = native.Qwen3_5RMSNormGated(dimension, eps=1e-6).to(device="cuda", dtype=dtype).eval()
            fused_norm = FusedRMSNormGated(dimension, eps=1e-6, activation="silu", device="cuda", dtype=dtype).eval()
            native_norm.weight.copy_(normalization["weight"])
            fused_norm.weight.copy_(normalization["weight"])
            native_norm.requires_grad_(requires_grad=False)
            fused_norm.requires_grad_(requires_grad=False)
            check(
                f"gated_rmsnorm_T{length}",
                lambda: native_norm(normalization["x"], gate=normalization["gate"]),
                lambda: fused_norm(normalization["x"], normalization["gate"]),
                norm_ids,
                extra={"eps": 1e-6, "activation": "silu", "weight_dtype": str(dtype)},
            )
            torch.cuda.synchronize()
    report["status"] = (
        "DRAFT_KERNEL_NUMERIC_CHECKS_PASS_NOT_FLORA_PARITY"
        if all(case["draft_pass"] for case in report["cases"])
        else "DRAFT_KERNEL_NUMERIC_DIFFERENCES_NOT_FLORA_PARITY"
    )
    report["completed_utc"] = datetime.now(UTC).isoformat()
    return 0 if all(case["draft_pass"] for case in report["cases"]) else 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="NEW JSON file; existing reports are never overwritten"
    )
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 63, 64, 65, 128])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--atol", type=float, default=0.02, help="Draft BF16 operator tolerance; does not authorize semantic changes"
    )
    parser.add_argument(
        "--rtol", type=float, default=0.02, help="Draft BF16 operator tolerance; does not authorize semantic changes"
    )
    args = parser.parse_args(argv)
    if platform.machine() not in ("aarch64", "arm64"):
        parser.error("Thor ARM64 only; local workstation execution is unsupported")
    if args.output.exists():
        parser.error("Use a new output path; existing evidence is preserved")
    if (
        len(args.lengths) > 8
        or len(set(args.lengths)) != len(args.lengths)
        or any(not 1 <= n <= 256 for n in args.lengths)
    ):
        parser.error("Use at most 8 unique sequence lengths in 1..256")
    if (
        not 0 <= args.warmup <= 10
        or not 1 <= args.repeats <= 30
        or not 0 <= args.atol <= 0.1
        or not 0 <= args.rtol <= 0.1
    ):
        parser.error("Warmup0..10, repeats1..30, finite draft tolerances0..0.1 required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation claims this report name before imports or GPU work.
    with args.output.open("x") as stream:
        stream.write("{}\n")
    report = {
        "status": "RUNNING_NOT_VALIDATED",
        "started_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "machine": platform.machine(),
        "script_sha256": sha_file(__file__),
        "contract": {
            "inference_only": True,
            "model_weights_loaded": False,
            "training": False,
            "reference": "Transformers5.8.1 explicit torch_* functions; no global modifications",
            "candidate": "FLA0.6.0 and causal-conv1d1.7.0 explicit functions",
            "draft_atol": args.atol,
            "draft_rtol": args.rtol,
            "seed": args.seed,
            "lengths": args.lengths,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "precision": "existing runtime flags retained; BF16 q/k/v/beta; FP32 g/initial_state/A_log/dt_bias",
            "memory_budget_bytes": MEMORY_LIMIT,
            "memory_scope": "PyTorch peak allocated/reserved; excludes driver context, host JIT compiler and other processes",
            "semantic_parity": "NOT_TESTED; complete Flora fixed-input greedy A/B still required",
        },
        "cases": [],
    }
    save_report(args.output, report)
    try:
        exit_code = run_checks(args, report)
    except Exception as error:
        report.update(
            status="ERROR_INCOMPLETE_NO_PASS_CLAIM",
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        )
        save_report(args.output, report)
        raise
    save_report(args.output, report)
    print(
        json.dumps({"status": report["status"], "output": str(args.output), "cases": len(report["cases"])}), flush=True
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
