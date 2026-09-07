"""Run the same recorded suite with a native JAX or converted PyTorch policy."""

import argparse
from collections import Counter
import dataclasses
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

from benchmark_pi05 import digest
from benchmark_pi05 import read_observation
import jax
import numpy as np
import torch

from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config


def checked_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Suite paths must stay inside the fixture directory")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--params-dtype", choices=("checkpoint", "float32", "bfloat16"), required=True)
    parser.add_argument("--compute-dtype", choices=("float32", "bfloat16"), required=True)
    parser.add_argument("--backend", choices=("jax", "pytorch"), default="jax")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--batch-vision", action="store_true")
    parser.add_argument("--native-attention-mask", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--compile-graph-parts", action="store_true")
    parser.add_argument("--thor-triton-autotune", action="store_true")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2 or args.steps <= 0 or args.warmups < 2:
        parser.error("Need at least two repeats and positive denoising steps")
    if args.output.exists():
        parser.error("Use a new run directory; existing experiments are immutable")
    is_pytorch = args.backend == "pytorch"
    if args.cuda_graph and (not is_pytorch or args.compile):
        parser.error("CUDA graph experiment requires PyTorch --no-compile")
    if args.compile_graph_parts and not args.cuda_graph:
        parser.error("Compiled graph parts require --cuda-graph")
    if args.thor_triton_autotune and (not is_pytorch or not (args.compile or args.compile_graph_parts)):
        parser.error("Thor Triton experiment requires a compiled PyTorch backend")
    weight_path = args.checkpoint / "model.safetensors"
    if weight_path.exists() != is_pytorch:
        parser.error("Checkpoint format does not match the selected backend")
    suite = json.loads(args.suite.read_text())
    root = args.suite.parent
    norm_path = checked_path(root, suite["norm_stats"])
    if suite.get("source_kind") != "real_yam_recording" or not suite.get("source_files"):
        parser.error("Real recorded source evidence required")
    if digest(norm_path) != suite["norm_stats_sha256"]:
        parser.error("Norm fingerprint mismatch")
    samples = []
    for entry in suite["samples"]:
        path = checked_path(root, entry["sample"])
        provenance_path = checked_path(root, entry["provenance"])
        provenance = json.loads(provenance_path.read_text())
        if (
            provenance.get("source_kind") != "real_yam_recording"
            or provenance.get("sample_sha256") != digest(path)
            or provenance.get("norm_stats_sha256") != digest(norm_path)
            or provenance.get("source_files") != suite["source_files"]
        ):
            parser.error("Sample provenance mismatch")
        samples.append((path, read_observation(path), provenance))
    if len(samples) != suite["observation_count"] or not samples:
        parser.error("Suite size mismatch")
    if is_pytorch:
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA execution is required; no CPU fallback")
        if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
            raise RuntimeError("Unset the vendor TF32 override before precision comparisons")
        conversion = json.loads((args.checkpoint / "conversion_audit.json").read_text())
        if conversion.get("output_precision") != "float32":
            raise ValueError("Start from the audited FP32 conversion; select compute precision explicitly")
        metadata_hash = conversion["checkpoint_metadata_sha256"]
        devices = [torch.cuda.get_device_name(), str(torch.cuda.get_device_capability())]
        packages = ("torch", "transformers", "jax", "jaxlib", "flax", "orbax-checkpoint")
    else:
        if any(d.platform != "gpu" for d in jax.devices()):
            raise RuntimeError("GPU execution is required; no CPU fallback")
        metadata_hash = digest(args.checkpoint / "params" / "_METADATA")
        devices = [str(d) for d in jax.devices()]
        packages = ("jax", "jaxlib", "flax", "orbax-checkpoint")
    jax.config.update("jax_default_matmul_precision", "highest")
    train_config = config.get_config("pi05_yam")
    train_config = dataclasses.replace(
        train_config, model=dataclasses.replace(train_config.model, dtype=args.compute_dtype)
    )
    args.output.mkdir(parents=True)
    record = {
        "run_id": args.output.name,
        "phase": "evaluate",
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "status": "running",
        "command": sys.argv,
        "config": "pi05_yam",
        "backend": args.backend,
        "compiled": args.compile if is_pytorch else True,
        "cuda_graph": args.cuda_graph,
        "compiled_graph_parts": args.compile_graph_parts,
        "graph_reference": "uncaptured_legacy_loop_same_kernel_backend" if args.cuda_graph else None,
        "static_denoising_loop": args.cuda_graph,
        "warmups_per_sample": args.warmups,
        "attention": args.attention if is_pytorch else "jax_native",
        "batch_vision": args.batch_vision if is_pytorch else False,
        "attention_mask": "query_dtype" if is_pytorch and args.native_attention_mask else "float32",
        "params_dtype": f"fp32_checkpoint_to_{args.compute_dtype}" if is_pytorch else args.params_dtype,
        "compute_dtype": args.compute_dtype,
        "matmul_precision": str(jax.config.jax_default_matmul_precision),
        "checkpoint_metadata_sha256": metadata_hash,
        "suite_sha256": digest(args.suite),
        "norm_stats_sha256": digest(norm_path),
        "suite": suite,
        "steps": args.steps,
        "horizon": 50,
        "seed": args.seed,
        "repeats": args.repeats,
        "versions": {p: importlib.metadata.version(p) for p in packages},
        "devices": devices,
        "precision_acceptance_threshold": None,
        "limitations": [
            "Base model is not YAM-finetuned; no robot execution or task success claim",
            "Benchmark-only norms from the same recordings; not production training norms",
            "Dataset units unchanged, not independently calibrated against hardware",
            "No numerical tolerance approved yet; measured differences do not constitute acceptance",
            "Same Thor JAX version reference, not a training-server JAX equivalence test",
        ],
    }
    if is_pytorch:
        record.update(
            converted_weights_sha256=digest(weight_path),
            conversion_audit=conversion,
            precision_policy="selected BF16 backbone with FP32 stability layers"
            if args.compute_dtype == "bfloat16"
            else "FP32 / TF32 disabled",
        )
    if args.thor_triton_autotune:
        from thor_triton_autotune import enable_thor_triton_autotune  # noqa: PLC0415

        record["thor_triton_autotune"] = enable_thor_triton_autotune()
    (args.output / "started.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print("MODEL_LOAD_START", flush=True)
    started = time.monotonic()
    backend_options = (
        {"pytorch_device": "cuda", "pytorch_precision": args.compute_dtype, "pytorch_compile": args.compile}
        if is_pytorch
        else {}
    )
    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        norm_stats=normalize.deserialize_json(norm_path.read_text()),
        sample_kwargs={"num_steps": args.steps},
        jax_param_dtype=args.params_dtype,
        **backend_options,
    )
    record["load_s"] = time.monotonic() - started
    if is_pytorch:
        policy._model.attention_implementation = args.attention  # noqa: SLF001
        policy._model.batch_vision = args.batch_vision  # noqa: SLF001
        if args.native_attention_mask:
            policy._model.attention_mask_dtype = (  # noqa: SLF001
                policy._model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype  # noqa: SLF001
            )
    # Inspect actual loaded leaves, not only the requested restoration dtype.
    if is_pytorch:
        # The legacy constructor requests "high" matmul; override AFTER load.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        leaves = list(policy._model.parameters())  # noqa: SLF001
        if not all(x.is_cuda for x in leaves):
            raise RuntimeError("Model parameters are not all on CUDA")
        record["loaded_param_dtypes"] = dict(Counter(str(x.dtype) for x in leaves))
        record["loaded_param_bytes"] = sum(x.numel() * x.element_size() for x in leaves)
        record["matmul_precision"] = "FP32 highest; TF32 disabled"
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    else:
        from flax import nnx  # noqa: PLC0415

        leaves = jax.tree.leaves(nnx.state(policy._model, nnx.Param))  # noqa: SLF001
        record["loaded_param_dtypes"] = dict(Counter(str(x.dtype) for x in leaves))
        record["loaded_param_bytes"] = sum(x.size * x.dtype.itemsize for x in leaves)
    print("MODEL_LOAD_OK", record["load_s"], record["loaded_param_dtypes"], flush=True)
    graph_sampler = None
    if args.cuda_graph:
        from cuda_graph_sampler import CudaGraphSampler  # noqa: PLC0415

        policy._model.static_denoising_loop = True  # noqa: SLF001
        if args.compile_graph_parts:
            # Fuse each reusable region without unrolling ten experts into a
            # huge compiler graph. Disable Inductor's own graphs: the outer
            # CUDAGraph captures all ten calls after these regions are warmed.
            model = policy._model  # noqa: SLF001
            model.embed_prefix = torch.compile(model.embed_prefix, mode="max-autotune-no-cudagraphs")
            model.paligemma_with_expert.forward = torch.compile(
                model.paligemma_with_expert.forward, mode="max-autotune-no-cudagraphs"
            )
            model.denoise_step = torch.compile(model.denoise_step, mode="max-autotune-no-cudagraphs")
        original_sampler = policy._sample_actions  # noqa: SLF001

        def legacy_loop_reference(*call_args, **call_kwargs):
            policy._model.static_denoising_loop = False  # noqa: SLF001
            try:
                return original_sampler(*call_args, **call_kwargs)
            finally:
                policy._model.static_denoising_loop = True  # noqa: SLF001

        graph_sampler = CudaGraphSampler(original_sampler, reference_sampler=legacy_loop_reference)
        policy._sample_actions = graph_sampler  # noqa: SLF001
    raw = {}
    output_transform = policy._output_transform  # noqa: SLF001

    def capture_output(data):
        raw["actions"] = np.asarray(data["actions"], dtype=np.float32).copy()
        return output_transform(data)

    policy._output_transform = capture_output  # noqa: SLF001
    noise = np.random.default_rng(args.seed).standard_normal((50, 32)).astype(np.float32)
    np.save(args.output / "noise.npy", noise, allow_pickle=False)
    record["noise_sha256"] = digest(args.output / "noise.npy")
    all_actions, all_raw, measurements = [], [], []
    for sample_path, observation, provenance in samples:
        actions, normalized, latencies, warmup = [], [], [], []
        for repeat in range(args.repeats + args.warmups):
            start = time.monotonic()
            action = np.asarray(policy.infer(observation, noise=noise)["actions"], dtype=np.float32)
            if is_pytorch:
                torch.cuda.synchronize()
            elapsed_ms = (time.monotonic() - start) * 1000
            if action.shape != (50, 14) or raw["actions"].shape != (50, 32):
                raise RuntimeError("Action shape mismatch")
            if not np.isfinite(action).all() or not np.isfinite(raw["actions"]).all():
                raise RuntimeError("Non-finite model output")
            if repeat < args.warmups:
                warmup.append(elapsed_ms)
                print("WARMUP", sample_path.stem, repeat, round(elapsed_ms, 2), flush=True)
            else:
                actions.append(action.copy())
                normalized.append(raw["actions"].copy())
                latencies.append(elapsed_ms)
        actions = np.stack(actions)
        normalized = np.stack(normalized)
        measurement = {
            "sample": sample_path.name,
            "sample_sha256": digest(sample_path),
            "episode": provenance["episode"]["source_episode_index"],
            "frame": provenance["frame_index"],
            "phase": provenance["phase"],
            "warmup_ms": warmup,
            "latencies_ms": latencies,
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "repeat_max_abs_difference": float(np.max(np.abs(actions - actions[0]))),
        }
        if graph_sampler is not None:
            measurement["cuda_graph_vs_eager_max_abs"] = graph_sampler.validate_current()
        # Preserve completed observations even if a later sample fails.
        np.savez_compressed(args.output / f"{sample_path.stem}.npz", actions=actions, normalized_actions=normalized)
        (args.output / f"{sample_path.stem}.json").write_text(json.dumps(measurement, indent=2))
        measurements.append(measurement)
        all_actions.append(actions)
        all_raw.append(normalized)
        print("SAMPLE_OK", json.dumps(measurement), flush=True)
    np.save(args.output / "actions.npy", np.stack(all_actions), allow_pickle=False)
    np.save(args.output / "normalized_actions.npy", np.stack(all_raw), allow_pickle=False)
    latencies = [value for m in measurements for value in m["latencies_ms"]]
    record.update(
        status="measured_not_accuracy_approved",
        finished_at=datetime.datetime.now(datetime.UTC).isoformat(),
        measurements=measurements,
        p50_ms=float(np.percentile(latencies, 50)),
        p95_ms=float(np.percentile(latencies, 95)),
        repeat_max_abs_difference=max(m["repeat_max_abs_difference"] for m in measurements),
        process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        actions_sha256=digest(args.output / "actions.npy"),
        normalized_actions_sha256=digest(args.output / "normalized_actions.npy"),
        action_shape=list(np.stack(all_actions).shape),
    )
    if is_pytorch:
        record["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        record["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    (args.output / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print("SUITE_COMPLETE", record["p50_ms"], record["p95_ms"], flush=True)


if __name__ == "__main__":
    main()
