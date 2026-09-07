"""Export the real three-camera H50 Pi0.5 sampler without quantization or FP16.

The prepared PyTorch wrapper must match the legacy eager sampler on every
recorded input before ONNX is written. Engine acceptance is a separate stage.
"""

import argparse
from collections import Counter
import dataclasses
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import time

from benchmark_pi05 import digest
from benchmark_pi05 import read_observation
from benchmark_suite import checked_path
import numpy as np
import onnx
from onnx_sampler import INPUT_NAMES
from onnx_sampler import FlatSamplerAdapter
from onnx_sampler import Pi05OnnxSampler
import torch

from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compute-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--exporter", choices=("dynamo", "legacy"), default="dynamo")
    parser.add_argument("--cache-time-modulation", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new immutable export directory")
    if not torch.cuda.is_available() or os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        parser.error("CUDA required and vendor TF32 override must be disabled")
    suite = json.loads(args.suite.read_text())
    root = args.suite.parent
    norm = checked_path(root, suite["norm_stats"])
    if suite.get("source_kind") != "real_yam_recording" or not suite.get("source_files"):
        raise ValueError("Recorded source evidence required")
    if digest(norm) != suite["norm_stats_sha256"]:
        raise ValueError("Norm hash mismatch")
    samples = []
    for sample in suite["samples"]:
        path = checked_path(root, sample["sample"])
        provenance = json.loads(checked_path(root, sample["provenance"]).read_text())
        if (
            digest(path) != provenance["sample_sha256"]
            or provenance["norm_stats_sha256"] != digest(norm)
            or provenance["source_files"] != suite["source_files"]
        ):
            raise ValueError("Sample provenance mismatch")
        samples.append((path.name, read_observation(path)))
    if len(samples) != suite["observation_count"] or not samples:
        raise ValueError("Suite count mismatch")
    conversion = json.loads((args.checkpoint / "conversion_audit.json").read_text())
    if conversion["output_precision"] != "float32":
        raise ValueError("Use the audited original FP32 conversion")
    args.output.mkdir(parents=True)
    report = {
        "run_id": args.output.name,
        "phase": "export",
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "status": "started",
        "suite_sha256": digest(args.suite),
        "norm_stats_sha256": digest(norm),
        "converted_weights_sha256": digest(args.checkpoint / "model.safetensors"),
        "conversion_audit_sha256": digest(args.checkpoint / "conversion_audit.json"),
        "versions": {
            p: importlib.metadata.version(p) for p in ("torch", "transformers", "onnx", "onnxscript", "tensorrt")
        },
        "contract": {"views": 3, "image_resolution": [224, 224], "horizon": 50, "steps": 10, "action_dim": 32},
        "compute_dtype": args.compute_dtype,
        "exporter": args.exporter,
        "cache_time_modulation": args.cache_time_modulation,
        "quantization": None,
        "tf32": False,
        "nonfinite_sanitization": False,
        "scope": "base model only; no LoRA; export preparation is not engine/task accuracy acceptance",
    }

    def save_report():
        (args.output / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    save_report()
    train = config.get_config("pi05_yam")
    train = dataclasses.replace(train, model=dataclasses.replace(train.model, dtype=args.compute_dtype))
    print("EXPORT_MODEL_LOAD_START", flush=True)
    policy = policy_config.create_trained_policy(
        train,
        args.checkpoint,
        norm_stats=normalize.deserialize_json(norm.read_text()),
        pytorch_device="cuda",
        pytorch_precision=args.compute_dtype,
        pytorch_compile=False,
        sample_kwargs={"num_steps": 10},
    )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = policy._model  # noqa: SLF001
    model.batch_vision = True
    model.attention_implementation = "eager"
    model.attention_mask_dtype = next(
        model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.parameters()
    ).dtype
    report["loaded_param_dtypes"] = dict(Counter(str(p.dtype) for p in model.parameters()))
    print("EXPORT_MODEL_LOAD_OK", report["loaded_param_dtypes"], flush=True)
    noise = np.random.default_rng(0).standard_normal((50, 32)).astype(np.float32)
    np.save(args.output / "noise.npy", noise, allow_pickle=False)
    report["noise_sha256"] = digest(args.output / "noise.npy")
    captured = {}
    original_output_transform = policy._output_transform  # noqa: SLF001

    def capture_output(data):
        captured["raw"] = np.asarray(data["actions"], dtype=np.float32).copy()
        return original_output_transform(data)

    policy._output_transform = capture_output  # noqa: SLF001
    references = []
    for name, observation in samples:
        action = policy.infer(observation, noise=noise)["actions"]
        references.append((action.copy(), captured["raw"].copy()))
        print("EXPORT_REFERENCE_OK", name, flush=True)
    wrapper = Pi05OnnxSampler(model, cache_time_modulation=args.cache_time_modulation).eval()
    if wrapper.cached_modulations:
        cache_path = args.output / "time_modulation_cache.npz"
        np.savez_compressed(
            cache_path,
            **{
                name: projection.table.detach().cpu().numpy()
                for name, projection in zip(wrapper.cache_names, wrapper.cached_modulations, strict=True)
            },
        )
        report["time_modulation_cache"] = {
            "sha256": digest(cache_path),
            "modules": wrapper.cache_names,
            "dtype": "float32",
            "batching": "ten independent batch-1 original GEMVs; no reduced precision",
            "invalidated_by": "any checkpoint/LoRA weight or denoising schedule change",
        }
    adapter = FlatSamplerAdapter(wrapper)
    policy._sample_actions = adapter  # noqa: SLF001
    report["wrapper_comparisons"] = []
    first_inputs = None
    for (name, observation), (ref_action, ref_raw) in zip(samples, references, strict=True):
        action = policy.infer(observation, noise=noise)["actions"]
        raw = captured["raw"]
        if first_inputs is None:
            first_inputs = tuple(x.clone() for x in adapter.last_inputs)
        difference = {
            "sample": name,
            "raw_max_abs": float(np.max(np.abs(raw - ref_raw))),
            "physical_max_abs": float(np.max(np.abs(action - ref_action))),
            "finite": bool(np.isfinite(raw).all() and np.isfinite(action).all()),
            "exact": bool(np.array_equal(raw, ref_raw) and np.array_equal(action, ref_action)),
        }
        report["wrapper_comparisons"].append(difference)
        print("EXPORT_WRAPPER_CHECK", difference, flush=True)
        np.savez_compressed(args.output / f"{Path(name).stem}.npz", reference=ref_raw, prepared=raw)
        save_report()
        if not difference["finite"] or not difference["exact"]:
            raise RuntimeError("Export preparation changed the eager output; investigate before exporting")
    np.save(args.output / "time_embeddings.npy", wrapper.time_embeddings.detach().cpu().numpy(), allow_pickle=False)
    report["time_embeddings_sha256"] = digest(args.output / "time_embeddings.npy")
    report["times_fp32"] = wrapper.times.detach().cpu().tolist()
    report["input_contract"] = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in zip(INPUT_NAMES, first_inputs, strict=True)
    }
    np.savez_compressed(
        args.output / "example_inputs.npz",
        **{name: value.detach().cpu().numpy() for name, value in zip(INPUT_NAMES, first_inputs, strict=True)},
    )
    save_report()
    onnx_path = args.output / "sampler.onnx"
    print("ONNX_EXPORT_START", flush=True)
    started = time.monotonic()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            first_inputs,
            str(onnx_path),
            opset_version=19,
            dynamo=args.exporter == "dynamo",
            do_constant_folding=True,
            external_data=True,
            input_names=list(INPUT_NAMES),
            output_names=["actions"],
            **(
                {"report": True, "artifacts_dir": str(args.output / "diagnostics")} if args.exporter == "dynamo" else {}
            ),
        )
    report["export_s"] = time.monotonic() - started
    metadata = onnx.load(str(onnx_path), load_external_data=False)
    report["onnx_inputs"] = [value.name for value in metadata.graph.input]
    report["onnx_initializer_dtypes"] = dict(
        Counter(onnx.TensorProto.DataType.Name(t.data_type) for t in metadata.graph.initializer)
    )
    report["onnx_nodes"] = len(metadata.graph.node)
    report["onnx_sha256"] = digest(onnx_path)
    external = set()
    for tensor in metadata.graph.initializer:
        for item in tensor.external_data:
            if item.key == "location":
                path = (args.output / item.value).resolve()
                if not path.is_relative_to(args.output.resolve()):
                    raise ValueError("ONNX external weights escaped export directory")
                external.add(path)
    report["external_weights_sha256"] = {str(p.relative_to(args.output.resolve())): digest(p) for p in sorted(external)}
    onnx.checker.check_model(str(onnx_path))
    report["status"] = "onnx_exported_engine_not_validated"
    report["finished_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    save_report()
    print("ONNX_EXPORT_COMPLETE", report["export_s"], report["onnx_initializer_dtypes"], flush=True)


if __name__ == "__main__":
    main()
