"""Small offline RLinf core comparison, not an RLinf training/deployment integration.

The upstream converter and Pi0 core are imported unchanged from a read-only
checkout. Only the package namespace, explicit dtype, and YAM IO are adapted.
"""

# Conversion must import on a CPU-only container, without model dependencies.
# ruff: noqa: PLC0415

import argparse
from collections import Counter
import datetime
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

RLINF_REVISION = "1c9eed00a60009d9b04337b85e4cc84d474e405e"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def load_package(name, path):
    """Avoid importing RL/Ray/FSDP orchestration for a model-core experiment."""
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py", submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_identity(root):
    parts = (
        root / "rlinf/models/embodiment/openpi_rlinf/pi0_model",
        root / "rlinf/utils/ckpt_convertor/openpi",
    )
    return {
        "upstream_revision": RLINF_REVISION,
        "files_sha256": {str(p.relative_to(root)): digest(p) for part in parts for p in sorted(part.glob("*.py"))},
    }


def load_converter(root):
    # Upstream converters use absolute imports. Supply only package namespaces,
    # not rlinf.__init__ (which registers training-stack OmegaConf resolvers).
    for name in ("rlinf", "rlinf.utils", "rlinf.utils.ckpt_convertor"):
        module = types.ModuleType(name)
        module.__path__ = [str(root.joinpath(*name.split(".")))]
        sys.modules[name] = module
    load_package("rlinf.utils.ckpt_convertor.openpi", root / "rlinf/utils/ckpt_convertor/openpi")
    return importlib.import_module("rlinf.utils.ckpt_convertor.openpi.jax_to_openpi_rlinf")


def convert(args):
    import numpy as np
    from safetensors import safe_open
    import torch

    if args.output.exists():
        raise ValueError("Use a new conversion directory")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    identity = source_identity(args.source)
    write_json(args.output / "started.json", {"phase": "convert", "command": sys.argv, **identity})
    converter = load_converter(args.source)
    # Reject adapters and already-rounded source leaves before invoking the
    # upstream converter's FP32 restoration. This does not implement LoRA merge.
    import orbax.checkpoint as ocp

    metadata = ocp.PyTreeCheckpointer().metadata(str(args.checkpoint / "params"))

    def audit_tree(tree, prefix=""):
        if isinstance(tree, dict):
            for key, value in tree.items():
                if "lora" in key.lower():
                    raise ValueError("Unmerged LoRA is outside this base-model probe")
                audit_tree(value, f"{prefix}/{key}")
        elif hasattr(tree, "dtype") and np.dtype(tree.dtype) != np.dtype("float32"):
            raise ValueError(f"Source is not original FP32: {prefix}")

    audit_tree(metadata)
    converter.convert(
        args.checkpoint,
        args.norm,
        args.output,
        args.output / "norm_stats.json",
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
    )
    weight_file = args.output / "model.safetensors"
    with safe_open(str(weight_file), framework="pt", device="cpu") as weights:
        dtypes = Counter()
        for key in weights.keys():  # noqa: SIM118 - safetensors reader, not a dict
            tensor = weights.get_tensor(key)
            if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid converted tensor: {key}")
            dtypes[str(tensor.dtype)] += 1
    write_json(
        args.output / "conversion_audit.json",
        {
            "phase": "convert",
            "finished_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "source_checkpoint": str(args.checkpoint),
            "checkpoint_metadata_sha256": digest(args.checkpoint / "params/_METADATA"),
            "converted_weights_sha256": digest(weight_file),
            "output_precision": "float32",
            "config_sha256": digest(args.output / "config.json"),
            "norm_stats_sha256": digest(args.norm),
            "loaded_tensor_dtypes": dict(dtypes),
            "duration_s": time.monotonic() - started,
            "lora": "absent; no merge performed",
            "inference_equivalence": "not established by this conversion audit",
            **identity,
        },
    )
    print("CONVERSION_COMPLETE", str(args.output), flush=True)


def compare_reference(reference_path, record, actions, normalized):
    import numpy as np

    from scripts.thor.compare_suites import error_metrics
    from scripts.thor.compare_suites import read_run

    reference, arrays = read_run(reference_path)
    for key in (
        "suite_sha256",
        "norm_stats_sha256",
        "checkpoint_metadata_sha256",
        "noise_sha256",
        "steps",
        "horizon",
        "seed",
    ):
        if reference[key] != record[key]:
            raise ValueError(f"Confounded reference: {key}")
    if [x["sample_sha256"] for x in reference["measurements"]] != [x["sample_sha256"] for x in record["measurements"]]:
        raise ValueError("Reference sample order mismatch")
    return {
        "reference": reference["run_id"],
        "reference_result_sha256": digest(reference_path / "result.json"),
        "physical_dataset_units": error_metrics(arrays["actions"][:, 0], actions[:, 0]),
        "normalized_active_14d": error_metrics(arrays["normalized_actions"][:, 0, :, :14], normalized[:, 0, :, :14]),
        "normalized_internal_32d": error_metrics(arrays["normalized_actions"][:, 0], normalized[:, 0]),
        "scope": "first measured repeat only; fewer timing repeats in exploratory candidate",
        "repeat_difference": float(np.max(np.abs(actions - actions[:, :1]))),
    }


def audit_mapping(args):
    import safetensors.torch
    import torch

    torch.set_num_threads(8)
    if args.output.exists():
        raise ValueError("Use a new mapping audit output")
    load_converter(args.source)
    converter = importlib.import_module("rlinf.utils.ckpt_convertor.openpi.openpi_pytorch_to_openpi_rlinf")
    original = safetensors.torch.load_file(str(args.pytorch_checkpoint / "model.safetensors"))
    mapped = converter.old_to_new_state_dict(original)
    direct = safetensors.torch.load_file(str(args.checkpoint / "model.safetensors"))
    if set(mapped) != set(direct):
        raise ValueError(f"Mapped key mismatch: {set(mapped) ^ set(direct)}")
    mismatches = [
        {"name": name, "max_abs": float((mapped[name] - direct[name]).abs().max())}
        for name in direct
        if not torch.equal(mapped[name], direct[name])
    ]
    write_json(
        args.output,
        {
            "phase": "audit",
            "finished_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "comparison": "direct JAX-to-RLinf versus existing audited FP32 PyTorch-to-RLinf layout",
            "tensor_count": len(direct),
            "all_tensors_bit_equal": not mismatches,
            "mismatches": mismatches,
            "direct_weights_sha256": digest(args.checkpoint / "model.safetensors"),
            "pytorch_weights_sha256": digest(args.pytorch_checkpoint / "model.safetensors"),
            "source_identity": source_identity(args.source),
            "probe_script_sha256": digest(__file__),
        },
    )
    print("MAPPING_AUDIT_COMPLETE", len(direct), "bit_equal", not mismatches, flush=True)


def use_tanh_gelu(core, siglip):
    """Explicit diagnostic ablation; never label this as unmodified RLinf."""
    import torch.nn.functional as functional

    class SiglipFunctions:
        def __getattr__(self, name):
            return getattr(functional, name)

        @staticmethod
        def gelu(value):
            return functional.gelu(value, approximate="tanh")

    siglip.F = SiglipFunctions()
    core.gelu_glu = lambda gate, value: functional.gelu(gate, approximate="tanh") * value


def benchmark(args):
    import jax
    import numpy as np
    import safetensors.torch
    import torch

    from openpi import transforms
    from openpi.policies.policy import Policy
    from openpi.shared import normalize
    from openpi.training import config
    from scripts.thor.benchmark_pi05 import read_observation

    if args.output.exists():
        raise ValueError("Use a new benchmark directory")
    if not torch.cuda.is_available() or os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise RuntimeError("CUDA required; vendor TF32 override must be disabled")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(8)
    args.output.mkdir(parents=True)
    identity = source_identity(args.source)
    conversion = json.loads((args.checkpoint / "conversion_audit.json").read_text())
    if identity["files_sha256"] != conversion["files_sha256"]:
        raise ValueError("RLinf source changed since conversion")
    weight_file = args.checkpoint / "model.safetensors"
    if conversion["output_precision"] != "float32" or digest(weight_file) != conversion["converted_weights_sha256"]:
        raise ValueError("FP32 checkpoint fingerprint mismatch")
    suite = json.loads(args.suite.read_text())
    norm_path = args.suite.parent / suite["norm_stats"]
    if digest(norm_path) != suite["norm_stats_sha256"] or digest(norm_path) != conversion["norm_stats_sha256"]:
        raise ValueError("Normalization mismatch")
    noise = np.random.default_rng(0).standard_normal((50, 32)).astype(np.float32)
    np.save(args.output / "noise.npy", noise, allow_pickle=False)
    record = {
        "run_id": args.output.name,
        "phase": "evaluate",
        "status": "running",
        "command": sys.argv,
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "backend": "rlinf_pytorch_core",
        "config": "pi05_yam",
        "suite_sha256": digest(args.suite),
        "suite": suite,
        "norm_stats_sha256": digest(norm_path),
        "noise_sha256": digest(args.output / "noise.npy"),
        "checkpoint_metadata_sha256": conversion["checkpoint_metadata_sha256"],
        "converted_weights_sha256": conversion["converted_weights_sha256"],
        "conversion_audit_sha256": digest(args.checkpoint / "conversion_audit.json"),
        "storage_dtype": "float32",
        "compute_dtype": args.dtype,
        "compiled": False,
        "gelu": args.gelu,
        "runtime_ablation": "SigLIP and Gemma GELU use tanh approximation" if args.gelu == "tanh" else None,
        "precision_policy": "FP32 all parameters"
        if args.dtype == "float32"
        else "BF16 parameters; upstream FP32 internal attention/stem/norm operations",
        "steps": 10,
        "horizon": 50,
        "seed": 0,
        "repeats": args.repeats,
        "warmups_per_sample": 2,
        "text_bucket": 200,
        "tf32": False,
        "source_identity": identity,
        "probe_script_sha256": digest(__file__),
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda, "jax": jax.__version__},
        "device": torch.cuda.get_device_name(),
        "precision_acceptance_threshold": None,
        "limitations": [
            "Exploratory core inference only; no SFT, optimizer, LoRA, task success or RLinf orchestration test",
            "IO adapter preserves project YAM transforms; optional GELU ablation is separately labeled",
            "Explicit core compute dtype, not the factory's fixed bfloat16 config",
            "TorchDynamo disabled, including upstream decorated GELU; not an optimized speed comparison",
            "Nine distinct recorded inputs; repeats estimate timing only; dataset physical units uncalibrated",
        ],
    }
    write_json(args.output / "started.json", record)
    load_package("rlinf_probe_core", args.source / "rlinf/models/embodiment/openpi_rlinf/pi0_model")
    core = importlib.import_module("rlinf_probe_core.pi0")
    cfg_module = importlib.import_module("rlinf_probe_core.pi0_config")
    obs_module = importlib.import_module("rlinf_probe_core.model")
    if args.gelu == "tanh":
        use_tanh_gelu(
            importlib.import_module("rlinf_probe_core.gemma"), importlib.import_module("rlinf_probe_core.siglip")
        )
    model_config = cfg_module.Pi0Config(
        pi05=True, dtype=args.dtype, action_dim=32, action_horizon=50, max_token_len=200
    )
    started = time.monotonic()
    with torch.device("meta"):
        model = core.Pi0(model_config)
    weights = safetensors.torch.load_file(str(weight_file), device="cpu")
    if any(value.dtype != torch.float32 for value in weights.values()):
        raise ValueError("Expected preserved FP32 master weights")
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    model.to(device="cuda", dtype=getattr(torch, args.dtype)).eval()
    record["loaded_param_dtypes"] = dict(Counter(str(p.dtype) for p in model.parameters()))
    record["load_s"] = time.monotonic() - started

    class Adapter(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        @torch.no_grad()
        def sample_actions(self, _device, observation, **kwargs):
            # OpenPI Policy builds BCHW floats for PyTorch; RLinf expects BHWC.
            images = {key: value.permute(0, 2, 3, 1) for key, value in observation.images.items()}
            if any(value.shape != (1, 224, 224, 3) for value in images.values()):
                raise ValueError("Expected three transformed 224px images")
            obs = obs_module.Observation(
                images=images,
                image_masks=observation.image_masks,
                state=observation.state,
                tokenized_prompt=observation.tokenized_prompt,
                tokenized_prompt_mask=observation.tokenized_prompt_mask,
            )
            return self.wrapped.sample_actions(obs, **kwargs)

    train_config = config.get_config("pi05_yam")
    data = train_config.data.create(train_config.assets_dirs, train_config.model)
    norms = normalize.deserialize_json(norm_path.read_text())
    raw = {}

    def capture(outputs):
        raw["actions"] = outputs["actions"].copy()
        return outputs

    policy = Policy(
        Adapter(model),
        is_pytorch=True,
        pytorch_device="cuda",
        sample_kwargs={"num_steps": 10},
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data.data_transforms.inputs,
            transforms.Normalize(norms, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ],
        output_transforms=[
            capture,
            *data.model_transforms.outputs,
            transforms.Unnormalize(norms, use_quantiles=data.use_quantile_norm),
            *data.data_transforms.outputs,
        ],
    )
    print("MODEL_LOAD_OK", record["load_s"], record["loaded_param_dtypes"], flush=True)
    all_actions, all_raw, measurements = [], [], []
    for entry in suite["samples"]:
        sample_path = args.suite.parent / entry["sample"]
        provenance = json.loads((args.suite.parent / entry["provenance"]).read_text())
        if digest(sample_path) != provenance["sample_sha256"] or provenance["norm_stats_sha256"] != digest(norm_path):
            raise ValueError("Sample provenance mismatch")
        observation = read_observation(sample_path)
        actions, normalized, times, warmups = [], [], [], []
        for index in range(args.repeats + 2):
            started = time.monotonic()
            action = np.asarray(policy.infer(observation, noise=noise)["actions"], dtype=np.float32)
            torch.cuda.synchronize()
            elapsed = (time.monotonic() - started) * 1000
            if action.shape != (50, 14) or raw["actions"].shape != (50, 32):
                raise ValueError("Unexpected action shape")
            if not np.isfinite(action).all() or not np.isfinite(raw["actions"]).all():
                raise ValueError("Non-finite model output")
            if index < 2:
                warmups.append(elapsed)
            else:
                actions.append(action.copy())
                normalized.append(raw["actions"].copy())
                times.append(elapsed)
        measurement = {
            "sample": sample_path.name,
            "sample_sha256": digest(sample_path),
            "episode": provenance["episode"]["source_episode_index"],
            "frame": provenance["frame_index"],
            "warmup_ms": warmups,
            "latencies_ms": times,
        }
        np.savez_compressed(args.output / f"{sample_path.stem}.npz", actions=actions, normalized_actions=normalized)
        write_json(args.output / f"{sample_path.stem}.json", measurement)
        measurements.append(measurement)
        all_actions.append(actions)
        all_raw.append(normalized)
        print("SAMPLE_OK", sample_path.name, float(np.median(times)), flush=True)
    all_actions, all_raw = np.asarray(all_actions), np.asarray(all_raw)
    np.save(args.output / "actions.npy", all_actions, allow_pickle=False)
    np.save(args.output / "normalized_actions.npy", all_raw, allow_pickle=False)
    times = [t for m in measurements for t in m["latencies_ms"]]
    record.update(
        status="measured_not_accuracy_approved",
        finished_at=datetime.datetime.now(datetime.UTC).isoformat(),
        measurements=measurements,
        p50_ms=float(np.percentile(times, 50)),
        p95_ms=float(np.percentile(times, 95)),
        actions_sha256=digest(args.output / "actions.npy"),
        normalized_actions_sha256=digest(args.output / "normalized_actions.npy"),
        action_shape=list(all_actions.shape),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    record["comparisons"] = [compare_reference(ref, record, all_actions, all_raw) for ref in args.reference]
    write_json(args.output / "result.json", record)
    print(
        "PROBE_COMPLETE",
        record["p50_ms"],
        record["p95_ms"],
        json.dumps([{c["reference"]: c["physical_dataset_units"]["mae"]} for c in record["comparisons"]]),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("convert", "benchmark", "audit-mapping"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--norm", type=Path)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--gelu", choices=("upstream", "tanh"), default="upstream")
    parser.add_argument("--pytorch-checkpoint", type=Path)
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.phase == "convert" and args.norm is None:
        parser.error("Conversion requires --norm")
    if args.phase == "benchmark" and (args.suite is None or not args.reference or args.repeats < 2):
        parser.error("Benchmark requires suite, reference and at least two repeats")
    if args.phase == "audit-mapping" and args.pytorch_checkpoint is None:
        parser.error("Mapping audit requires --pytorch-checkpoint")
    {"convert": convert, "benchmark": benchmark, "audit-mapping": audit_mapping}[args.phase](args)


if __name__ == "__main__":
    main()
