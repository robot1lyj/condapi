"""Export trained-prefix RTC Pi0.5 with dynamic H50 mask, no quantization.

Cases are real YAM observations with committed absolute targets from the same
episode. This script will not invent an action prefix from the model itself.
"""

import argparse
from collections import Counter
import dataclasses
import datetime
import json
import os
from pathlib import Path

from benchmark_pi05 import digest
from benchmark_pi05 import read_observation
from benchmark_suite import checked_path
import numpy as np
import onnx
from rtc_onnx_sampler import Pi05RtcOnnxSampler
from rtc_onnx_sampler import RTC_INPUT_NAMES
from rtc_onnx_sampler import RtcFlatSamplerAdapter
from rtc_policy import RtcEagerAdapter
from rtc_policy import TrainedRtcInference
import torch

from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--jax-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compute-dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    if args.output.exists() or not torch.cuda.is_available() or os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        parser.error("Use a new output directory on Thor CUDA with TF32 override disabled")
    manifest_path = args.checkpoint / "rtc_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit_path = args.checkpoint / "conversion_audit.json"
    norm_path = args.checkpoint / "assets" / "yam" / "norm_stats.json"
    if (
        manifest["route"] != "pi05_yam_trained_rtc"
        or manifest["conversion_audit_sha256"] != digest(audit_path)
        or manifest["norm_stats_sha256"] != digest(norm_path)
        or manifest["model_weights_sha256"] != digest(args.checkpoint / "model.safetensors")
    ):
        raise ValueError("RTC converted checkpoint provenance mismatch")
    max_delay = manifest["max_delay_steps"]
    case_set = json.loads(args.cases.read_text())
    reference_manifest = json.loads((args.jax_reference / "reference_manifest.json").read_text())
    if (
        reference_manifest.get("status") != "original_jax_trained_rtc_reference"
        or reference_manifest.get("training_contract_sha256") != manifest["training_contract_sha256"]
        or reference_manifest.get("source_params_files_sha256") != manifest["source_params_files_sha256"]
        or reference_manifest.get("norm_stats_sha256") != digest(norm_path)
        or reference_manifest.get("cases_sha256") != digest(args.cases)
    ):
        raise ValueError("JAX RTC reference does not match converted checkpoint and cases")
    if case_set.get("source_kind") != "real_yam_recording" or case_set.get("norm_stats_sha256") != digest(norm_path):
        raise ValueError("Real YAM cases and matching checkpoint norm required")
    cases = []
    for row in case_set["cases"]:
        sample_path = checked_path(args.cases.parent, row["sample"])
        provenance_path = checked_path(args.cases.parent, row["provenance"])
        prefix_path = checked_path(args.cases.parent, row["committed_actions"])
        provenance = json.loads(provenance_path.read_text())
        if (
            digest(sample_path) != provenance["sample_sha256"]
            or provenance["norm_stats_sha256"] != digest(norm_path)
            or digest(prefix_path) != row["committed_actions_sha256"]
            or not provenance.get("source_files")
            or row.get("source_episode") not in provenance["source_files"]
        ):
            raise ValueError("RTC case provenance or committed actions changed")
        committed = np.load(prefix_path, allow_pickle=False)
        rtc = {
            "delay_steps": row["delay_steps"],
            "observation_policy_tick": row["observation_policy_tick"],
            "target_start_tick": row["target_start_tick"],
            "committed_start_tick": row["committed_start_tick"],
            "committed_actions": committed,
        }
        cases.append((row["sample"], read_observation(sample_path), rtc))
    delays = {rtc["delay_steps"] for _, _, rtc in cases}
    if not {0, 1, max_delay}.issubset(delays):
        raise ValueError("RTC export cases must include delays 0, 1 and trained maximum")
    if len(reference_manifest["cases"]) != len(cases):
        raise ValueError("JAX RTC reference case count mismatch")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train = config.get_config("pi05_yam")
    train = dataclasses.replace(train, model=dataclasses.replace(train.model, dtype=args.compute_dtype))
    stats = normalize.deserialize_json(norm_path.read_text())
    policy = policy_config.create_trained_policy(
        train, args.checkpoint, norm_stats=stats, pytorch_device="cuda",
        pytorch_precision=args.compute_dtype, pytorch_compile=False,
    )
    model = policy._model  # noqa: SLF001
    model.batch_vision = True
    model.attention_implementation = "eager"
    model.attention_mask_dtype = next(
        model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.parameters()
    ).dtype
    eager = RtcEagerAdapter(model, max_delay=max_delay)
    wrapper = Pi05RtcOnnxSampler(model, cache_time_modulation=False, text_bucket=200).eval()
    prepared = RtcFlatSamplerAdapter(wrapper, max_delay=max_delay)
    eager_policy = TrainedRtcInference(policy, stats, max_delay=max_delay, sampler=eager)
    prepared_policy = TrainedRtcInference(policy, stats, max_delay=max_delay, sampler=prepared)
    noise = np.random.default_rng(0).standard_normal((1, 50, 32)).astype(np.float32)
    args.output.mkdir(parents=True)
    report = {
        "phase": "rtc_export",
        "status": "started",
        "rtc_mode": "trained",
        "max_delay_steps": max_delay,
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "converted_weights_sha256": digest(args.checkpoint / "model.safetensors"),
        "conversion_audit_sha256": digest(audit_path),
        "rtc_manifest_sha256": digest(manifest_path),
        "norm_stats_sha256": digest(norm_path),
        "cases_sha256": digest(args.cases),
        "jax_reference_manifest_sha256": digest(args.jax_reference / "reference_manifest.json"),
        "compute_dtype": args.compute_dtype,
        "contract": {"views": 3, "image_resolution": [224, 224], "horizon": 50, "steps": 10, "action_dim": 32},
        "text_bucket": 200,
        "cache_time_modulation": False,
        "tf32": False,
        "quantization": None,
        "nonfinite_sanitization": False,
        "reference_scope": "trained_rtc_eager_same_checkpoint_norm_noise_cases",
        "wrapper_comparisons": [],
        "jax_comparisons": [],
    }

    def save():
        (args.output / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    save()
    first = None
    for index, (name, observation, rtc) in enumerate(cases):
        expected = eager_policy.infer_rtc(observation, rtc, noise=noise)["actions"]
        raw = eager.last_raw.detach().cpu().numpy()
        actual = prepared_policy.infer_rtc(observation, rtc, noise=noise)["actions"]
        prepared_raw = prepared.last_raw.detach().cpu().numpy()
        finite = bool(np.isfinite(raw).all() and np.isfinite(prepared_raw).all() and np.isfinite(actual).all())
        exact = bool(np.array_equal(raw, prepared_raw) and np.array_equal(expected, actual))
        report["wrapper_comparisons"].append({
            "sample": name, "delay_steps": rtc["delay_steps"], "finite": finite, "exact": exact,
            "raw_max_abs": float(np.max(np.abs(raw.astype(np.float64) - prepared_raw.astype(np.float64)))),
            "physical_max_abs": float(np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64)))),
        })
        ref_row = reference_manifest["cases"][index]
        ref_path = checked_path(args.jax_reference, ref_row["reference"])
        if ref_row["sample"] != name or ref_row["delay_steps"] != rtc["delay_steps"] or digest(ref_path) != ref_row["sha256"]:
            raise ValueError("JAX RTC reference case identity mismatch")
        with np.load(ref_path, allow_pickle=False) as data:
            jax_raw, jax_physical = data["normalized"], data["physical"]
        raw_error = np.abs(prepared_raw.astype(np.float64) - jax_raw[None].astype(np.float64))
        physical_error = np.abs(actual.astype(np.float64) - jax_physical.astype(np.float64))
        numeric_ok = bool(
            np.isfinite(jax_raw).all() and np.isfinite(jax_physical).all()
            and raw_error.max() <= 1e-4 and physical_error.max() <= 1e-4
        )
        report["jax_comparisons"].append({
            "sample": name, "delay_steps": rtc["delay_steps"],
            "normalized_max_abs": float(raw_error.max()),
            "physical_max_abs": float(physical_error.max()),
            "numeric_gate_1e-4": numeric_ok,
            "scope": "fixed-noise numerical gate only; not robot task accuracy",
        })
        if first is None and rtc["delay_steps"]:
            first = tuple(x.clone() for x in prepared.last_inputs)
        save()
        if not finite or not exact:
            raise RuntimeError("Trained RTC wrapper differs from eager; ONNX export blocked")
        if args.compute_dtype == "float32" and not numeric_ok:
            raise RuntimeError("FP32 trained RTC differs from original JAX beyond numerical gate")
    report["input_contract"] = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in zip(RTC_INPUT_NAMES, first, strict=True)
    }
    np.savez_compressed(args.output / "example_inputs.npz", **{
        name: value.cpu().numpy() for name, value in zip(RTC_INPUT_NAMES, first, strict=True)
    })
    report["loaded_param_dtypes"] = dict(Counter(str(p.dtype) for p in model.parameters()))
    save()
    onnx_path = args.output / "sampler.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper, first, str(onnx_path), opset_version=19, dynamo=True,
            do_constant_folding=True, external_data=True,
            input_names=list(RTC_INPUT_NAMES), output_names=["actions"],
            report=True, artifacts_dir=str(args.output / "diagnostics"),
        )
    metadata = onnx.load(str(onnx_path), load_external_data=False)
    report["onnx_inputs"] = [value.name for value in metadata.graph.input]
    if not {"previous_actions", "prefix_mask"}.issubset(report["onnx_inputs"]):
        raise RuntimeError("RTC dynamic prefix inputs disappeared during export")
    report["onnx_initializer_dtypes"] = dict(
        Counter(onnx.TensorProto.DataType.Name(t.data_type) for t in metadata.graph.initializer)
    )
    report["onnx_sha256"] = digest(onnx_path)
    external = set()
    for tensor in metadata.graph.initializer:
        for item in tensor.external_data:
            if item.key == "location":
                path = (args.output / item.value).resolve()
                if not path.is_relative_to(args.output.resolve()):
                    raise ValueError("ONNX external weight escaped export directory")
                external.add(path)
    report["external_weights_sha256"] = {
        str(path.relative_to(args.output.resolve())): digest(path) for path in sorted(external)
    }
    onnx.checker.check_model(str(onnx_path))
    report["status"] = "onnx_exported_engine_not_validated"
    report["finished_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    save()


if __name__ == "__main__":
    main()
