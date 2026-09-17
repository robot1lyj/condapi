"""Build a strongly typed, non-quantized engine from an audited Pi0.5 export."""

import argparse
import datetime
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import time

from benchmark_pi05 import digest
import onnx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimization-level", type=int, choices=range(6), default=3)
    parser.add_argument(
        "--tf32-experiment", action="store_true",
        help="Build an RTC-only TF32 candidate, never a production-approved engine",
    )
    parser.add_argument(
        "--bf16-experiment", action="store_true",
        help="Build an RTC-only BF16 candidate, never a production-approved engine",
    )
    args = parser.parse_args()
    if args.tf32_experiment and args.bf16_experiment:
        parser.error("Choose only one precision experiment")
    if args.output.exists():
        parser.error("Use a new engine output directory")
    export = json.loads((args.source / "export_report.json").read_text())
    if export["status"] != "onnx_exported_engine_not_validated" or export["quantization"] is not None:
        raise ValueError("A completed non-quantized export is required")
    if args.tf32_experiment and (export.get("rtc_mode") != "trained" or export.get("compute_dtype") != "float32"):
        raise ValueError("TF32 experiment requires the audited FP32 trained-RTC export")
    if args.bf16_experiment and (export.get("rtc_mode") != "trained" or export.get("compute_dtype") != "bfloat16"):
        raise ValueError("BF16 experiment requires a trained-RTC mixed-precision export")
    optimized_wrapper = bool(export.get("cache_time_modulation") or export.get("text_bucket", 200) != 200)
    if not export["wrapper_comparisons"] or not all(
        row["finite"] and (row["exact"] or (optimized_wrapper and row.get("optimized_numerical_gate_1e-5")))
        for row in export["wrapper_comparisons"]
    ):
        raise ValueError("Export preparation did not pass its own equivalence checks")
    if export.get("rtc_mode") == "trained" and (
        not export.get("jax_comparisons")
        or (not args.bf16_experiment and not all(row["numeric_gate_1e-4"] for row in export["jax_comparisons"]))
        or (args.bf16_experiment and not all(
            math.isfinite(row["normalized_max_abs"]) and math.isfinite(row["physical_max_abs"])
            for row in export["jax_comparisons"]
        ))
        or not {"previous_actions", "prefix_mask"}.issubset(export.get("onnx_inputs", []))
    ):
        raise ValueError("Trained RTC requires original-JAX agreement and dynamic prefix ONNX inputs")
    if export["tf32"] or export["nonfinite_sanitization"]:
        raise ValueError("Non-quantized reference cannot enable TF32 or sanitize non-finite activations")
    legacy_bucket_evidence = (
        export.get("reference_scope") == "legacy_eager_same_text_bucket"
        and export.get("padding_experiment", {}).get("status") == "offline_experiment_supported_not_accuracy_approved"
        and bool(export.get("full_text_reference_comparisons"))
    )
    rtc_bucket_evidence = (
        export.get("rtc_mode") == "trained"
        and len(export.get("jax_comparisons", [])) >= 9
        and all(row.get("numeric_gate_1e-4") for row in export["jax_comparisons"])
        and all(row.get("optimized_numerical_gate_1e-5") for row in export["wrapper_comparisons"])
    )
    if export.get("text_bucket", 200) < 200 and not (legacy_bucket_evidence or rtc_bucket_evidence):
        raise ValueError("Padding engine requires separately recorded full-text diagnostic evidence")
    onnx_path = args.source / "sampler.onnx"
    if digest(onnx_path) != export["onnx_sha256"]:
        raise ValueError("ONNX fingerprint mismatch")
    metadata = onnx.load(str(onnx_path), load_external_data=False)
    if args.bf16_experiment and not any(t.data_type == onnx.TensorProto.BFLOAT16 for t in metadata.graph.initializer):
        raise ValueError("BF16 candidate ONNX has no BF16 initializers")
    if any(node.op_type in ("QuantizeLinear", "DequantizeLinear") for node in metadata.graph.node):
        raise ValueError("Unexpected quantization nodes in the non-quantized route")
    external_paths = set()
    for tensor in metadata.graph.initializer:
        for item in tensor.external_data:
            if item.key == "location":
                path = (args.source / item.value).resolve()
                if not path.is_relative_to(args.source.resolve()):
                    raise ValueError("ONNX external weights must stay inside its export directory")
                external_paths.add(path)
    args.output.mkdir(parents=True)
    external_hashes = {str(p.relative_to(args.source.resolve())): digest(p) for p in sorted(external_paths)}
    if external_hashes != export.get("external_weights_sha256"):
        raise ValueError("External weight fingerprints changed after export")
    engine = args.output / "sampler.engine"
    command = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine}",
        "--stronglyTyped",
        "--skipInference",
        f"--builderOptimizationLevel={args.optimization_level}",
        "--maxAuxStreams=0",
        "--memPoolSize=workspace:8192",
        "--profilingVerbosity=detailed",
        f"--exportLayerInfo={args.output / 'layers.json'}",
        f"--timingCacheFile={args.output / 'timing.cache'}",
    ]
    if not args.tf32_experiment:
        command.insert(4, "--noTF32")
    report = {
        "run_id": args.output.name,
        "phase": "engine_build_tactic_profile",
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "status": "started",
        "source_export": str(args.source),
        "source_export_report_sha256": digest(args.source / "export_report.json"),
        "source_onnx_sha256": digest(onnx_path),
        "external_weights_sha256": external_hashes,
        "tensorrt_version": importlib.metadata.version("tensorrt"),
        "command": command,
        "tf32": args.tf32_experiment,
        "precision_candidate_kind": "tf32" if args.tf32_experiment else "bf16" if args.bf16_experiment else None,
        "quantization": None,
        "strongly_typed": True,
        "contract": export["contract"],
    }
    report_path = args.output / "engine_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    started = time.monotonic()
    code = subprocess.call(command)
    report.update(exit_code=code, build_s=time.monotonic() - started)
    if code == 0:
        report.update(
            status=(
                "built_experiment_not_accuracy_validated"
                if (args.tf32_experiment or args.bf16_experiment)
                else "built_not_inference_or_accuracy_validated"
            ),
            engine_sha256=digest(engine),
        )
    else:
        report["status"] = "build_failed"
    report["finished_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
