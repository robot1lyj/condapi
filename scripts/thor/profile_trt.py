"""Profile the existing engine on real local inputs, separately from latency scores."""

import argparse
from collections import defaultdict
import dataclasses
import datetime
import gzip
import json
from pathlib import Path

from benchmark_pi05 import digest
from benchmark_pi05 import read_observation
from benchmark_suite import checked_path
import numpy as np
import torch
from trt_policy import create_trt_policy

from openpi.shared import normalize
from openpi.training import config


def aggregate_layers(rows, invocation_count, layer_metadata):
    if invocation_count <= 0 or not rows:
        raise ValueError("Nonempty profiling invocations required")
    totals, counts = defaultdict(float), defaultdict(int)
    for name, milliseconds in rows:
        if not np.isfinite(milliseconds) or milliseconds < 0:
            raise ValueError("Invalid layer timing")
        totals[name] += milliseconds
        counts[name] += 1
    layers = [
        {
            "name": name,
            "layer_type": layer_metadata.get(name, {}).get("LayerType", "unmatched"),
            "mean_ms_per_inference": total / invocation_count,
            "callback_count": counts[name],
        }
        for name, total in totals.items()
    ]
    layers.sort(key=lambda row: row["mean_ms_per_inference"], reverse=True)
    return layers


def main():
    import tensorrt as trt  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new immutable profiling output")
    suite = json.loads(args.suite.read_text())
    root = args.suite.parent
    norm = checked_path(root, suite["norm_stats"])
    if suite.get("source_kind") != "real_yam_recording" or digest(norm) != suite["norm_stats_sha256"]:
        raise ValueError("Real suite with matching norm required")
    samples = []
    for entry in suite["samples"]:
        path = checked_path(root, entry["sample"])
        provenance = json.loads(checked_path(root, entry["provenance"]).read_text())
        if (
            digest(path) != provenance["sample_sha256"]
            or provenance["norm_stats_sha256"] != digest(norm)
            or provenance["source_files"] != suite["source_files"]
        ):
            raise ValueError("Sample provenance mismatch")
        samples.append((path, read_observation(path)))
    if not samples or len(samples) != suite["observation_count"]:
        raise ValueError("Suite observation count mismatch")
    args.output.mkdir(parents=True)
    report = {
        "run_id": args.output.name,
        "status": "started",
        "phase": "layer_profile_not_latency_benchmark",
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "suite_sha256": digest(args.suite),
        "norm_stats_sha256": digest(norm),
        "scope": "three real RGB views, H50, ten denoising steps; no robot IO",
        "limitations": [
            "Profiler instrumentation changes timing; do not merge into benchmark latency results",
            "Layer times describe the compiled engine, not necessarily individual original ONNX operators",
            "Summed layer times are not host end-to-end latency; categories may include fused operations",
        ],
    }
    report_path = args.output / "profile_report.json"

    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    save()
    train = config.get_config("pi05_yam")
    train = dataclasses.replace(train, model=dataclasses.replace(train.model, dtype="bfloat16"))
    policy = create_trt_policy(train, args.engine, normalize.deserialize_json(norm.read_text()))
    model = policy._model  # noqa: SLF001
    engine_report = model.build_report
    source = Path(engine_report["source_export"]) / "export_report.json"
    if digest(source) != engine_report["source_export_report_sha256"]:
        raise ValueError("Export report fingerprint mismatch")
    export = json.loads(source.read_text())
    if any(report[key] != export[key] for key in ("suite_sha256", "norm_stats_sha256")):
        raise ValueError("Engine/suite mismatch")
    report["engine_report"] = engine_report
    report["tensorrt_version"] = trt.__version__
    report["layers_sha256"] = digest(args.engine / "layers.json")
    metadata = json.loads((args.engine / "layers.json").read_text())
    metadata = {row["Name"]: row for row in metadata["Layers"]}
    noise = np.random.default_rng(0).standard_normal((50, 32)).astype(np.float32)
    np.save(args.output / "noise.npy", noise, allow_pickle=False)
    if digest(args.output / "noise.npy") != export["noise_sha256"]:
        raise ValueError("Noise mismatch")
    captured = {}
    transform = policy._output_transform  # noqa: SLF001

    def capture(data):
        captured["raw"] = np.asarray(data["actions"], dtype=np.float32).copy()
        return transform(data)

    policy._output_transform = capture  # noqa: SLF001
    references = []
    for path, observation in samples:
        action = policy.infer(observation, noise=noise)["actions"].copy()
        references.append((action, captured["raw"].copy()))
        print("PROFILE_REFERENCE_OK", path.name, flush=True)

    class LayerProfiler(trt.IProfiler):
        def __init__(self):
            trt.IProfiler.__init__(self)
            self.rows = []

        def report_layer_time(self, layer_name, milliseconds):
            self.rows.append([layer_name, float(milliseconds)])

    profiler = LayerProfiler()
    model.context.profiler = profiler
    model.context.enqueue_emits_profile = True
    report["comparisons"] = []
    all_rows = []
    for (path, observation), (reference, raw_reference) in zip(samples, references, strict=True):
        profiler.rows.clear()
        outputs = []
        for _ in range(3):
            action = policy.infer(observation, noise=noise)["actions"]
            raw = captured["raw"]
            if (
                not np.isfinite(raw).all()
                or not np.array_equal(action, reference)
                or not np.array_equal(raw, raw_reference)
            ):
                raise ValueError("Profiling changed the unprofiled engine output")
            outputs.append(raw.copy())
        torch.cuda.synchronize()
        if not profiler.rows:
            raise RuntimeError("TensorRT returned no per-layer measurements")
        with gzip.open(args.output / f"{path.stem}.layers.json.gz", "wt") as stream:
            json.dump(profiler.rows, stream)
        np.savez_compressed(args.output / path.name, reference=raw_reference, profiled=np.stack(outputs))
        all_rows.extend(profiler.rows)
        report["comparisons"].append({"sample": path.name, "exact": True, "profiled_calls": 3})
        save()
        print("PROFILE_SAMPLE_OK", path.name, len(profiler.rows), flush=True)
    layers = aggregate_layers(all_rows, len(samples) * 3, metadata)
    categories = defaultdict(float)
    for row in layers:
        categories[row["layer_type"]] += row["mean_ms_per_inference"]
    report.update(
        status="profiled_not_latency_or_accuracy_approved",
        profiled_calls=len(samples) * 3,
        layers=layers,
        layer_types_mean_ms=dict(sorted(categories.items(), key=lambda pair: pair[1], reverse=True)),
        summed_layers_mean_ms=sum(row["mean_ms_per_inference"] for row in layers),
        finished_at=datetime.datetime.now(datetime.UTC).isoformat(),
    )
    save()
    print("PROFILE_COMPLETE", report["layer_types_mean_ms"], flush=True)


if __name__ == "__main__":
    main()
