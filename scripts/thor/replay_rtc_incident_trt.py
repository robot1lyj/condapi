"""Offline same-noise JAX/TF32-TensorRT RTC incident comparison on Thor."""

import argparse
import dataclasses
import json
from pathlib import Path

from benchmark_pi05 import digest
from benchmark_pi05 import read_observation
import numpy as np
from rtc_policy import TrainedRtcInference
from rtc_trt_policy import RtcTensorRTAdapter
from rtc_trt_policy import create_rtc_transform_policy

from openpi.shared import normalize
from openpi.training import config

JOINTS = list(range(6)) + list(range(7, 13))


def seam(actions, delay):
    if not 0 < delay < 50:
        return None
    delta = np.abs(actions[delay, JOINTS] - actions[delay - 1, JOINTS])
    return {"max_rad": float(delta.max()), "joint_14d_index": JOINTS[int(delta.argmax())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--jax-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    engine_report = json.loads((args.engine / "engine_report.json").read_text())
    export = json.loads((Path(engine_report["source_export"]) / "export_report.json").read_text())
    manifest = json.loads((args.checkpoint / "rtc_manifest.json").read_text())
    norm_path = args.checkpoint / "assets" / "yam" / "norm_stats.json"
    cases = json.loads(args.cases.read_text())
    references = json.loads((args.jax_reference / "reference_manifest.json").read_text())
    if (
        cases["norm_stats_sha256"] != digest(norm_path)
        or references["cases_sha256"] != digest(args.cases)
        or references["num_steps"] != export["contract"]["steps"]
        or len(cases["cases"]) != len(references["cases"])
    ):
        raise ValueError("Incident JAX/TRT contract mismatch")
    train = config.get_config("pi05_yam")
    train = dataclasses.replace(
        train,
        model=dataclasses.replace(
            train.model,
            dtype=export["compute_dtype"],
            rtc_training_max_delay=manifest["max_delay_steps"],
        ),
    )
    stats = normalize.deserialize_json(norm_path.read_text())
    use_quantiles = train.data.create(train.assets_dirs, train.model).use_quantile_norm
    policy = create_rtc_transform_policy(train, stats)
    backend = RtcTensorRTAdapter(args.engine, max_delay=manifest["max_delay_steps"], allow_experimental=True)
    backend.enable_cuda_graph()
    serving = TrainedRtcInference(
        policy,
        stats,
        max_delay=manifest["max_delay_steps"],
        use_quantiles=use_quantiles,
        num_steps=export["contract"]["steps"],
        sampler=backend,
        text_bucket=export["text_bucket"],
    )
    noise = np.random.default_rng(references["noise_seed"]).standard_normal((1, 50, 32)).astype(np.float32)
    args.output.mkdir(parents=True)
    rows = []
    for index, (case, ref_row) in enumerate(zip(cases["cases"], references["cases"], strict=True)):
        observation = read_observation(args.cases.parent / case["sample"])
        prefix = np.load(args.cases.parent / case["committed_actions"], allow_pickle=False)
        recorded = np.load(args.cases.parent / case["recorded_reply_actions"], allow_pickle=False)
        with np.load(args.jax_reference / ref_row["reference"], allow_pickle=False) as data:
            jax = np.asarray(data["physical"], dtype=np.float32)
        tick = case["observation_policy_tick"]
        rtc = {
            "delay_steps": case["delay_steps"],
            "observation_policy_tick": tick,
            "target_start_tick": tick,
            "committed_start_tick": tick,
            "committed_actions": prefix,
        }
        trt = np.asarray(serving.infer_rtc(observation, rtc, noise=noise)["actions"], dtype=np.float32)
        delay = case["delay_steps"]
        if not np.array_equal(jax[:delay], prefix) or not np.array_equal(trt[:delay], prefix):
            raise ValueError("Replayed prefix was not preserved exactly")
        difference = np.abs(jax.astype(np.float64) - trt.astype(np.float64))
        # Original online noise was not logged: recorded-vs-replay is diagnostic,
        # never treated as the backend conversion error.
        rows.append(
            {
                "request_id": case["request_id"],
                "reply_row": case["reply_row"],
                "observation_row": case["observation_row"],
                "delay_steps": delay,
                "recorded_seam": seam(recorded, delay),
                "jax_seam": seam(jax, delay),
                "trt_seam": seam(trt, delay),
                "jax_trt_max_abs_joint_rad": float(difference[:, JOINTS].max()),
                "jax_trt_p99_abs_joint_rad": float(np.percentile(difference[:, JOINTS], 99)),
                "recorded_vs_same_seed_jax_max_abs_joint_rad": float(
                    np.max(np.abs(recorded[:, JOINTS] - jax[:, JOINTS]))
                ),
                "recorded_original_noise_known": False,
            }
        )
        np.savez_compressed(args.output / f"case-{index:03d}.npz", recorded=recorded, jax=jax, trt=trt)
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    receipt = {
        "scope": "Thor offline real-episode replay; no robot commands",
        "cases_sha256": digest(args.cases),
        "jax_reference_manifest_sha256": digest(args.jax_reference / "reference_manifest.json"),
        "engine_sha256": engine_report["engine_sha256"],
        "noise_seed": references["noise_seed"],
        "limitations": [
            "Original request prefix reconstructed from reply.actions[:d].",
            "Original online noise absent; recorded-vs-replay is not an exact reproduction.",
        ],
        "cases": rows,
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
