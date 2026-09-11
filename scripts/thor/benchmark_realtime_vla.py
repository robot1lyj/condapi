"""Offline YAM replay of pinned Dexmal kernels; no robot or production IO."""

import argparse
import datetime
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
import time

from benchmark_pi05 import digest
from benchmark_pi05 import read_observation
from compare_suites import error_metrics
from compare_suites import read_run
import numpy as np
import torch

from openpi import transforms
from openpi.shared import normalize
from openpi.training import config

UPSTREAM = "b86a942a073ea241f9bd6916a705f81906f4638b"
VIEWS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--position-offset", type=int, choices=(-1, 0), default=-1)
    parser.add_argument("--fix-time-broadcast", action="store_true")
    parser.add_argument("--text-capacity", type=int, choices=(80, 200), default=200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("This replay must run on Thor CUDA")
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.source))
    from pi05_infer import Pi05Inference  # noqa: PLC0415

    if args.fix_time_broadcast:
        import pi05_infer  # noqa: PLC0415

        original_time_mlp = pi05_infer.matmul_1_1024_1024_bias_silu

        def broadcast_time_mlp(x, weight, bias, out):
            original_time_mlp(x, weight, bias, out)
            if x.shape[0] == 1 and out.shape[0] > 1:
                out[1:].copy_(out[:1].expand_as(out[1:]))

        pi05_infer.matmul_1_1024_1024_bias_silu = broadcast_time_mlp

    suite = json.loads(args.suite.read_text())
    norm_path = args.suite.parent / suite["norm_stats"]
    ref, ref_arrays = read_run(args.reference)
    if ref["suite_sha256"] != digest(args.suite) or ref["norm_stats_sha256"] != digest(norm_path):
        raise ValueError("Reference fixture mismatch")
    noise = np.load(args.reference / "noise.npy", allow_pickle=False)
    cfg = config.get_config("pi05_yam")
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm = normalize.deserialize_json(norm_path.read_text())
    before_model = transforms.compose(
        [
            transforms.InjectDefaultPrompt(None),
            *data.data_transforms.inputs,
            transforms.Normalize(norm, use_quantiles=data.use_quantile_norm),
        ]
    )
    model_input = transforms.compose(data.model_transforms.inputs)
    output_transform = transforms.compose(
        [
            *data.model_transforms.outputs,
            transforms.Unnormalize(norm, use_quantiles=data.use_quantile_norm),
            *data.data_transforms.outputs,
        ]
    )
    record = {
        "upstream_commit": UPSTREAM,
        "position_offset": args.position_offset,
        "fix_time_broadcast": args.fix_time_broadcast,
        "source_sha256": {p.name: digest(p) for p in args.source.glob("*.py")},
        "harness_sha256": digest(__file__),
        "suite_sha256": digest(args.suite),
        "norm_stats_sha256": digest(norm_path),
        "noise_sha256": digest(args.reference / "noise.npy"),
        "weights_sha256": digest(args.bundle / "converted.pkl"),
        "tokenizer_sha256": digest(args.bundle / "tokenizer/tokenizer.model"),
        "reference": ref["run_id"],
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "triton", "transformers", "jax")},
        "command": sys.argv,
        "config": "pi05_yam",
        "views": 3,
        "horizon": 50,
        "steps": 10,
        "text_capacity": args.text_capacity,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "precision": "upstream BF16 weights/buffers with FP32 accumulations; no FP8",
        "adaptations": [
            "local Gemma tokenizer from original SentencePiece asset",
            "canonical YAM transforms instead of upstream example robot transforms",
            "read diffusion_noise output buffer because upstream forward returns None",
        ],
        "acceptance": "offline measurement only; no task success or physical-unit tolerance",
    }
    (args.output / "started.json").write_text(json.dumps(record, indent=2))
    with (args.bundle / "converted.pkl").open("rb") as stream:
        weights = pickle.load(stream)  # Only our locally produced conversion, never arbitrary downloads.
    print("LOAD_MODEL", flush=True)
    model = Pi05Inference(
        weights,
        num_views=3,
        chunk_size=50,
        tokenizer_path=str(args.bundle / "tokenizer"),
        discrete_state_input=True,
        max_tokenize_len=args.text_capacity,
    )
    if set(model.weights) != set(weights) - {"embedding_weight"}:
        raise ValueError("Checkpoint does not cover all upstream model weights")
    if args.position_offset == 0:

        def corrected_positions(prompt_len):
            start = 3 * 256 + prompt_len
            return model._rope_table[start : start + 50]  # noqa: SLF001

        model.get_decoder_rope_weights = corrected_positions
    del weights
    torch.cuda.synchronize()
    print("LOAD_MODEL_OK", flush=True)
    record["time_cache_rows_max_difference"] = {
        key: float((model.buffers[key].float() - model.buffers[key][:, :1].float()).abs().max())
        for key in ("decoder_time_emb",)
    }

    def prepare(obs):
        intermediate = before_model(dict(obs))
        digits = np.digitize(intermediate["state"], bins=np.linspace(-1, 1, 257)[:-1]) - 1
        inputs = model_input(intermediate)
        images = torch.stack([torch.from_numpy(np.asarray(inputs["image"][key])) for key in VIEWS]).cuda()
        if images.dtype != torch.uint8:
            raise ValueError("Expected canonical resized uint8 RGB images")
        # Policy normally does this in Observation.from_dict, after GPU transfer.
        images = images.float() / 255.0 * 2.0 - 1.0
        return inputs, digits, images

    measurements, all_actions, all_raw = [], [], []
    for index, entry in enumerate(suite["samples"]):
        path = args.suite.parent / entry["sample"]
        if digest(path) != ref["measurements"][index]["sample_sha256"]:
            raise ValueError("Sample order/hash mismatch")
        obs = read_observation(path)
        inputs, digits, _ = prepare(obs)
        prompt = obs["prompt"].strip().replace("_", " ")
        text = f"Task: {prompt}, State: {' '.join(map(str, digits.tolist()))};\nAction: "
        actual_tokens = model.tokenizer(text, return_tensors="pt")["input_ids"][0].numpy()
        expected_tokens = inputs["tokenized_prompt"][inputs["tokenized_prompt_mask"]]
        if len(actual_tokens) > args.text_capacity or not np.array_equal(actual_tokens, expected_tokens):
            raise ValueError("Tokenizer mismatch or truncated input")
        if not all(inputs["image_mask"].values()):
            raise ValueError("Upstream assumes three valid cameras")
        actions, raws, latencies = [], [], []
        for repeat in range(args.warmups + args.repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            inputs, digits, images = prepare(obs)
            model.forward(images, torch.from_numpy(noise).cuda(), obs["prompt"], digits)
            raw = model.buffers["diffusion_noise"].float().cpu().numpy().copy()
            action = np.asarray(output_transform({"state": inputs["state"], "actions": raw.copy()})["actions"])
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            if repeat >= args.warmups:
                actions.append(action.copy())
                raws.append(raw)
                latencies.append(elapsed)
        actions, raws = np.stack(actions), np.stack(raws)
        finite = bool(np.isfinite(actions).all() and np.isfinite(raws).all())
        np.savez_compressed(args.output / path.name, actions=actions, normalized_actions=raws)
        measurement = {
            "sample": path.name,
            "sample_sha256": digest(path),
            "token_count": len(actual_tokens),
            "finite": finite,
            "latencies_ms": latencies,
            "p50_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
        }
        if finite:
            measurement["error"] = error_metrics(ref_arrays["actions"][index, 0], actions[0])
            measurement["repeat_max_abs"] = float(np.abs(actions - actions[0]).max())
        measurements.append(measurement)
        all_actions.append(actions)
        all_raw.append(raws)
        print("SAMPLE", json.dumps(measurement), flush=True)
    all_actions, all_raw = np.stack(all_actions), np.stack(all_raw)
    np.save(args.output / "actions.npy", all_actions, allow_pickle=False)
    np.save(args.output / "normalized_actions.npy", all_raw, allow_pickle=False)
    latencies = [t for sample in measurements for t in sample["latencies_ms"]]
    finite = bool(np.isfinite(all_actions).all() and np.isfinite(all_raw).all())
    record.update(
        measurements=measurements,
        finite=finite,
        p50_ms=float(np.median(latencies)),
        p95_ms=float(np.percentile(latencies, 95)),
        finished_at=datetime.datetime.now(datetime.UTC).isoformat(),
        actions_sha256=digest(args.output / "actions.npy"),
        normalized_actions_sha256=digest(args.output / "normalized_actions.npy"),
    )
    if finite:
        record["error_dataset_units"] = error_metrics(ref_arrays["actions"][:, 0], all_actions[:, 0])
        record["error_normalized_14d"] = error_metrics(
            ref_arrays["normalized_actions"][:, 0, :, :14], all_raw[:, 0, :, :14]
        )
    (args.output / "result.json").write_text(json.dumps(record, indent=2))
    print(
        "RESULT",
        json.dumps({k: v for k, v in record.items() if k in ("finite", "p50_ms", "p95_ms", "error_dataset_units")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
