"""Thor-only native XR-1 latency probe; never sends commands to a robot.

Uses unmodified upstream forward, 30x60 output and five Euler steps. Real YAM
RGB fixtures exercise the vision path; state/prefix are explicitly normalized
model-space fixtures, NOT valid physical YAM end-effector commands or stats.
"""

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor

from mibot.models.VLA.XR1 import xr1
from mibot.server.runtime.client import Client
from mibot.utils.io import build_action_mask, resize_image


def summary(values):
    return {"p50_ms": statistics.median(values),
            "p95_ms": float(np.percentile(values, 95)),
            "min_ms": min(values), "max_ms": max(values), "samples_ms": values}


def tree_clone(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, (tuple, list)):
        return type(value)(tree_clone(x) for x in value)
    raise TypeError(type(value))


def tree_update(target, value):
    if isinstance(target, torch.Tensor):
        if (target.shape, target.dtype, target.device) != (value.shape, value.dtype, value.device):
            raise ValueError("CUDA Graph input shape/dtype/device changed")
        target.copy_(value)
    else:
        for a, b in zip(target, value, strict=True):
            tree_update(a, b)


class GraphDiT:
    """One fixed-d graph; copies ALL current observation KV and prefix inputs."""

    def __init__(self, original, prefix_length):
        self.original = original
        self.prefix_length = prefix_length
        self.graph = None

    def __call__(self, noisy_action, timestep, action_mask, state_embed,
                 position_embeds, past_key_values, attn_mask, prefix_length):
        if prefix_length != self.prefix_length:
            raise ValueError("Use a separate graph for each prefix length")
        supplied = (noisy_action, timestep, action_mask, state_embed,
                    position_embeds, tuple(past_key_values), attn_mask)
        if self.graph is None:
            self.static = tree_clone(supplied)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.original(*self.static, prefix_length=prefix_length)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.result = self.original(*self.static, prefix_length=prefix_length)
        tree_update(self.static, supplied)
        self.graph.replay()
        return self.result


def prepare(processor, fixtures, size):
    prepared = []
    for index, fixture in enumerate(fixtures):
        with np.load(fixture, allow_pickle=False) as data:
            images = [Image.fromarray(data[f"observation.images.{key}_rgb"].copy())
                      for key in ("top", "left", "right")]
            prompt = str(data["prompt"].item())
        # A larger profile measures token-count cost, not recovered image detail.
        if size:
            images = [x.resize((size, size)) for x in images]
        images = [resize_image(x, factor=32, max_pixels=160000) for x in images]
        messages = [Client._messages(prompt, *images)]

        def preprocess(messages=messages, index=index):
            batch = dict(processor.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt",
                padding=True, images_kwargs={"do_resize": False}))
            # No Pi normalization or joint-action transformation is borrowed.
            rng = np.random.default_rng(100 + index)
            state = rng.uniform(-0.2, 0.2, (1, 1, 60)).astype(np.float32)
            state[:, :, 16:] = 0
            state[:, :, [6, 14]] = 0  # YAM has six joints per arm, not seven.
            mask = build_action_mask(30)[None]
            batch["state"] = torch.from_numpy(state)
            batch["action"] = torch.from_numpy(rng.uniform(-0.1, 0.1, (1, 30, 60)).astype(np.float32) * mask)
            batch["action_mask"] = torch.from_numpy(mask)
            return {k: v.cuda().to(torch.bfloat16 if v.is_floating_point() else v.dtype)
                    if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        batch = preprocess()
        torch.cuda.synchronize()
        prepared.append((str(fixture), batch, preprocess, [list(x.size) for x in images]))
    return prepared


def infer(model, data, d, seed):
    torch.manual_seed(seed)
    # Upstream auto_cast / forward pop entries, so never reuse the dict itself.
    return model(dict(data, prefix_length=d))


def measure(model, prepared, d, warmups, repeats):
    times = []
    for index in range(warmups + repeats):
        _, batch, _, _ = prepared[index % len(prepared)]
        torch.manual_seed(index + 42)
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(dict(batch, prefix_length=d))
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        if index >= warmups:
            times.append(elapsed)
        if not torch.isfinite(output).all() or output.shape != (1, 30, 60):
            raise RuntimeError("Non-finite or incorrect native output")
        if not torch.equal(output[:, :d], batch["action"][:, :d]):
            raise RuntimeError("Native committed prefix was modified")
    return summary(times)


def total_measure(model, prepared, d, repeats):
    times, preprocess_times = [], []
    for index in range(repeats):
        _, _, preprocess, _ = prepared[index % len(prepared)]
        torch.manual_seed(index + 42)
        torch.cuda.synchronize()
        start = time.perf_counter()
        batch = preprocess()
        torch.cuda.synchronize()
        pre_end = time.perf_counter()
        output = model(dict(batch, prefix_length=d)).float().cpu().numpy()
        end = time.perf_counter()
        if not np.isfinite(output).all():
            raise RuntimeError("Non-finite output")
        preprocess_times.append((pre_end - start) * 1000)
        times.append((end - start) * 1000)
    return {"local_preprocess_forward_d2h": summary(times),
            "preprocess_h2d": summary(preprocess_times)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefixes", type=int, nargs="+", default=[0, 1, 3, 6, 8, 10])
    parser.add_argument("--sizes", type=int, nargs="+", default=[224, 384])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if any(d < 0 or d >= 30 for d in args.prefixes) or args.repeats < 3 or args.warmups < 1:
        parser.error("Require 0 <= d < 30, repeats >= 3 and warmups >= 1")
    torch.set_num_threads(8)
    report = {
        "versions": {k: importlib.metadata.version(k) for k in
                     ("torch", "transformers", "flash-attn", "liger-kernel", "mmengine", "lightning")},
        "gpu": torch.cuda.get_device_name(), "dtype": "bfloat16", "steps": 5,
        "action_shape": [1, 30, 60], "checkpoint": str(args.checkpoint),
        "scope": "real RGB; synthetic normalized state/prefix; no IK, robot, training or network RTT",
        "profiles": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    model = xr1(ffn_gradient_checkpointing=False).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state = checkpoint["module"]
    report["checkpoint_tensor_dtypes"] = dict(Counter(str(v.dtype) for v in state.values()))
    report["checkpoint_tensor_count"] = len(state)
    if not all(k.startswith("model.") for k in state):
        raise RuntimeError("Unexpected native checkpoint namespace")
    loaded = model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()}, strict=True)
    report["strict_load"] = str(loaded)
    print(json.dumps({"strict_load": report["strict_load"], "versions": report["versions"]}), flush=True)
    del checkpoint, state
    model.cuda()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", local_files_only=True)
    processor.tokenizer.padding_side = "right"
    fixtures = [args.fixtures / f"episode-{n:06d}-early.npz" for n in (95, 96, 97)]
    original = model.dit_forward
    with torch.inference_mode():
        for size in args.sizes:
            prepared = prepare(processor, fixtures, size)
            profile = {"requested_size": size, "fixtures": [p[0] for p in prepared],
                       "image_sizes": prepared[0][3],
                       "input_shapes": {k: list(v.shape) for k, v in prepared[0][1].items()
                                        if isinstance(v, torch.Tensor)}, "results": []}
            report["profiles"].append(profile)
            for d in args.prefixes:
                model.dit_forward = original
                references = [infer(model, p[1], d, 42 + i).float().cpu()
                              for i, p in enumerate(prepared)]
                result = {"prefix_length": d,
                          "eager": measure(model, prepared, d, args.warmups, args.repeats)}
                profile["results"].append(result)
                print(json.dumps({"size": size, "d": d, "eager": result["eager"]}), flush=True)
                graph = GraphDiT(original, d)
                model.dit_forward = graph
                comparisons = []
                for i, p in enumerate(prepared):
                    output = infer(model, p[1], d, 42 + i).float().cpu()
                    error = (output - references[i]).abs()
                    comparisons.append({"fixture": p[0], "max_abs": float(error.max()),
                                        "mae": float(error.mean()), "bit_equal": bool(torch.equal(output, references[i]))})
                result["graph_vs_eager"] = comparisons
                if not all(c["bit_equal"] for c in comparisons):
                    save()
                    raise RuntimeError("CUDA Graph changed native BF16 output; do not accept optimization")
                result["graph_dit"] = measure(model, prepared, d, args.warmups, args.repeats)
                result.update(total_measure(model, prepared, d, 12))
                # Same noise/observation: verify the suffix actually reads prefix.
                if d:
                    batch = prepared[0][1]
                    changed = dict(batch, action=batch["action"].clone())
                    changed["action"][:, :d] += batch["action_mask"][:, :d] * 0.25
                    a = infer(model, batch, d, 77)
                    b = infer(model, changed, d, 77)
                    result["changed_prefix_suffix_max_abs"] = float((a[:, d:] - b[:, d:]).abs().max())
                    if result["changed_prefix_suffix_max_abs"] == 0:
                        raise RuntimeError("Prefix has no effect on generated suffix")
                result["finite"] = True
                result["prefix_bit_equal"] = True
                print(json.dumps({"size": size, **result}), flush=True)
                save()
                model.dit_forward = original
                del graph
                torch.cuda.empty_cache()
    report["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
    save()


if __name__ == "__main__":
    main()
