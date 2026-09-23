"""Thor-only experimental exact first-frame cache and segmented attention.

Uses the pinned OpenWAM-Alpha model with one synthetic observation. Patches only
the in-memory eager MoT driver during this process; checkpoint/image are read-only.
This is an offline action/latency probe, not a production inference backend.
"""

import copy
import json
import os
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
from openwam.deploy.obs_preprocess import ObsPreprocessor
from openwam.deploy.optimizations import DiTVelocityCache
from openwam.deploy.server import build_server_from_config
from PIL import Image
import torch
from torch.nn import functional


class PrefixExperiment:
    def __init__(self, driver, *, cache_prefix: bool, split_attention: bool):
        self.driver = driver
        self.video = driver.vb
        self.cache_prefix = cache_prefix
        self.split_attention = split_attention
        self.original_step = driver.step
        self.original_pre = self.video.pre_attn_at_layer
        self.original_post = self.video.post_attn_at_layer
        self.original_mixed = driver._mixed_attention  # noqa: SLF001
        self.reset()

    def reset(self):
        self.passes = 0
        self.layer = -1
        self.prefix = None
        self.saved = {}
        self.hit_layers = 0
        self.skipped_attention_queries = 0

    @staticmethod
    def _suffix_state(state, prefix):
        result = copy.copy(state)
        result.hidden_states = state.hidden_states[:, prefix:]
        if state.time_mod.ndim == 4:
            result.time_mod = state.time_mod[:, prefix:]
        result.rope_freqs = state.rope_freqs[prefix:]
        return result

    def step(self, layer_id, vstate, astate, **kwargs):
        if layer_id == 0:
            self.passes += 1
            current_prefix = self.driver._video_tokens_per_frame(vstate)  # noqa: SLF001
            if self.prefix is None:
                self.prefix = current_prefix
            elif self.prefix != current_prefix:
                raise RuntimeError("first-frame token count changed within request")
            if vstate.vace_hints:
                raise RuntimeError("VACE residuals require separate prefix-cache handling")
        self.layer = layer_id
        return self.original_step(layer_id, vstate, astate, **kwargs)

    def pre(self, layer_id, vstate):
        if self.passes == 1:
            q, k, v, post = self.original_pre(layer_id, vstate)
            self.saved[layer_id] = {
                "q": q[:, : self.prefix].detach().clone(),
                "k": k[:, : self.prefix].detach().clone(),
                "v": v[:, : self.prefix].detach().clone(),
            }
            return q, k, v, post
        tail = self._suffix_state(vstate, self.prefix)
        q, k, v, post = self.original_pre(layer_id, tail)
        saved = self.saved[layer_id]
        self.hit_layers += 1
        return (
            torch.cat((saved["q"], q), dim=1),
            torch.cat((saved["k"], k), dim=1),
            torch.cat((saved["v"], v), dim=1),
            post,
        )

    def post(self, layer_id, vstate, attention, post):
        if self.passes == 1:
            vstate = self.original_post(layer_id, vstate, attention, post)
            self.saved[layer_id]["hidden"] = vstate.hidden_states[:, : self.prefix].detach().clone()
            return vstate
        tail = self._suffix_state(vstate, self.prefix)
        tail = self.original_post(layer_id, tail, attention[:, self.prefix :], post)
        vstate.hidden_states = torch.cat((self.saved[layer_id]["hidden"], tail.hidden_states), dim=1)
        return vstate

    def mixed(self, q_cat, k_cat, v_cat, mask):
        if not self.split_attention:
            return self.original_mixed(q_cat, k_cat, v_cat, mask)
        n = self.driver.num_heads
        d = self.driver.head_dim
        sequence = q_cat.shape[1]
        if mask.shape != (sequence, sequence):
            raise RuntimeError("non-standard joint mask shape")
        q = q_cat.reshape(q_cat.shape[0], sequence, n, d).transpose(1, 2)
        k = k_cat.reshape(k_cat.shape[0], sequence, n, d).transpose(1, 2)
        v = v_cat.reshape(v_cat.shape[0], sequence, n, d).transpose(1, 2)
        if self.passes == 1 or not self.cache_prefix:
            head = functional.scaled_dot_product_attention(
                q[:, :, : self.prefix], k[:, :, : self.prefix], v[:, :, : self.prefix]
            )
            if self.cache_prefix:
                self.saved[self.layer]["attention"] = head.detach().clone()
        else:
            head = self.saved[self.layer]["attention"]
            self.skipped_attention_queries += self.prefix
        tail = functional.scaled_dot_product_attention(q[:, :, self.prefix :], k, v)
        return torch.cat((head, tail), dim=2).transpose(1, 2).reshape(q_cat.shape)

    def install(self):
        if self.driver.attention_mask_mode != "mutual" or self.video.video_attention_mask_mode != "first_frame_causal":
            raise RuntimeError("requires mutual/first_frame_causal attention semantics")
        self.driver.step = self.step
        if self.cache_prefix:
            self.video.pre_attn_at_layer = self.pre
            self.video.post_attn_at_layer = self.post
        if self.split_attention:
            self.driver._mixed_attention = self.mixed  # noqa: SLF001

    def restore(self):
        self.driver.step = self.original_step
        self.video.pre_attn_at_layer = self.original_pre
        self.video.post_attn_at_layer = self.original_post
        self.driver._mixed_attention = self.original_mixed  # noqa: SLF001


def as_actions(output):
    actions = output["actions"]
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().float().cpu().numpy()
    actions = np.asarray(actions)
    if actions.shape != (32, 20) or not np.isfinite(actions).all():
        raise ValueError("nonfinite or wrong-shape action output")
    return actions


def compare(value, reference):
    delta = np.abs(value - reference)
    return {
        "max_abs": float(delta.max()),
        "mae": float(delta.mean()),
        "position_max_abs": float(delta[:, [0, 1, 2, 10, 11, 12]].max()),
        "rotation6d_max_abs": float(delta[:, [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18]].max()),
        "gripper_max_abs": float(delta[:, [9, 19]].max()),
        "per_dimension_max": delta.max(axis=0).tolist(),
    }


def main():
    compiled = os.environ.get("OPENWAM_PREFIX_COMPILE", "0") == "1"
    compile_mode = os.environ.get("OPENWAM_COMPILE_MODE", "reduce-overhead")
    if compile_mode not in {"default", "reduce-overhead"}:
        raise ValueError(f"unsupported compile mode: {compile_mode}")
    requested_case = os.environ.get("OPENWAM_PREFIX_CASE", "all")
    if requested_case not in {"all", "baseline", "split_only", "prefix_only", "prefix_split"}:
        raise ValueError(f"unknown case: {requested_case}")
    cfg = OmegaConf.load("/opt/openwam/configs/deploy.yaml")
    cfg.optimization.compile.enabled = False
    cfg.optimization.dit_cache.enabled = False
    cfg.optimization.prompt_embed_cache.enabled = False
    cfg.optimization.decode_video = False
    server = build_server_from_config(cfg, "/checkpoint", device="cuda")
    engine = server.engine
    rng = np.random.default_rng(42)
    images = [Image.fromarray(rng.integers(0, 256, (384, 320, 3), dtype=np.uint8)) for _ in range(3)]
    state = np.zeros(20, dtype=np.float32)
    state[3:9] = state[13:19] = [1, 0, 0, 0, 1, 0]
    obs = ObsPreprocessor.from_cfg(server.cfg, engine).preprocess(
        {
            "images": dict(zip(["head_camera", "left_wrist_camera", "right_wrist_camera"], images, strict=True)),
            "state": state,
            "prompt": "Pick up the object and place it in the tray.",
        }
    )
    conditions = {
        "first_frame_image": [obs["image"]],
        "proprio": state,
        "prompt": obs["prompt"],
        "seed": 42,
        "denoise_steps": 10,
    }
    report = {
        "scope": "one_synthetic_observation_bf16_not_robot_accuracy",
        "compiled": compiled,
        "compile_mode": compile_mode if compiled else None,
        "cases": [],
    }
    for use_dit_cache in (True,) if compiled else (False, True):
        engine._dit_cache = DiTVelocityCache(cosine_threshold=0.99, max_consecutive_skips=3) if use_dit_cache else None  # noqa: SLF001
        reference = None
        for name, use_prefix, use_split in (
            ("baseline", False, False),
            ("split_only", False, True),
            ("prefix_only", True, False),
            ("prefix_split", True, True),
        ):
            if requested_case not in {"all", name}:
                continue
            experiment = PrefixExperiment(
                engine.architecture.mot_driver, cache_prefix=use_prefix, split_attention=use_split
            )
            if not compiled or name != "baseline":
                experiment.install()
            try:
                if compiled:
                    cfg.optimization.compile.enabled = True
                    cfg.optimization.compile.self_attn.torch_mode = compile_mode
                    engine.architecture.apply_compile_optimizations(cfg.optimization.compile)
                times = []
                for repeat in range(5 if compiled else 7):
                    experiment.reset()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        result = engine.generate(conditions)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - started) * 1000)
                    actions = as_actions(result)
                    if compiled and engine.architecture._compiled_mot_run_joint_loop is None:  # noqa: SLF001
                        raise RuntimeError("compiled joint loop fell back to eager")
                    if name != "baseline" and experiment.passes < 2:
                        raise RuntimeError(f"too few full forwards: {experiment.passes}")
                    print(
                        json.dumps({"dit_cache": use_dit_cache, "case": name, "repeat": repeat, "ms": times[-1]}),
                        flush=True,
                    )
                if reference is None:
                    reference = actions.copy()
                row = {
                    "dit_cache": use_dit_cache,
                    "case": name,
                    "full_forwards": experiment.passes,
                    "cache_hit_layers": experiment.hit_layers,
                    "skipped_attention_queries": experiment.skipped_attention_queries,
                    "cold_ms": times[0],
                    "p50_ms": float(np.median(times[1:])),
                    "p95_ms": float(np.percentile(times[1:], 95)),
                    "times_ms": times[1:],
                    "vs_baseline": compare(actions, reference),
                }
                if compiled:
                    row["compiled_loop_alive"] = engine.architecture._compiled_mot_run_joint_loop is not None  # noqa: SLF001
                np.save(Path("/reports") / f"{'dit' if use_dit_cache else 'plain'}-{name}-actions.npy", actions)
                report["cases"].append(row)
                Path("/reports/prefix-cache-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({key: value for key, value in row.items() if key != "times_ms"}), flush=True)
            finally:
                experiment.restore()


if __name__ == "__main__":
    main()
