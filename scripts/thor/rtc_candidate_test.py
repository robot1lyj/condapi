"""Small CPU contract tests; no model checkpoint, GPU run, or training loop."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from rtc_action_space import decode_model_actions
from rtc_action_space import encode_committed_actions
from rtc_onnx_sampler import CachedRtcProjection
from rtc_onnx_sampler import Pi05RtcOnnxSampler
from rtc_onnx_sampler import validate_trained_prefix
from rtc_norm_identity import checkpoint_norm_identity
from rtc_policy import validate_request
from serve_pi05_rtc_trt import check_validated_tf32_7step

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
from openpi.shared.normalize import NormStats
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pi05 = True
        self.attention_implementation = "eager"
        self.config = SimpleNamespace(action_horizon=50, action_dim=32)
        self.action_in_proj = nn.Linear(32, 8)
        self.time_mlp_in = nn.Linear(8, 8)
        self.time_mlp_out = nn.Linear(8, 8)
        language = SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager"))
        self.paligemma_with_expert = SimpleNamespace(
            paligemma=SimpleNamespace(language_model=language),
            forward=lambda **_: (None, None),
        )
        self.times_seen = []

    def embed_prefix(self, *args):
        return torch.zeros(1, 1, 8), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.bool)

    def _preprocess_observation(self, observation, train=False):
        return None, None, None, None, observation.state

    def _prepare_attention_masks_4d(self, value):
        return value[:, None]

    def denoise_step(self, state, pad, cache, actions, time, *, time_embedding=None):
        self.times_seen.append(time.clone())
        assert tuple(time.shape) == (1, 50)
        if time_embedding is not None:
            assert tuple(time_embedding.shape) == (1, 50, 8)
        return torch.ones_like(actions)


class RtcCandidateTest(unittest.TestCase):
    def test_seven_step_tf32_service_requires_matching_validation(self):
        report = {
            "status": "built_experiment_not_accuracy_validated", "exit_code": 0,
            "precision_candidate_kind": "tf32", "tf32": True, "quantization": None,
            "strongly_typed": True, "engine_sha256": "engine",
        }
        export = {
            "compute_dtype": "float32", "contract": {"steps": 7},
            "jax_reference_manifest_sha256": "jax",
        }
        manifest = {"model_weights_sha256": "weights"}
        validation = {
            "status": "compared_not_robot_task_validated", "engine_sha256": "engine",
            "engine_tf32": True, "num_steps": 7,
            "checkpoint_weights_sha256": "weights", "jax_reference_manifest_sha256": "jax",
            "physical_max_abs": 0.0009,
            "cases": [
                {"delay_steps": delay, "finite": True, "prefix_exact": True}
                for delay in (0, 1, 10) for _ in range(3)
            ],
        }
        check_validated_tf32_7step(report, export, manifest, validation)
        with self.assertRaises(ValueError):
            check_validated_tf32_7step(report, export, manifest, {**validation, "engine_sha256": "other"})
        with self.assertRaises(ValueError):
            check_validated_tf32_7step(report, export, manifest, {**validation, "physical_max_abs": 0.0021})
        with self.assertRaises(ValueError):
            check_validated_tf32_7step(report, {**export, "contract": {"steps": 8}}, manifest, validation)

    def test_norm_identity_allows_only_the_missing_final_newline(self):
        source = b'{"action":{"mean":[1]}}\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "norm_stats.json"
            path.write_bytes(source[:-1])
            expected = hashlib.sha256(source).hexdigest()
            self.assertEqual(
                checkpoint_norm_identity(path, expected)["identity_relation"],
                "checkpoint_omitted_one_final_newline",
            )
            path.write_bytes(b'{"action":{"mean":[2]}}')
            with self.assertRaises(ValueError):
                checkpoint_norm_identity(path, expected)

    def test_physical_request_requires_aligned_committed_ticks(self):
        obs = {
            "observation.state": np.zeros(14, dtype=np.float32),
            "prompt": "pick up block",
            **{f"observation.images.{view}": np.zeros((224, 224, 3), dtype=np.uint8)
               for view in ("top_rgb", "left_rgb", "right_rgb")},
        }
        rtc = {
            "delay_steps": 3,
            "observation_policy_tick": 103,
            "target_start_tick": 103,
            "committed_start_tick": 103,
            "committed_actions": np.ones((3, 14), dtype=np.float32),
        }
        state, prefix, delay = validate_request(obs, rtc, max_delay=10)
        self.assertEqual((state.shape, prefix.shape, delay), ((14,), (3, 14), 3))
        _, empty, zero_delay = validate_request(
            obs, {"delay_steps": 0, "observation_policy_tick": 103, "target_start_tick": 103,
                  "committed_start_tick": 103, "committed_actions": []}, max_delay=10
        )
        self.assertEqual((empty.shape, zero_delay), ((0, 14), 0))
        with self.assertRaisesRegex(ValueError, "committed_start_tick"):
            validate_request(obs, {**rtc, "committed_start_tick": 102}, max_delay=10)
        with self.assertRaisesRegex(ValueError, r"action\[0\]"):
            validate_request(obs, {**rtc, "target_start_tick": 104, "committed_start_tick": 104}, max_delay=10)
        with self.assertRaisesRegex(ValueError, "trained range"):
            validate_request(obs, {**rtc, "delay_steps": 11}, max_delay=10)
        with self.assertRaisesRegex(ValueError, "committed_actions"):
            validate_request(obs, {**rtc, "committed_actions": np.ones((2, 14))}, max_delay=10)

    def test_time_embedding_keeps_scalar_path_and_supports_per_token(self):
        time = torch.tensor([0.2, 0.8])
        fraction = torch.linspace(0, 1, 4, dtype=torch.float64)
        periods = 4e-3 * (4.0 / 4e-3) ** fraction
        old = (1 / periods * 2 * torch.pi)[None, :] * time[:, None]
        expected = torch.cat((old.sin(), old.cos()), dim=1)
        scalar = create_sinusoidal_pos_embedding(time, 8, 4e-3, 4.0, device=time.device)
        self.assertTrue(torch.equal(scalar, expected))
        token = create_sinusoidal_pos_embedding(time[:, None].expand(2, 50), 8, 4e-3, 4.0, device=time.device)
        self.assertTrue(torch.equal(token, scalar[:, None, :].expand(2, 50, 8)))

    def test_adaptive_norm_changes_only_selected_token(self):
        norm = GemmaRMSNorm(8, cond_dim=8).float().eval()
        with torch.no_grad():
            norm.dense.weight.fill_(0.03)
            norm.dense.bias.zero_()
        x = torch.arange(1, 49, dtype=torch.float32).reshape(1, 6, 8)
        cond = torch.ones(1, 8)
        old, _ = norm(x, cond)
        same, _ = norm(x, cond[:, None, :].expand(1, 6, 8))
        self.assertTrue(torch.allclose(old, same))
        changed_cond = cond[:, None, :].expand(1, 6, 8).clone()
        changed_cond[:, 2, :] = 0
        changed, _ = norm(x, changed_cond)
        self.assertTrue(torch.equal(changed[:, 0], old[:, 0]))
        self.assertFalse(torch.equal(changed[:, 2], old[:, 2]))

    def test_clean_prefix_is_frozen_and_postfix_is_denoised(self):
        model = FakeModel().eval()
        sampler = Pi05RtcOnnxSampler(model, cache_time_modulation=False, text_bucket=80).eval()
        values = (
            torch.zeros(1, 9, 224, 224),
            torch.ones(1, 3, dtype=torch.bool),
            torch.zeros(1, 80, dtype=torch.int64),
            torch.ones(1, 80, dtype=torch.bool),
            torch.zeros(1, 32),
            torch.ones(1, 50, 32),
            torch.full((1, 50, 32), 0.25),
        )
        mask = (torch.arange(50) < 7)[None]
        result = sampler(*values, mask)
        eager = PI0Pytorch.sample_actions_trained_rtc(
            model, "cpu", SimpleNamespace(state=torch.zeros(1, 32)),
            previous_actions=values[-1], prefix_mask=mask, noise=values[5], num_steps=10,
        )
        self.assertTrue(torch.equal(result, eager))
        self.assertTrue(torch.equal(result[:, :7], values[-1][:, :7]))
        self.assertTrue(torch.allclose(result[:, 7:], torch.zeros_like(result[:, 7:]), atol=2e-7))
        self.assertEqual(len(model.times_seen), 20)
        for seen in model.times_seen:
            self.assertTrue(torch.equal(seen[:, :7], torch.zeros_like(seen[:, :7])))
        self.assertEqual(validate_trained_prefix(values[-1], mask, max_delay=10), 7)
        with self.assertRaises(ValueError):
            validate_trained_prefix(values[-1], mask.roll(1), max_delay=10)
        with self.assertRaises(ValueError):
            validate_trained_prefix(values[-1], mask, max_delay=6)

    def test_official_and_intermediate_denoising_steps_keep_rtc_prefix(self):
        for num_steps in (5, 6, 7, 8):
            with self.subTest(num_steps=num_steps):
                model = FakeModel().eval()
                sampler = Pi05RtcOnnxSampler(model, num_steps=num_steps).eval()
                values = (
                    torch.zeros(1, 9, 224, 224),
                    torch.ones(1, 3, dtype=torch.bool),
                    torch.zeros(1, 200, dtype=torch.int64),
                    torch.ones(1, 200, dtype=torch.bool),
                    torch.zeros(1, 32),
                    torch.ones(1, 50, 32),
                    torch.full((1, 50, 32), 0.25),
                )
                mask = (torch.arange(50) < 7)[None]
                actual = sampler(*values, mask)
                reference = PI0Pytorch.sample_actions_trained_rtc(
                    model, "cpu", SimpleNamespace(state=torch.zeros(1, 32)),
                    previous_actions=values[-1], prefix_mask=mask, noise=values[5], num_steps=num_steps,
                )
                self.assertTrue(torch.equal(actual, reference))
                self.assertTrue(torch.equal(actual[:, :7], values[-1][:, :7]))
                self.assertEqual(len(model.times_seen), 2 * num_steps)

    def test_rtc_cache_selects_clean_or_step_value(self):
        linear = nn.Linear(8, 24).float().eval()
        clean = torch.randn(1, 8)
        steps = [torch.randn(1, 8) for _ in range(10)]
        cache = CachedRtcProjection(linear, clean, steps).eval()
        cache.step = 3
        cache.mask = (torch.arange(50) < 4)[None]
        output = cache(torch.zeros(1, 50, 8))
        self.assertTrue(torch.equal(output[:, :4], cache.clean[:, None, :].expand(1, 4, 24)))
        self.assertTrue(torch.equal(output[:, 4:], cache.table[3][:, None, :].expand(1, 46, 24)))

    def test_yam_absolute_prefix_round_trip_with_current_state(self):
        stats = {
            "state": NormStats(mean=np.zeros(14), std=np.ones(14), q01=-np.ones(14), q99=np.ones(14)),
            "actions": NormStats(mean=np.zeros(14), std=np.ones(14), q01=-np.ones(14), q99=np.ones(14)),
        }
        state = np.linspace(-0.3, 0.3, 14, dtype=np.float32)
        absolute = np.tile(state, (50, 1)).astype(np.float32)
        absolute[:, 0] += 0.07
        absolute[:, 6] = 0.8
        absolute[:, 13] = 0.2
        for quantile in (False, True):
            encoded = encode_committed_actions(absolute, state, stats, use_quantiles=quantile)
            self.assertEqual(encoded.shape, (50, 32))
            self.assertTrue(np.all(encoded[:, 14:] == 0))
            restored = decode_model_actions(encoded, state, stats, use_quantiles=quantile)
            np.testing.assert_allclose(restored, absolute, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
