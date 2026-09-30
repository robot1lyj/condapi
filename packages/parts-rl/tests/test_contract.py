# ruff: noqa: SLF001, PT011, PT018
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from parts_policy import PartsPolicyExtension
from parts_rl.data import curate_attempts
from parts_rl.data import discounted_return
from parts_rl.state import NON_VISUAL_DIM
from parts_rl.state import encode_state
import pytest
from rtc_prefix import physical_prefix_array
from rtc_prefix import restore_physical_prefix
from vla_platform import parts as wire

from adapters.parts.common import require_server
from openpi.serving.websocket_policy_server import RequestError
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.serving.websocket_policy_server import _split_payload
from openpi.serving.websocket_policy_server import _split_request

ROOT = Path(__file__).resolve().parents[3]


def declaration(modes=None):
    return {
        "protocol": wire.PROTOCOL,
        "contract_sha": "fixture-contract",
        "residual_space": "joint_delta_rad",
        "horizon": 50,
        "state_dim": 14,
        "action_dt": 1 / 30,
        "supported_modes": modes or ["off", "shadow"],
        "feature_schema_id": "fixture-feature",
        "behavior_manifest_ref": "fixture-manifest.json",
        "per_arm": {arm: {"indices": indices, "B_rad": [0.001] * 6} for arm, indices in wire.INDICES.items()},
    }


def sent(mode="shadow", tick=0, request_id=1, arm="left"):
    return {
        "protocol": wire.PROTOCOL,
        "contract_sha": "fixture-contract",
        "mode": mode,
        "active_arm": arm,
        "context": {
            "run_id": "fixture",
            "session_id": "session",
            "epoch": 0,
            "request_id": request_id,
            "observation_id": request_id,
            "observation_policy_tick": tick,
        },
        "arms": {
            side: {
                "phase": "ACTIVE_DESCENT" if side == arm else "READY",
                "eligible": True,
                "attempt_id": "a1" if side == arm else None,
                "pose_valid": True,
                "height_valid": True,
                "height_m": 0.05,
                "h_goal_m": 0.01,
                "error_m": 0.04,
                "force_valid": False,
                "effort_nm": None,
                "elapsed_s": tick / 30,
                "confirmation_s": 0.0,
            }
            for side in wire.INDICES
        },
        "scheduler": {
            "targets": np.zeros((50, 14)),
            "valid_mask": np.zeros(50, bool),
            "committed_mask": np.zeros(50, bool),
        },
    }


class Features:
    z = np.array([0.25, 0.5], np.float32)

    @contextmanager
    def capture(self):
        yield self


def manifest(modes=None):
    return {
        "contract": declaration(modes),
        "behavior_snapshot_id": "fixture-zero",
        "feature_dim": 2,
        "state_schema": wire.STATE_SCHEMA,
        "exploration_std": 0.1,
        "exploration_seed": 7,
        "actors": {
            arm: {"initialization": "zero_residual", "actor_snapshot_id": "zero-" + arm} for arm in wire.INDICES
        },
    }


class Base:
    def __init__(self):
        self.calls = 0
        self.actions = np.full((50, 14), 0.123456789, np.float64)

    def infer(self, obs):
        self.calls += 1
        return {"actions": self.actions}

    def infer_rtc(self, obs, rtc):
        if rtc.get("invalid"):
            raise ValueError("unaligned")
        return {"actions": self.actions}


def test_legacy_off_and_unsupported_collect():
    obs = {"observation.state": np.zeros(14)}
    assert _split_payload(obs) == (obs, None)
    assert _split_request({"type": "infer", "obs": obs, "parts": sent()})[2]["mode"] == "shadow"
    with pytest.raises(RequestError):
        _split_request({"type": "infer", "obs": obs, "parts": []})
    base = Base()
    server = WebsocketPolicyServer(base, rtc_mode="off")
    out, used, _, _ = server._infer_with_parts(obs, None, sent("off"))
    assert out["actions"] is base.actions and not used
    assert "parts" not in server._infer_with_parts(obs, None, sent())[0]
    with pytest.raises(RequestError, match="unsupported"):
        server._infer_with_parts(obs, None, sent("collect"))


def test_trained_rtc_rejection_never_calls_ordinary():
    base = Base()
    server = WebsocketPolicyServer(base, rtc_mode="trained")
    with pytest.raises(RequestError):
        server._infer_with_parts({}, {"invalid": True}, sent("off"))
    assert base.calls == 0


def test_shadow_preserves_float64_actions_and_prefix():
    base = Base()
    extension = PartsPolicyExtension(manifest())
    server = WebsocketPolicyServer(base, rtc_mode="trained", parts_extension=extension)
    out, used, _, _ = server._infer_with_parts({}, {"delay_steps": 3, "observation_policy_tick": 0}, sent())
    assert used and out["actions"] is base.actions
    np.testing.assert_array_equal(out["actions"], base.actions)
    for candidate in out["parts"]["candidates"].values():
        assert not candidate["exploration_applied"]
        assert not candidate["editable_mask"][:3].any()
        assert not candidate["u"].any()
    assert out["parts"]["features"]["status"] == "missing"


def test_collect_deterministic_once_and_queue_required():
    m = manifest(["off", "shadow", "collect"])
    base = Base()
    obs = {"observation.state": np.zeros(14)}
    extensions = [PartsPolicyExtension(copy.deepcopy(m), features=Features()) for _ in range(2)]

    def callback():
        return base.infer(obs), False, [], None

    outputs = [ext.infer(obs, None, sent("collect"), callback)[0]["parts"] for ext in extensions]
    np.testing.assert_array_equal(outputs[0]["candidates"]["left"]["u"], outputs[1]["candidates"]["left"]["u"])
    assert outputs[0]["candidates"]["left"]["u"].any()
    assert outputs[0]["candidates"]["left"]["exploration_applied"]
    with pytest.raises(ValueError, match="already served"):
        extensions[0].infer(obs, None, sent("collect"), callback)
    bad = sent("collect", request_id=2)
    bad["scheduler"]["committed_mask"][0] = True
    with pytest.raises(ValueError, match="scheduling"):
        extensions[0].infer(obs, None, bad, callback)
    with pytest.raises(ValueError, match="visual feature"):
        PartsPolicyExtension(m)
    encoded = encode_state(Features.z, np.zeros(14), base.actions, sent("collect"), "left", feature_dim=2)
    assert encoded.shape == (2 + NON_VISUAL_DIM,)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf")])
def test_nonfinite_height_rejected(bad):
    value = sent()
    value["arms"]["left"]["height_m"] = bad
    with pytest.raises(ValueError):
        wire.request(value, declaration())


def test_exact_prefix_avoids_rounding_and_delay_zero_keeps_dtype():
    old = np.zeros((50, 14), np.float32)
    prefix = np.full((2, 14), 0.123456789, np.float64)
    original = physical_prefix_array(prefix, 2)
    out = restore_physical_prefix(old, original)
    np.testing.assert_array_equal(out[:2], prefix)
    assert out.dtype == np.float64 and old.dtype == np.float32
    assert restore_physical_prefix(old, physical_prefix_array([], 0)) is old
    with pytest.raises(ValueError):
        physical_prefix_array(np.ones((2, 14), bool), 2)


def test_variable_duration_and_curated_attempts():
    assert discounted_return([1, 2, 3], 0.5) == 2.75
    labels = [{"attempt_id": "s", "result": "success", "split_role": "train", "mock": False}]
    labels += [
        {"attempt_id": "f" + str(i), "result": "failure", "split_role": "train", "mock": False} for i in range(4)
    ]
    labels += [
        {"attempt_id": "eval", "result": "success", "split_role": "eval", "mock": False},
        {"attempt_id": "mock", "result": "success", "split_role": "train", "mock": True},
        {"attempt_id": "canceled", "result": "canceled", "split_role": "train", "mock": False},
    ]
    selected = curate_attempts(labels, 0.5, 2)
    assert len(selected) == 3 and "s" in selected and set(selected).issubset({"s", "f0", "f1", "f2", "f3"})
    assert selected == curate_attempts(labels, 0.5, 2)


def test_server_guard_and_check_only_have_no_ml_imports(tmp_path):
    with pytest.raises(ValueError, match="listed GPU server"):
        require_server({"allowed_training_hosts": []}, execute_on_server=True)

    recipe = json.loads((ROOT / "configs/parts/train_recipe.example.json").read_text())
    recipe["replay"] = {"left": "not-loaded-left/READY.json", "right": "not-loaded-right/READY.json"}
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    code = (
        "import sys; from adapters.parts.train import main; "
        'main(sys.argv[1:]); assert "torch" not in sys.modules; assert "jax" not in sys.modules'
    )
    result = subprocess.run(
        [sys.executable, "-c", code, "--recipe", str(path), "--output", str(tmp_path / "out"), "--check-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "out").exists()


def test_reward_scope_is_explicit_and_grasp_is_terminal_component():
    recipe = {
        "schema": "fixture",
        "formula": "negative_absolute_height_error_v1",
        "height_scope": "active_descent",
        "height_weight": 2.0,
        "grasp_weight": 1.0,
    }
    assert wire.height_reward(recipe, 0.03)["total_reward"] == -0.06
    assert wire.height_reward(recipe, 0.03, 1, phase="WAIT_REARM")["total_reward"] == 1.0
    del recipe["height_scope"]
    with pytest.raises(ValueError, match="height_scope"):
        wire.height_reward(recipe, 0.03)


def test_zero_manifest_needs_explicit_feature_bounds_and_no_eval(tmp_path):
    from scripts.parts.build_behavior_manifest import build  # noqa: PLC0415

    identity = {"checkpoint_weights_sha256": "a" * 64, "norm_stats_sha256": "b" * 64}
    output = tmp_path / "behavior"
    result = build(output, declaration(["shadow", "collect"]), identity, feature_dim=2, exploration_std=0.1, seed=0)
    value = json.loads((output / "behavior.json").read_text())
    assert result["behavior_snapshot_id"] == value["behavior_snapshot_id"]
    assert all(item["initialization"] == "zero_residual" for item in value["actors"].values())
    with pytest.raises(ValueError, match="learned actors"):
        build(tmp_path / "eval", declaration(["eval"]), identity, feature_dim=2)
    assert not (tmp_path / "eval").exists()
