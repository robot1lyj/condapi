"""Transport regression after retiring legacy RL; no model or training loop."""

import pytest

from openpi.serving.websocket_policy_server import RequestError
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.serving.websocket_policy_server import _split_payload


class BasePolicy:
    def infer(self, obs):
        return {"actions": obs["sentinel"]}


@pytest.mark.parametrize("legacy", [None, {"mode": "off"}, {"mode": "shadow"}])
def test_base_and_legacy_shadow_preserve_observation_and_rtc(legacy):
    obs = {"sentinel": [1, 2]}
    rtc = {"delay_steps": 0}
    envelope = {"type": "infer", "obs": obs, "rtc": rtc}
    if legacy is not None:
        envelope["parts"] = legacy
    parsed, prefix = _split_payload(envelope)
    assert parsed is obs
    assert prefix is rtc
    server = WebsocketPolicyServer(BasePolicy(), rtc_mode="off")
    assert server._infer(parsed, None) == ({"actions": [1, 2]}, False, [], None)  # noqa: SLF001
    assert "parts" not in server._metadata  # noqa: SLF001


@pytest.mark.parametrize("mode", ["collect", "eval", "unknown", None])
def test_retired_rl_requests_are_rejected_before_inference(mode):
    with pytest.raises(RequestError, match="retired"):
        _split_payload({"obs": {}, "parts": {"mode": mode}})


@pytest.mark.parametrize("legacy", [[], "shadow", 1])
def test_malformed_retired_extension_is_rejected(legacy):
    with pytest.raises(RequestError, match="object"):
        _split_payload({"obs": {}, "parts": legacy})


def test_plain_observation_remains_supported():
    obs = {"sentinel": [3]}
    assert _split_payload(obs) == (obs, None)


@pytest.mark.parametrize("payload", [[], {"rtc": {}}, {"obs": [], "rtc": {}}, {"obs": {}, "rtc": []},
                                     {"type": "train", "obs": {}}])
def test_invalid_envelopes_remain_rejected(payload):
    with pytest.raises(RequestError):
        _split_payload(payload)


def test_trained_rtc_preserves_prefix_and_never_falls_back():
    obs, prefix = {"sentinel": [4]}, {"delay_steps": 3}

    class TrainedPolicy(BasePolicy):
        def infer_rtc(self, received, rtc):
            assert received is obs
            assert rtc is prefix
            return {"actions": [5]}

    server = WebsocketPolicyServer(TrainedPolicy(), rtc_mode="trained")
    assert server._infer(obs, prefix) == ({"actions": [5]}, True, [], None)  # noqa: SLF001
    with pytest.raises(RequestError, match="requires"):
        server._infer(obs, None)  # noqa: SLF001

    class RejectedPolicy(BasePolicy):
        def infer_rtc(self, _obs, _rtc):
            raise ValueError("prefix rejected")

        def infer(self, _obs):
            raise AssertionError("unsafe fallback")

    rejected = WebsocketPolicyServer(RejectedPolicy(), rtc_mode="trained")
    with pytest.raises(RequestError, match="prefix rejected"):
        rejected._infer(obs, prefix)  # noqa: SLF001


def test_auto_rtc_keeps_ordinary_fallback_for_bad_envelope():
    server = WebsocketPolicyServer(BasePolicy(), rtc_mode="auto")
    action, used, _, error = server._infer({"sentinel": [6]}, {})  # noqa: SLF001
    assert action == {"actions": [6]}
    assert not used
    assert "action_horizon" in error
