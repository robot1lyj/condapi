"""Transport regression after retiring legacy RL; no model or training loop."""

import asyncio

import numpy as np
from openpi_client import msgpack_numpy
import pytest
import websockets

from openpi.serving.websocket_policy_server import RequestError
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.serving.websocket_policy_server import _pld_payload
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


def test_pld_requires_loaded_runtime_and_trained_rtc():
    for mode in ("collect", "eval", "probe", "shadow"):
        with pytest.raises(RequestError, match="not loaded"):
            _pld_payload({"pld": {"mode": mode}}, enabled=False)
    assert _pld_payload({"pld": {"mode": "off"}}, enabled=False) == {"mode": "off"}
    with pytest.raises(ValueError, match="trained RTC"):
        WebsocketPolicyServer(BasePolicy(), rtc_mode="off", pld_runtime=object())
    obs = {"sentinel": [3]}
    assert _split_payload({"obs": obs, "pld": {"mode": "off"}}) == (obs, None)


def test_pld_transport_applies_only_explicit_mode_and_recovers_rejection():
    class Policy:
        calls = 0

        def infer_rtc(self, obs, rtc):
            self.calls += 1
            return {"actions": np.ones((50, 14), dtype=np.float64)}

    class Runtime:
        def __init__(self):
            self.metadata = {"fixture": True}

        def validate_request(self, obs, rtc, request):
            if request.get("invalid_before_inference"):
                raise ValueError("unsupported mode before Pi inference")

        def __call__(self, obs, rtc, request, action):
            if request.get("invalid"):
                raise ValueError("fixture invalid receipt")
            return {"actions": action["actions"] * 2, "pld": {"fixture": True}}

    class Socket:
        remote_address = ("fixture", 0)

        def __init__(self):
            self.results = []
            packer = msgpack_numpy.Packer()
            self.requests = [packer.pack({"obs": {}, "rtc": {}, "pld": p}) for p in (
                {"mode": "collect", "invalid_before_inference": True}, {"mode": "collect", "invalid": True},
                {"mode": "off"}, {"mode": "collect"})]

        async def recv(self):
            if not self.requests:
                raise websockets.ConnectionClosedOK(None, None)
            return self.requests.pop(0)

        async def send(self, payload):
            self.results.append(payload if isinstance(payload, str) else msgpack_numpy.unpackb(payload))

    socket = Socket()
    policy = Policy()
    server = WebsocketPolicyServer(policy, rtc_mode="trained", pld_runtime=Runtime())
    asyncio.run(server._handler(socket))  # noqa: SLF001
    assert "PLD request rejected" in socket.results[1]
    assert "PLD request rejected" in socket.results[2]
    assert np.all(socket.results[3]["actions"] == 1)
    assert np.all(socket.results[4]["actions"] == 2)
    assert socket.results[4]["pld"]["fixture"]
    assert policy.calls == 3
    for result in socket.results[3:]:
        timing = result["server_timing"]
        assert timing["request_decode_ms"] >= 0
        assert timing["base_infer_ms"] >= 0
        assert timing["pld_postprocess_ms"] >= 0
        assert timing["infer_ms"] >= timing["base_infer_ms"] + timing["pld_postprocess_ms"]
    assert socket.results[3]["server_timing"]["pld_postprocess_ms"] == 0


def test_pld_gate_sessions_are_isolated_and_reset_when_off():
    class Policy:
        def infer_rtc(self, obs, rtc):
            return {"actions": np.zeros((50, 14))}

    class Session:
        def __init__(self):
            self.counter = 0

        def reset(self):
            self.counter = 0

        def __call__(self, obs, rtc, request, action):
            self.counter += 1
            return {**action, "pld": {"counter": self.counter}}

    class Runtime:
        @property
        def metadata(self):
            return {"fixture": True}

        def new_session(self):
            return Session()

    class Socket:
        remote_address = ("fixture", 0)

        def __init__(self, modes):
            self.results = []
            packer = msgpack_numpy.Packer()
            self.requests = [packer.pack({"obs": {}, "rtc": {}, "pld": {"mode": mode}}) for mode in modes]

        async def recv(self):
            if not self.requests:
                raise websockets.ConnectionClosedOK(None, None)
            return self.requests.pop(0)

        async def send(self, payload):
            self.results.append(msgpack_numpy.unpackb(payload))

    server = WebsocketPolicyServer(Policy(), rtc_mode="trained", pld_runtime=Runtime())
    first, second = Socket(["collect", "off", "collect"]), Socket(["collect"])
    asyncio.run(server._handler(first))  # noqa: SLF001
    asyncio.run(server._handler(second))  # noqa: SLF001
    assert first.results[1]["pld"]["counter"] == 1
    assert "pld" not in first.results[2]
    assert first.results[3]["pld"]["counter"] == 1
    assert second.results[1]["pld"]["counter"] == 1
