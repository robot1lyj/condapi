import asyncio
import http
import logging
import time
import traceback
from typing import Any

import numpy as np

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class RequestError(Exception):
    """Recoverable request error that should not drop the websocket connection."""


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        rtc_mode: str = "off",
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._rtc_mode = rtc_mode
        self._metadata = dict(metadata or {})
        self._metadata.setdefault("rtc_mode", self._rtc_mode)
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                payload = msgpack_numpy.unpackb(await websocket.recv())
                obs, rtc_payload = _split_payload(payload)
                if self._rtc_mode == "only" and rtc_payload is None:
                    raise RequestError("RTC payload required when rtc_mode=only.")
                if self._rtc_mode == "off" and rtc_payload is not None:
                    logger.debug("Ignoring rtc payload because rtc_mode=off.")
                    rtc_payload = None

                infer_time = time.monotonic()
                action, rtc_used, rtc_warnings, rtc_error = self._infer(obs, rtc_payload)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                    "rtc_used": rtc_used,
                }
                if rtc_warnings:
                    action["server_timing"]["rtc_warnings"] = rtc_warnings
                if rtc_error is not None:
                    action["server_timing"]["rtc_error"] = rtc_error
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except RequestError as exc:
                await websocket.send(str(exc))
                continue
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _infer(self, obs: dict, rtc_payload: dict | None) -> tuple[dict, bool, list[str], str | None]:
        rtc_used = False
        rtc_warnings: list[str] = []
        rtc_error: str | None = None
        if rtc_payload is not None and self._rtc_mode != "off":
            try:
                rtc_request, rtc_warnings = _parse_rtc_payload(rtc_payload, self._metadata)
            except RequestError as exc:
                if self._rtc_mode == "only":
                    raise
                rtc_error = str(exc)
                logger.warning("RTC payload rejected, falling back to normal inference: %s", exc)
                rtc_request = None
            if rtc_request is not None:
                try:
                    action = self._policy.infer_rtc(obs, rtc_request)
                    rtc_used = True
                    return action, rtc_used, rtc_warnings, rtc_error
                except NotImplementedError as exc:
                    if self._rtc_mode == "only":
                        raise RequestError(str(exc))
                    rtc_error = str(exc)
                    logger.warning("RTC inference not supported, falling back to normal inference: %s", exc)
                except Exception as exc:
                    if self._rtc_mode == "only":
                        raise RequestError(f"RTC inference failed: {exc}")
                    rtc_error = f"RTC inference failed: {exc}"
                    logger.warning("RTC inference failed, falling back to normal inference: %s", exc)
        return self._policy.infer(obs), rtc_used, rtc_warnings, rtc_error


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None


def _split_payload(payload: object) -> tuple[dict, dict | None]:
    if isinstance(payload, dict) and ("obs" in payload or "rtc" in payload or "type" in payload):
        msg_type = payload.get("type", "infer")
        if msg_type != "infer":
            raise RequestError(f"Unsupported message type: {msg_type}")
        obs = payload.get("obs")
        if obs is None:
            raise RequestError("RTC envelope missing 'obs'.")
        rtc_payload = payload.get("rtc")
        if rtc_payload is not None and not isinstance(rtc_payload, dict):
            raise RequestError("RTC payload must be an object.")
        if not isinstance(obs, dict):
            raise RequestError("Observation must be an object.")
        return obs, rtc_payload
    if not isinstance(payload, dict):
        raise RequestError("Observation must be an object.")
    return payload, None


def _parse_rtc_payload(payload: dict[str, Any], metadata: dict | None) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    if not isinstance(payload, dict):
        raise RequestError("RTC payload must be an object.")

    reset = bool(payload.get("reset", False))

    action_horizon_payload = _coerce_optional_int(payload.get("action_horizon"), "action_horizon", warnings)
    action_dim_payload = _coerce_optional_int(payload.get("action_dim"), "action_dim", warnings)

    action_horizon_meta = None
    action_dim_meta = None
    if metadata is not None:
        action_horizon_meta = _coerce_optional_int(metadata.get("action_horizon"), "action_horizon", warnings)
        action_dim_meta = _coerce_optional_int(metadata.get("action_dim"), "action_dim", warnings)

    if action_horizon_meta is not None and action_horizon_payload is not None:
        if action_horizon_meta != action_horizon_payload:
            raise RequestError(
                f"RTC action_horizon mismatch: payload={action_horizon_payload} metadata={action_horizon_meta}"
            )
    action_horizon = action_horizon_meta if action_horizon_meta is not None else action_horizon_payload
    if action_horizon is None:
        raise RequestError("RTC requires action_horizon from payload or server metadata.")

    prev_actions = payload.get("prev_actions")
    prev_actions_array = None
    if prev_actions is not None:
        try:
            prev_actions_array = np.asarray(prev_actions, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise RequestError(f"RTC prev_actions must be a numeric array: {exc}") from exc
        if prev_actions_array.ndim == 1 and prev_actions_array.size == 0:
            prev_actions_array = None
        elif prev_actions_array.ndim != 2:
            raise RequestError("RTC prev_actions must be a 2D array.")

    action_dim = action_dim_meta if action_dim_meta is not None else action_dim_payload
    if action_dim is None and prev_actions_array is not None:
        action_dim = prev_actions_array.shape[1]
    if action_dim is None:
        raise RequestError("RTC requires action_dim from payload, metadata, or prev_actions.")
    if prev_actions_array is not None and prev_actions_array.shape[1] != action_dim:
        raise RequestError(
            f"RTC prev_actions action_dim mismatch: payload={prev_actions_array.shape[1]} expected={action_dim}"
        )

    d = _coerce_int(payload.get("d"), "d", warnings)
    s = _coerce_int(payload.get("s"), "s", warnings)

    d = max(d, 0)
    if d > action_horizon:
        warnings.append(f"RTC d clamped from {d} to {action_horizon}.")
        d = action_horizon
    if s < d:
        warnings.append(f"RTC s clamped from {s} to {d}.")
        s = d
    max_s = max(action_horizon - d, 0)
    if s > max_s:
        warnings.append(f"RTC s clamped from {s} to {max_s}.")
        s = max_s

    if reset:
        warnings.append("RTC reset requested; ignoring prev_actions.")
        prev_actions_array = None

    if prev_actions_array is not None:
        target_len = max(action_horizon - s, 0)
        if prev_actions_array.shape[0] < target_len:
            pad_len = target_len - prev_actions_array.shape[0]
            if prev_actions_array.shape[0] > 0:
                pad_value = prev_actions_array[-1:, :]
            else:
                pad_value = np.zeros((1, action_dim), dtype=prev_actions_array.dtype)
            pad = np.repeat(pad_value, pad_len, axis=0)
            prev_actions_array = np.concatenate([prev_actions_array, pad], axis=0)
            warnings.append(f"RTC prev_actions padded to {target_len}.")
        elif prev_actions_array.shape[0] > target_len:
            prev_actions_array = prev_actions_array[-target_len:, :]
            warnings.append(f"RTC prev_actions truncated to {target_len}.")

    rtc_request = {
        "prev_actions": prev_actions_array,
        "d": d,
        "s": s,
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "reset": reset,
    }
    return rtc_request, warnings


def _coerce_int(value: object, name: str, warnings: list[str]) -> int:
    if value is None:
        raise RequestError(f"RTC payload missing '{name}'.")
    if isinstance(value, bool):
        raise RequestError(f"RTC field '{name}' must be an integer.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not float(value).is_integer():
            warnings.append(f"RTC field '{name}' rounded down from {value}.")
        return int(value)
    raise RequestError(f"RTC field '{name}' must be an integer.")


def _coerce_optional_int(value: object, name: str, warnings: list[str]) -> int | None:
    if value is None:
        return None
    return _coerce_int(value, name, warnings)
