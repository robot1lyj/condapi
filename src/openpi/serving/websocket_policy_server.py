import asyncio
import http
import logging
import time
import traceback

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
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
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
