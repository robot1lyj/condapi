"""Run one real OpenArm request against a websocket policy server."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import time
from typing import Any

import numpy as np
from openpi_client import websocket_client_policy

DEFAULT_PROMPT = "Fold the T-shirt properly, Advantage: positive"


def _observation(prompt: str) -> dict[str, Any]:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    return {
        "observation.state": np.zeros(16, dtype=np.float32),
        "observation.images.base": image,
        "observation.images.left_wrist": image.copy(),
        "observation.images.right_wrist": image.copy(),
        "prompt": prompt,
    }


def _action_summary(value: Any) -> dict[str, Any]:
    actions = np.asarray(value, dtype=np.float32)
    if actions.shape != (50, 16):
        raise ValueError(f"OpenArm server must return actions with shape (50, 16), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("OpenArm server returned non-finite actions")
    return {
        "shape": list(actions.shape),
        "dtype": str(actions.dtype),
        "min": float(actions.min()),
        "max": float(actions.max()),
        "mean": float(actions.mean()),
        "right_gripper_first": float(actions[0, 7]),
        "left_gripper_first": float(actions[0, 15]),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json_atomic(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _timeout(_signum: int, _frame: Any) -> None:
    raise TimeoutError("Timed out waiting for OpenArm websocket inference")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    previous_handler = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(args.timeout_seconds)
    try:
        client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        result = client.infer(_observation(args.prompt))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

    if "actions" not in result:
        raise KeyError(f"OpenArm server response has no actions key: {sorted(result)}")
    payload = {
        "schema_version": "openarm_policy_server_smoke_v1",
        "passed": True,
        "host": args.host,
        "port": args.port,
        "checkpoint": args.checkpoint,
        "prompt": args.prompt,
        "elapsed_ms": 1000.0 * (time.perf_counter() - started),
        "actions": _action_summary(result["actions"]),
        "server_metadata": _jsonable(client.get_server_metadata()),
    }
    _write_json_atomic(args.output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="gpu25")
    parser.add_argument("--port", type=int, default=6666)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
