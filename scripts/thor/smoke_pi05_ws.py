"""Send one recorded three-camera YAM observation over the ordinary WebSocket protocol."""

import argparse
import datetime
import json
from pathlib import Path
import time

from benchmark_pi05 import read_observation
import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new smoke result path")
    observation = read_observation(args.sample)
    packer = msgpack_numpy.Packer()
    started = time.monotonic()
    with connect(args.url, compression=None, max_size=None) as ws:
        metadata = msgpack_numpy.unpackb(ws.recv())
        start_infer = time.monotonic()
        ws.send(packer.pack(observation))
        response = ws.recv()
        roundtrip_ms = (time.monotonic() - start_infer) * 1000
    if isinstance(response, str):
        raise RuntimeError(response)
    result = msgpack_numpy.unpackb(response)
    actions = np.asarray(result["actions"])
    if actions.shape != (50, 14) or not np.isfinite(actions).all():
        raise RuntimeError("Invalid returned actions")
    if (
        metadata["checkpoint_step"] != 100000
        or metadata["backend"] != "tensorrt"
        or metadata["robot_action_dim"] != 14
        or metadata["action_horizon"] != 50
    ):
        raise RuntimeError("Unexpected handshake metadata")
    receipt = {
        "observed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "url": args.url,
        "status": "passed",
        "observation_keys": list(observation),
        "input_state_shape": list(observation["observation.state"].shape),
        "input_images": {
            key: list(value.shape) for key, value in observation.items() if key.startswith("observation.images.")
        },
        "output_shape": list(actions.shape),
        "output_all_finite": True,
        "server_infer_ms": result["server_timing"]["infer_ms"],
        "roundtrip_ms": roundtrip_ms,
        "connect_and_handshake_ms": (start_infer - started) * 1000,
        "metadata": metadata,
    }
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
