"""Smoke trained-RTC WebSocket with real recorded YAM observations and prefixes."""

import argparse
import datetime
import json
from pathlib import Path
import time

from benchmark_pi05 import read_observation
import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect


def checked_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("RTC case path escaped fixture directory")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--required-delays", default="0,1,10", help="Comma-separated RTC delays covered by this smoke")
    parser.add_argument("--expected-prefix-norm", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new smoke receipt path")
    required_delays = {int(value) for value in args.required_delays.split(",")}
    case_set = json.loads(args.cases.read_text())
    if case_set.get("source_kind") != "real_yam_recording":
        raise ValueError("RTC smoke requires real recorded YAM cases")

    packer = msgpack_numpy.Packer()
    started = time.monotonic()
    rows = []
    with connect(args.url, compression=None, max_size=None) as ws:
        metadata = msgpack_numpy.unpackb(ws.recv())
        if (
            metadata.get("rtc_mode") != "trained"
            or metadata.get("backend") != "tensorrt_cuda_graph"
            or metadata.get("checkpoint_weights_sha256") != args.expected_weights_sha256
            or metadata.get("norm_stats_sha256") != case_set["norm_stats_sha256"]
            or metadata.get("rtc_max_delay_steps") != 10
            or metadata.get("action_horizon") != 50
            or metadata.get("action_dim") != 14
            or metadata.get("action_0_relative_to_observation_policy_tick") != 0
            or not np.isclose(metadata.get("action_dt_s", -1), 1 / 30)
            or (args.expected_prefix_norm is not None and metadata.get("rtc_prefix_norm") != args.expected_prefix_norm)
        ):
            raise RuntimeError("Unexpected trained-RTC handshake metadata")
        handshake_ms = (time.monotonic() - started) * 1000
        for case in case_set["cases"]:
            observation = read_observation(checked_path(args.cases.parent, case["sample"]))
            committed = np.load(checked_path(args.cases.parent, case["committed_actions"]), allow_pickle=False)
            delay = case["delay_steps"]
            rtc = {
                "delay_steps": delay,
                "observation_policy_tick": case["observation_policy_tick"],
                "target_start_tick": case["target_start_tick"],
                "committed_start_tick": case["committed_start_tick"],
                "committed_actions": committed,
            }
            request = {"type": "infer", "obs": observation, "rtc": rtc}
            start = time.monotonic()
            ws.send(packer.pack(request))
            response = ws.recv()
            roundtrip_ms = (time.monotonic() - start) * 1000
            if isinstance(response, str):
                raise RuntimeError(f"RTC request rejected: {response}")
            result = msgpack_numpy.unpackb(response)
            actions = np.asarray(result["actions"])
            prefix_exact = bool(np.array_equal(actions[:delay], committed.astype(np.float32)))
            if (
                actions.shape != (50, 14)
                or not np.isfinite(actions).all()
                or not prefix_exact
                or result["server_timing"].get("rtc_used") is not True
            ):
                raise RuntimeError(f"Invalid RTC output for {case['sample']}")
            joint_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
            if delay:
                new_joint_steps = np.diff(actions[delay - 1 :, joint_indices], axis=0)
            else:
                state = np.asarray(observation["observation.state"])
                new_joint_steps = np.diff(
                    np.vstack((state[joint_indices], actions[:, joint_indices])), axis=0
                )
            max_new_joint_step_rad = float(np.max(np.abs(new_joint_steps)))
            guard = metadata.get("rtc_joint_step_guard_rad_per_tick")
            if guard is not None and max_new_joint_step_rad > guard + 1e-6:
                raise RuntimeError(f"RTC output exceeded advertised joint-step guard: {case['sample']}")
            rows.append({
                "sample": case["sample"],
                "delay_steps": delay,
                "output_shape": list(actions.shape),
                "prefix_exact": prefix_exact,
                "output_finite": True,
                "max_new_joint_step_rad": max_new_joint_step_rad,
                "server_infer_ms": result["server_timing"]["infer_ms"],
                "roundtrip_ms": roundtrip_ms,
            })
    if required_delays - {row["delay_steps"] for row in rows}:
        raise RuntimeError(f"RTC smoke did not cover delays {sorted(required_delays)}")
    receipt = {
        "status": "passed",
        "observed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "url": args.url,
        "handshake_ms": handshake_ms,
        "metadata": metadata,
        "cases": rows,
        "server_infer_p50_ms": float(np.percentile([row["server_infer_ms"] for row in rows], 50)),
        "server_infer_p95_ms": float(np.percentile([row["server_infer_ms"] for row in rows], 95)),
        "roundtrip_p50_ms": float(np.percentile([row["roundtrip_ms"] for row in rows], 50)),
        "roundtrip_p95_ms": float(np.percentile([row["roundtrip_ms"] for row in rows], 95)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
