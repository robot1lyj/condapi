"""Inference-only portable bundle check; no optimizer, Pi service or robot I/O."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src"))
from yam_pld_rl.data import write_json
from yam_pld_rl.runtime import BlockRuntime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    if manifest["contract"]["base_weights_sha256"] != "fixture-only":
        raise ValueError("this check uses synthetic images/actions, only fixture bundles permitted")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = BlockRuntime(args.bundle, args.encoder, base_weights_sha256="fixture-only")
    rng = np.random.default_rng(1431)
    obs = {"prompt": manifest["contract"]["task"], "observation.state": np.full(14, .123456789)}
    obs.update({f"observation.images.{view}_rgb": rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
                for view in ("top", "left", "right")})
    base = np.full((50, 14), .12345678901234, dtype=np.float64)
    rtc = {"delay_steps": 3, "observation_policy_tick": 0}
    request = {**runtime.metadata, "mode": "eval", "run_id": "fixture", "episode_id": "0", "attempt_id": "0",
               "decision_id": "0", "arm": "left", "elapsed_seconds": 0, "previous_command": base[0]}
    zero = runtime(obs, rtc, request, {"actions": base})
    if not manifest["provenance"].get("fixture"):
        raise ValueError("zero-head bootstrap fixture required")
    assert zero["actions"].tobytes() == base.tobytes()
    timings = []
    for arm in ("left", "right"):
        request.update(arm=arm, mode="collect")
        for _ in range(3):
            start = time.monotonic()
            result = runtime(obs, rtc, request, {"actions": base})
            timings.append((time.monotonic() - start) * 1000)
            assert result["actions"][:3].tobytes() == base[:3].tobytes()
            other = slice(7, 14) if arm == "left" else slice(0, 7)
            assert result["actions"][:, other].tobytes() == base[:, other].tobytes()
            assert np.max(np.abs(result["actions"] - base)) <= .010000001
            assert np.any(result["actions"] != base)
    report = {"status": "passed_inference_fixture_only", "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "bootstrap_snapshot": runtime.snapshot, "samples": 6,
              "postprocessor_ms_p50": float(np.median(timings)), "postprocessor_ms_max": max(timings),
              "Pi_latency_included": False, "cross_IPC_tested": False, "robot_connected": False,
              "checks": ["official ResNet18 forward", "zero-head exact Pi", "both active arms",
                         "FP64 prefix and inactive arm exact", "bounded nonzero stochastic output"]}
    write_json(args.output, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
