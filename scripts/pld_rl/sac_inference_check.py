"""Inference-only portable bundle check; no optimizer, Pi service or robot I/O."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src"))
from block_fixtures import inference_case
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
    obs, rtc, request, output = inference_case(runtime, mode="eval")
    zero = runtime.new_session()(obs, rtc, request, output)
    if not manifest["provenance"].get("fixture"):
        raise ValueError("zero-head bootstrap fixture required")
    assert zero["actions"].tobytes() == output["actions"].tobytes()
    timings = []
    for arm in ("left", "right"):

        for _ in range(3):
            obs, rtc, request, output = inference_case(runtime, arm=arm)
            base = output["actions"]
            session = runtime.new_session()
            start = time.monotonic()
            result = session(obs, rtc, request, output)
            assert result["pld"]["arm"] == arm
            assert "attempt_id" not in result["pld"]
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
              "checks": ["official ResNet18 forward", "zero-head exact Pi", "Thor causal selection of both arms without client arm fields",
                         "FP64 prefix and inactive arm exact", "bounded nonzero stochastic output"]}
    write_json(args.output, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
