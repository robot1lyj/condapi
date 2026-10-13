"""Compare full parent/new postprocessors on recorded RGB and Pi outputs, no robot."""
# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"packages/pld-rl/src"), str(ROOT/"scripts/thor")]
from benchmark_pi05 import read_observation
from yam_pld_rl.block_contract import decode
from yam_pld_rl.block_contract import map_action
from yam_pld_rl.collector import validate_collector_trace
from yam_pld_rl.runtime import BlockRuntime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--parent-wire", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.bundle/"manifest.json").read_text())
    old, new = [BlockRuntime(path, args.encoder, device=args.device,
                            base_weights_sha256=manifest["contract"]["base_weights_sha256"])
                for path in (args.parent, args.bundle)]
    if old.snapshot != manifest["collector"]["parent_snapshot_sha256"]:
        raise ValueError("wrong parent")
    cases = json.loads(args.cases.read_text())["cases"]
    wires = json.loads(args.parent_wire.read_text())
    arms, checked, errors = set(), 0, []
    for case in cases:
        source = next(r for r in wires if r["request"]["episode_id"] == case["sample"])
        obs = read_observation(args.cases.parent/case["sample"])
        rtc = {"delay_steps": len(np.load(args.cases.parent/case["prefix"], allow_pickle=False)),
               "observation_policy_tick": case["tick"]}
        base = np.asarray(source["response"]["pld"]["raw_base"])
        results = []
        for runtime in (old, new):
            req = {**source["request"], **runtime.metadata, "mode": "eval"}
            results.append(runtime.new_session()(obs, rtc, req, {"actions": base}))
        np.testing.assert_array_equal(results[0]["actions"], results[1]["actions"])
        np.testing.assert_array_equal(results[0]["pld"]["x"], results[1]["pld"]["x"])
        req = {**source["request"], **new.metadata, "mode": "collect"}
        for seed in range(3):
            torch.manual_seed(809+seed)
            result = new.new_session()(obs, rtc, req, {"actions": base})
            trace = result["pld"]
            validate_collector_trace(trace, manifest)
            np.testing.assert_array_equal(result["actions"][:rtc["delay_steps"]], base[:rtc["delay_steps"]])
            action = map_action(np.tanh(trace["collector_latent_center"]), trace["scale"], trace["negative_scale"])
            center = decode(trace["physical_base"], action, trace["arm"], rtc["delay_steps"], new.contract)
            error = float(np.max(np.abs(center-results[0]["pld"]["physical_actions"])))
            if error > 1e-7 or not np.isfinite(trace["log_prob_u"]):
                raise ValueError("physical center/density changed")
            errors.append(error)
            arms.add(trace["arm"])
            checked += 1
    if arms != {"left", "right"}:
        raise ValueError("both arms required")
    report = {"status": "passed", "device": args.device, "snapshot_sha256": new.snapshot,
              "parent_snapshot_sha256": old.snapshot, "recorded_cases": len(cases),
              "sampled_collect_cases": checked, "eval_wire_and_observation_exact": True,
              "max_zero_noise_center_error": max(errors), "collector_trace_valid": True,
              "prefix_exact": True, "full_encoder_and_actor_used": True,
              "training": False, "robot_connected": False, "live_Pi_tested": False}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
