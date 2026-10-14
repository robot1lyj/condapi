"""Full encoder/actor and continuous decoder on recorded inputs, no actuator I/O."""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "packages/pld-rl/src"), str(ROOT / "scripts/thor")]
from benchmark_pi05 import read_observation
from yam_pld_rl.block_contract import decode
from yam_pld_rl.collector import execution_contract
from yam_pld_rl.collector import validate_collector_trace
from yam_pld_rl.runtime import BlockRuntime
from yam_pld_rl.sac import SquashedActor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--parent-wire", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    m = json.loads((args.bundle / "manifest.json").read_text())
    parent = json.loads((args.parent / "manifest.json").read_text())
    assert m["actor_sha256"] == parent["actor_sha256"]
    runtime = BlockRuntime(
        args.bundle, args.encoder, base_weights_sha256=m["contract"]["base_weights_sha256"], device=args.device
    )
    actor = SquashedActor(parent["observation_dim"], parent["config"]).to(args.device).eval()
    actor.load_state_dict(torch.load(args.parent / "actor.pt", map_location=args.device, weights_only=True))
    cases = json.loads(args.cases.read_text())["cases"]
    wires = json.loads(args.parent_wire.read_text())
    rows = []
    arms = set()
    packets = []
    native = m.get("collector", {}).get("native_actor_coordinates") is True
    diagnostics = bool(m.get("actor_contract", {}).get("trajectory"))
    reference = BlockRuntime(args.parent, args.encoder, base_weights_sha256=parent["contract"]["base_weights_sha256"],
                             device=args.device) if m["collector"].get("eval_preserves_parent_physical_policy") else None
    for mode in (("collect", "eval", "probe", "shadow") if diagnostics else ("collect", "eval")):
        for case in cases:
            original = next(r for r in wires if r["request"]["episode_id"] == case["sample"])
            obs = read_observation(args.cases.parent / case["sample"])
            base = np.asarray(original["response"]["pld"]["raw_base"])
            d = len(np.load(args.cases.parent / case["prefix"], allow_pickle=False))
            rtc = {"delay_steps": d, "observation_policy_tick": case["tick"]}
            request = {**original["request"], **runtime.metadata, "mode": mode}
            torch.manual_seed(809)
            result = runtime.new_session()(obs, rtc, request, {"actions": base})
            if mode == "eval" and reference is not None:
                before = reference.new_session()(obs, rtc, {**request, **reference.metadata}, {"actions": base})
                np.testing.assert_array_equal(result["actions"], before["actions"])
            trace = result["pld"]
            nonzero = False
            if mode in {"probe", "shadow"}:
                np.testing.assert_array_equal(result["actions"], trace["base"])
                assert trace["residual_applied"] is False
                assert trace["replay_eligible"] is False
            elif trace["gate"]["active"]:
                validate_collector_trace(trace, m)
                x = np.r_[np.asarray(trace["x"])[:2264], trace["actor_context"]].astype(np.float32)
                with torch.inference_mode():
                    mean = actor(torch.from_numpy(x)[None].to(args.device), deterministic=True)[0][0].cpu().numpy()
                np.testing.assert_allclose(mean, trace["actor_mean_u"], atol=1e-7, rtol=0)
                expected = decode(
                    trace["physical_base"],
                    trace["action"],
                    trace["arm"],
                    d,
                    execution_contract(m, mode),
                    context=trace["trajectory_context"],
                )
                np.testing.assert_array_equal(result["actions"][d:], expected[d:])
                off = 0 if trace["arm"] == "left" else 7
                np.testing.assert_array_equal(
                    result["actions"][d:, 7 - off : 14 - off],
                    np.asarray(trace["physical_base"])[d:, 7 - off : 14 - off],
                )
                nonzero = bool(np.any(expected != trace["physical_base"]))
                if nonzero:
                    arms.add(trace["arm"])
            np.testing.assert_array_equal(result["actions"][:d], base[:d])
            rows.append(
                {
                    "mode": mode,
                    "sample": case["sample"],
                    "active": trace["gate"]["active"],
                    "nonzero": nonzero,
                    "context_reason": trace.get("trajectory_context", {}).get("reason"),
                }
            )
            packets.append({"request": request, "response": result})
    result = {
        "status": "passed" if arms == {"left", "right"} else "insufficient_recorded_coverage",
        "device": args.device,
        "snapshot_sha256": runtime.snapshot,
        "actor_weights_equal_reference": True,
        "comparison": "saved native actor vs runtime" if native else "parent actor preserved in expanded collector",
        "parent_actor_same_input_mean_exact": True,
        "prefix_and_inactive_arm_exact": True,
        "new_decoder_and_trace_verified": True,
        "arms": sorted(arms),
        "cases": rows,
        "training": False,
        "robot_connected": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.output.with_suffix(".wire.json").write_text(
        json.dumps(packets, default=lambda v: v.tolist() if hasattr(v, "tolist") else v) + "\n"
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
