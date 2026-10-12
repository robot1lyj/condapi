"""Real recorded-input PLD bundle checks; never connects to an actuator."""
import argparse
import json
from pathlib import Path
import time

from benchmark_pi05 import read_observation
import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-p95-ms", type=float, default=266.0)
    args = p.parse_args()
    import sys  # noqa: PLC0415
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"packages/pld-rl/src"))
    from yam_pld_rl.block_contract import decode  # noqa: PLC0415
    from yam_pld_rl.block_contract import physical_baseline  # noqa: PLC0415
    from yam_pld_rl.block_contract import wire_plan  # noqa: PLC0415

    if args.output.exists():
        raise FileExistsError(args.output)
    suite = json.loads(args.cases.read_text())
    contract = json.loads((args.cases.parent/"contract.json").read_text())
    if suite["source_kind"] != "real_yam_recording":
        raise ValueError("real recorded inputs required")
    packer, rows, active_arms = msgpack_numpy.Packer(), [], set()
    with connect(args.url, compression=None, max_size=None) as ws:
        metadata = msgpack_numpy.unpackb(ws.recv())
        identity = metadata["pld"]
        if not identity["actor_loaded"] or identity["actor_phase"] != "bootstrap_exploration":
            raise ValueError("zero-mean bootstrap bundle required")
        epoch = 0
        for repeat in range(3):
            for mode in ("probe", "shadow", "eval", "collect"):
                for case in suite["cases"]:
                    epoch += 1
                    obs = read_observation(args.cases.parent/case["sample"])
                    prefix = np.load(args.cases.parent/case["prefix"], allow_pickle=False)
                    d, tick = len(prefix), case["tick"]
                    rtc = {"delay_steps": d, "observation_policy_tick": tick, "target_start_tick": tick,
                           "committed_start_tick": tick, "committed_actions": prefix}
                    request = {**{k: identity[k] for k in ("schema", "snapshot_sha256", "contract_sha256")},
                               "mode": mode, "run_id": "recorded-bundle-smoke", "episode_id": case["sample"],
                               "epoch": epoch, "request_id": epoch, "observation_id": epoch,
                               "previous_command": case["previous_command"]}
                    started = time.monotonic()
                    ws.send(packer.pack({"type": "infer", "obs": obs, "rtc": rtc, "pld": request}))
                    reply = ws.recv()
                    elapsed = (time.monotonic()-started)*1000
                    if isinstance(reply, str):
                        raise ValueError(reply)
                    result = msgpack_numpy.unpackb(reply)
                    trace = result["pld"]
                    raw, base = np.asarray(trace["raw_base"]), np.asarray(trace["base"])
                    if not np.array_equal(base, physical_baseline(raw, d, contract)):
                        raise ValueError("wrong SDK physical baseline")
                    expected = base
                    if trace["gate"]["active"]:
                        expected = decode(base, trace["action"], trace["arm"], d, contract)
                        active_arms.add(trace["arm"])
                        if mode != "collect" and np.any(trace["action"]):
                            raise ValueError("bootstrap zero mode changed physical commands")
                        if mode == "collect" and not np.any(trace["action"]):
                            raise ValueError("collect has no stochastic residual")
                    wire = wire_plan(expected, raw, d)
                    if (not np.array_equal(result["actions"], wire) or not np.array_equal(wire[:d], prefix)
                            or not np.isfinite(wire).all()):
                        raise ValueError("wire output or native prefix changed")
                    rows.append({"mode": mode, "sample": case["sample"], "repeat": repeat,
                                 "active": trace["gate"]["active"], "arm": trace.get("arm"),
                                 "infer_ms": result["server_timing"]["infer_ms"], "roundtrip_ms": elapsed,
                                 "prefix_exact": True, "physical_decoder_exact": True})
    timed = [r["infer_ms"] for r in rows if r["repeat"] > 0 and r["active"]]
    if active_arms != {"left", "right"} or not timed:
        raise ValueError("real-input suite did not exercise both arms")
    p95 = float(np.percentile(timed, 95))
    status = "passed" if p95 < args.max_p95_ms else "failed_latency"
    report = {"status": status, "metadata": metadata, "cases": rows, "active_arms": sorted(active_arms),
              "active_server_p50_ms": float(np.median(timed)), "active_server_p95_ms": p95,
              "max_p95_ms": args.max_p95_ms, "robot_connected": False, "cross_IPC_tested": False,
              "training": False, "zero_mean_actor_is_not_an_improved_expert": True}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "active_arms", "active_server_p50_ms", "active_server_p95_ms")}))
    if status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
