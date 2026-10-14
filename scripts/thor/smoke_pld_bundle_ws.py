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
    p.add_argument("--expected-actor-phase", choices=("bootstrap_exploration", "candidate_requires_AB"),
                   default="bootstrap_exploration")
    p.add_argument("--expected-snapshot-sha256")
    p.add_argument("--bundle-manifest", type=Path)
    args = p.parse_args()
    import sys  # noqa: PLC0415
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"packages/pld-rl/src"))
    from yam_pld_rl.block_contract import decode  # noqa: PLC0415
    from yam_pld_rl.block_contract import fingerprint  # noqa: PLC0415
    from yam_pld_rl.block_contract import physical_baseline  # noqa: PLC0415
    from yam_pld_rl.block_contract import wire_plan  # noqa: PLC0415
    from yam_pld_rl.collector import execution_contract  # noqa: PLC0415
    from yam_pld_rl.collector import validate_collector_trace  # noqa: PLC0415

    if args.output.exists():
        raise FileExistsError(args.output)
    suite = json.loads(args.cases.read_text())
    contract = json.loads((args.cases.parent/"contract.json").read_text())
    manifest = json.loads(args.bundle_manifest.read_text()) if args.bundle_manifest else {"contract": contract}
    if suite["source_kind"] != "real_yam_recording":
        raise ValueError("real recorded inputs required")
    packer, rows, active_arms, client_samples = msgpack_numpy.Packer(), [], set(), []
    learned_eval_nonzero = 0
    with connect(args.url, compression=None, max_size=None) as ws:
        metadata = msgpack_numpy.unpackb(ws.recv())
        identity = metadata["pld"]
        if identity["contract_sha256"] != fingerprint(manifest["contract"]):
            raise ValueError("bundle execution contract identity differs")
        if not identity["actor_loaded"] or identity["actor_phase"] != args.expected_actor_phase:
            raise ValueError("actor phase differs from the requested test")
        if args.expected_actor_phase == "candidate_requires_AB" and not args.expected_snapshot_sha256:
            raise ValueError("learned actor test requires a pinned snapshot")
        if args.expected_snapshot_sha256 and identity["snapshot_sha256"] != args.expected_snapshot_sha256:
            raise ValueError("actor snapshot differs")
        epoch = 0
        for repeat in range(3):
            for mode in ("probe", "shadow", "eval", "collect"):
                contract = execution_contract(manifest, mode)
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
                    if trace.get("execution_contract_sha256", fingerprint(contract)) != fingerprint(contract):
                        raise ValueError("mode-specific execution contract differs")
                    raw, base = np.asarray(trace["raw_base"]), np.asarray(trace["physical_base"])
                    if not np.array_equal(base, physical_baseline(raw, d, contract)):
                        raise ValueError("wrong SDK physical baseline")
                    expected = base
                    if trace["gate"]["active"]:
                        if mode in {"collect", "eval"} and (mode == "collect" or contract.get("trajectory")):
                            validate_collector_trace(trace, manifest)
                        if mode in {"collect", "eval"}:
                            expected = decode(base, trace["action"], trace["arm"], d, contract, context=trace.get("trajectory_context"))
                        elif contract.get("trajectory") and trace.get("replay_eligible") is not False:
                            raise ValueError("native trajectory diagnostic must not be compiled as correction replay")
                        active_arms.add(trace["arm"])
                        zero_expected = mode in {"probe", "shadow"} or (
                            mode == "eval" and args.expected_actor_phase == "bootstrap_exploration")
                        if zero_expected and np.any(trace["action"]):
                            raise ValueError("bootstrap zero mode changed physical commands")
                        if mode == "eval" and np.any(trace["action"]):
                            learned_eval_nonzero += 1
                        if mode == "collect" and not np.any(trace["action"]) and not trace.get("trajectory_context", {}).get("reason"):
                            raise ValueError("collect has no stochastic residual")
                    wire = wire_plan(expected, raw, d)
                    if (not np.array_equal(result["actions"], wire) or not np.array_equal(wire[:d], prefix)
                            or not np.isfinite(wire).all() or not np.array_equal(trace["actions"], wire)
                            or (mode in {"probe", "shadow"} and not np.array_equal(wire, trace["base"]))):
                        raise ValueError("wire output or native prefix changed")
                    if repeat == 0:
                        client_samples.append({"request": request, "response": result})
                    rows.append({"mode": mode, "sample": case["sample"], "repeat": repeat,
                                 "active": trace["gate"]["active"], "arm": trace.get("arm"),
                                 "infer_ms": result["server_timing"]["infer_ms"], "roundtrip_ms": elapsed,
                                 "prefix_exact": True, "physical_decoder_exact": True})
    timed = [r["infer_ms"] for r in rows if r["repeat"] > 0 and r["active"]]
    if active_arms != {"left", "right"} or not timed:
        raise ValueError("real-input suite did not exercise both arms")
    if args.expected_actor_phase == "candidate_requires_AB" and learned_eval_nonzero == 0:
        raise ValueError("learned eval actor never applied a correction")
    p95 = float(np.percentile(timed, 95))
    status = "passed" if p95 < args.max_p95_ms else "failed_latency"
    report = {"status": status, "metadata": metadata, "cases": rows, "active_arms": sorted(active_arms),
              "active_server_p50_ms": float(np.median(timed)), "active_server_p95_ms": p95,
              "max_p95_ms": args.max_p95_ms, "robot_connected": False, "cross_IPC_tested": False,
              "training": False, "actor_phase_tested": args.expected_actor_phase,
              "learned_eval_nonzero_cases": learned_eval_nonzero, "robot_improvement_verified": False}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    args.output.with_suffix(".wire.json").write_text(json.dumps(client_samples,
        default=lambda value: value.tolist() if hasattr(value, "tolist") else value)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "active_arms", "active_server_p50_ms", "active_server_p95_ms")}))
    if status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
