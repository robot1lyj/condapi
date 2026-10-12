"""Thor-local real-recording protocol smoke for the probe-only PLD container.

No actuator connection, RL exploration, reward judgement or training.
"""

import argparse
import datetime
import json
from pathlib import Path
import time

from benchmark_pi05 import read_observation
import numpy as np
from openpi_client import msgpack_numpy
from smoke_pi05_rtc_ws import checked_path
from websockets.sync.client import connect


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--expected-weights-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    suite = json.loads(args.cases.read_text())
    if suite.get("source_kind") != "real_yam_recording":
        raise ValueError("real recorded observations required")
    packer = msgpack_numpy.Packer()
    rows, rejects = [], []
    with connect(args.url, compression=None, max_size=None) as ws:
        metadata = msgpack_numpy.unpackb(ws.recv())
        identity = metadata["pld"]
        if (metadata["inference_profile"] != "pld-rl" or identity["expert"] != "PLD-Grasp"
                or identity["modes"] != ["off","probe"] or identity["actor_loaded"] or identity["replay_contract_ready"]
                or metadata["checkpoint_weights_sha256"] != args.expected_weights_sha256
                or metadata["rtc_prefix_norm"] != "quantile" or metadata["norm_stats_sha256"] != suite["norm_stats_sha256"]):
            raise ValueError("unexpected deployment identity/capability")
        for mode in ("collect","eval","shadow"):
            ws.send(packer.pack({"type":"infer","obs":{},"rtc":{},"pld":{"mode":mode}}))
            reply = ws.recv()
            if not isinstance(reply,str) or "probe-only" not in reply:
                raise ValueError("unsupported mode did not reject explicitly")
            rejects.append({"mode":mode,"response":reply})
        for number, case in enumerate(suite["cases"],1):
            obs = read_observation(checked_path(args.cases.parent,case["sample"]))
            prefix = np.load(checked_path(args.cases.parent,case["committed_actions"]),allow_pickle=False)
            d, tick = case["delay_steps"],case["observation_policy_tick"]
            rtc = {"delay_steps":d,"observation_policy_tick":tick,"target_start_tick":case["target_start_tick"],
                   "committed_start_tick":case["committed_start_tick"],"committed_actions":prefix}
            # This is an offline protocol replay, not recorded robot execution.
            anchor = prefix[0] if d else np.asarray(obs["observation.state"])
            request = {**{k:identity[k] for k in ("schema","contract_sha256","snapshot_sha256")},
                       "mode":"probe","run_id":"network-smoke","episode_id":case["sample"],"epoch":number,
                       "request_id":number,"observation_id":number,"previous_command":anchor}
            start = time.monotonic()
            ws.send(packer.pack({"type":"infer","obs":obs,"rtc":rtc,"pld":request}))
            reply = ws.recv()
            elapsed = (time.monotonic()-start)*1000
            if isinstance(reply,str):
                raise ValueError(reply)
            output = msgpack_numpy.unpackb(reply)
            actions, trace = np.asarray(output["actions"]),output["pld"]
            if (actions.shape!=(50,14) or not np.isfinite(actions).all() or trace["replay_eligible"]
                    or trace["residual_applied"] or actions.tobytes()!=np.asarray(trace["base"]).tobytes()
                    or not np.array_equal(actions[:d],prefix.astype(actions.dtype)) or "gate" not in trace):
                raise ValueError("probe changed Pi output/prefix or claimed training eligibility")
            rows.append({"sample":case["sample"],"delay":d,"finite":True,"whole_chunk_equals_pi":True,
                         "prefix_exact":True,"gate_active":trace["gate"]["active"],"server_ms":output["server_timing"]["infer_ms"],
                         "roundtrip_ms":elapsed})
        if {r["delay"] for r in rows} != {0,1,10}:
            raise ValueError("incomplete RTC delay coverage")
    report = {"status":"passed","observed_at":datetime.datetime.now(datetime.UTC).isoformat(),"url":args.url,
              "metadata":metadata,"cases":rows,"rejections":rejects,"robot_connected":False,"cross_IPC_tested":False,
              "previous_command_kind":"offline_fixture_anchor_not_actual","training":False,
              "roundtrip_p50_ms":float(np.median([r["roundtrip_ms"] for r in rows])),
              "roundtrip_p95_ms":float(np.percentile([r["roundtrip_ms"] for r in rows],95))}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":"passed","cases":len(rows),"rejected_modes":len(rejects),
                      "roundtrip_p50_ms":report["roundtrip_p50_ms"],"roundtrip_p95_ms":report["roundtrip_p95_ms"]}))


if __name__ == "__main__":
    main()
