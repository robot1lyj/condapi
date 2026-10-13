"""Recorded gate equivalence and CPU FK/codec timings; no neural/robot work."""
# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import sys
import time

import msgpack
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "packages/pld-rl/src"), str(ROOT / "packages/openpi-client/src")]
from openpi_client import msgpack_numpy
from yam_pld_rl.block_contract import window_context
from yam_pld_rl.intervention import InterventionSession
from yam_pld_rl.kinematics import YamKinematics


class FixedSession(InterventionSession):
    def reset(self):
        super().reset()
        self.session_id = "offline-equivalence"


def scalar_heights(fk, state, base, contract):
    result = fk.heights(state)
    if contract.get("grasp_z_range_m"):
        result["planned_z_m"] = {arm: [float(fk.position(row[off:off+6])[2]) for row in base]
                                 for arm, off in (("left", 0), ("right", 7))}
    return result


def equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal(x, y)
    elif isinstance(a, float):
        np.testing.assert_allclose(a, b, atol=2e-15, rtol=0)
    else:
        assert a == b, (a, b)


def quantiles(v):
    return dict(zip(("p50_ms", "p95_ms", "max_ms"), map(float, np.percentile(v, [50, 95, 100])), strict=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    suite = json.loads(args.cases.read_text())
    fk = YamKinematics(ROOT / "third_party/yam-fk-model/yam_linear4310.xml")
    old, new = FixedSession(suite["contract"]), FixedSession(suite["contract"])
    times = [[], []]
    max_difference, contexts = 0.0, 0
    for case in suite["cases"]:
        state, base = np.asarray(case["state"]), np.asarray(case["base"])
        results = []
        for index, func in enumerate((scalar_heights, lambda f, s, b, c: f.gate_heights(s, b, c))):
            start = time.perf_counter()
            results.append(func(fk, state, base, suite["contract"]))
            times[index].append((time.perf_counter()-start)*1000)
        for arm in ("left", "right"):
            av = [results[0][arm], *results[0]["planned_z_m"][arm]]
            bv = [results[1][arm], *results[1]["planned_z_m"][arm]]
            max_difference = max(max_difference, float(np.max(np.abs(np.asarray(av)-bv))))
        gates = [s.select(case["request"], case["rtc"], state, base, h)
                 for s, h in zip((old, new), results, strict=True)]
        equal(*gates)
        for session in (old, new):
            session.note_proposal(case["rtc"]["observation_policy_tick"], case["rtc"]["delay_steps"])
        if gates[0]["active"]:
            contexts += 1
            a, b = [window_context(g, case["rtc"]["observation_policy_tick"], h[g["arm"]], suite["contract"]["action_dt_s"])
                    for g, h in zip(gates, results, strict=True)]
            np.testing.assert_array_equal(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32))
    # Raw serialization doesn't depend on scene content; keep full RGB shape.
    payload = {f"observation.images.{v}_rgb": np.zeros((480, 640, 3), np.uint8)
               for v in ("top", "left", "right")}
    payload["observation.state"] = np.zeros(14, np.float64)
    def old_default(obj):
        if isinstance(obj, np.ndarray):
            return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
        return msgpack_numpy.pack_array(obj)
    packers = [msgpack.Packer(default=old_default), msgpack_numpy.Packer()]
    encoded = [packer.pack(payload) for packer in packers]
    assert encoded[0] == encoded[1]
    codec = [[], []]
    for index in range(100):
        # Alternate order to avoid giving one path all warm-cache positions.
        for variant in (index % 2, 1-index % 2):
            start = time.perf_counter()
            packers[variant].pack(payload)
            codec[variant].append((time.perf_counter()-start)*1000)
    report = {"status": "passed", "recorded_requests": len(suite["cases"]), "active_float32_contexts_exact": contexts,
              "max_fk_difference_m": max_difference, "gate_decisions_equal": True,
              "scalar_fk": quantiles(times[0]), "batch_fk": quantiles(times[1]),
              "codec_bytes": len(encoded[0]), "codec_exact_wire_bytes": True,
              "codec_original": quantiles(codec[0]), "codec_buffer": quantiles(codec[1]),
              "neural_forward": False, "robot_io": False, "training": False,
              "cross_IPC_latency_verified": False}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
