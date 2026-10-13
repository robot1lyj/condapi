"""CPU-only probe replay/calibration for an immutable first collect contract."""
# ruff: noqa: E402
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/pld-rl/src"))
from yam_pld_rl.block_contract import feasible_bounds
from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.block_contract import physical_baseline
from yam_pld_rl.block_contract import validate_contract
from yam_pld_rl.curator import native_replies
from yam_pld_rl.curator import physical_candidates
from yam_pld_rl.data import sha256
from yam_pld_rl.data import write_json
from yam_pld_rl.intervention import DEFAULT_GATE
from yam_pld_rl.intervention import LEGACY_GATE
from yam_pld_rl.intervention import InterventionSession
from yam_pld_rl.kinematics import MODEL_SHA256
from yam_pld_rl.kinematics import YamKinematics
from yam_pld_rl.native_recording import read_episode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--legacy-candidates", type=Path, required=True)
    p.add_argument("--normalization", type=Path, required=True)
    p.add_argument("--base-weights-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, events, hashes = read_episode(args.episode)
    native, _ = native_replies(rows, events)
    traces = sorted([item["reply"]["pld"] for item in native.values()
                     if item["reply"].get("discarded") is False and not item["reply"].get("error")],
                    key=lambda item: item["start_tick"])
    old = json.loads(args.legacy_candidates.read_text())
    manifest = json.loads((args.episode / "manifest.json").read_text())
    fk = YamKinematics(ROOT / "third_party/yam-fk-model/yam_linear4310.xml")
    c = json.loads((ROOT / "configs/pld_rl/sac-contract-v2.template.json").read_text())
    c.update(gate=DEFAULT_GATE.copy(), base_weights_sha256=args.base_weights_sha256,
             action_mapping="signed_feasible_v1", reward_contract="drained_grasp_v1",
             baseline_transform="sdk_gripper_endpoints_v1", fk_model_sha256=MODEL_SHA256,
             lower=manifest["action_limits"]["lower"], upper=manifest["action_limits"]["upper"],
             radius=[.02]*6+[.05]+[.02]*6+[.05], max_residual_step=[.008]*6+[.02]+[.008]*6+[.02],
             limits_evidence={"manifest_sha256": sha256(args.episode/"manifest.json"),
                              "sdk_max_target_step": manifest["action_limits"]["max_target_step"],
                              "thor_max_target_step_is_separate_guard": True},
             grasp_z_range_m={a: [min(x["reference_base_z_m"] for x in old if x["arm"] == a)-.02,
                                   max(x["reference_base_z_m"] for x in old if x["arm"] == a)+.02]
                              for a in ("left", "right")},
             grasp_scope_evidence={"source_candidates_sha256": sha256(args.legacy_candidates),
                                   "margin_m": .02, "kind": "current_scene_FK_height_calibration_not_semantic_detector"},
             status="first two collect episodes; bounded exploration, not a trained expert or collision guarantee")
    validate_contract(c)
    legacy = InterventionSession({**c, "gate": LEGACY_GATE})
    gate = InterventionSession(c)
    windows, rejected, mismatch = {}, [], 0
    for trace in traces:
        rtc = {"observation_policy_tick": trace["start_tick"], "delay_steps": trace["delay_steps"]}
        original = legacy.select(trace, rtc, trace["state"], trace["base"], trace["base_z_m"])
        mismatch += int(original["active"] != trace["gate"]["active"]
                        or original.get("arm") != trace["gate"].get("arm"))
        base = physical_baseline(trace["base"], trace["delay_steps"], c)
        result = gate.select(trace, rtc, trace["state"], base, fk.gate_heights(trace["state"], base, c))
        if result["active"]:
            windows.setdefault(result["window_id"], {"arm": result["arm"], "start": result["start_tick"],
                                                     "end": rows[-1]["tick"]+1, "reason": None})
            try:
                feasible_bounds(base, trace["previous_command"], result["arm"], trace["delay_steps"], c)
                gate.note_proposal(trace["start_tick"], trace["delay_steps"])
            except ValueError as exc:
                rejected.append({"tick": trace["start_tick"], "reason": str(exc)})
        ended = result["ended"]
        if ended and ended["window_id"] in windows:
            windows[ended["window_id"]].update(end=trace["start_tick"], reason=ended["reason"])
    candidates = physical_candidates(rows, fk)
    for window in windows.values():
        window["relative_candidates"] = [x["candidate_id"] for x in candidates
                                         if x["arm"] == window["arm"] and window["start"] <= x["close_tick"] < window["end"]]
    covered = [x["candidate_id"] for x in old if any(w["arm"] == x["arm"]
               and w["start"] <= x["close_tick"] < w["end"] for w in windows.values())]
    vertices = np.asarray(list(product((-1., 1.), repeat=6)))
    displacement = []
    for item in old:
        offset = 0 if item["arm"] == "left" else 7
        state = np.asarray(rows[item["close_row"]]["measured_state"])[offset:offset+6]
        center = fk.position(state)
        displacement += [float(np.linalg.norm(fk.position(state+.02*v)-center)) for v in vertices]
    report = {"schema": "pld_collect_calibration_v1", "source_hashes": hashes,
              "legacy_candidates_sha256": sha256(args.legacy_candidates), "legacy_replay_mismatches": mismatch,
              "legacy_candidate_coverage": len(covered), "legacy_candidates": len(old),
              "relative_candidates": len(candidates), "relative_candidate_arms": dict(Counter(x["arm"] for x in candidates)),
              "qualified_relative_candidates": sum(x["evidence_eligible"] for x in candidates),
              "online_windows": len(windows), "online_window_arms": dict(Counter(w["arm"] for w in windows.values())),
              "multiple_relative_closures": [w for w in windows.values() if len(w["relative_candidates"]) > 1],
              "boundary_rejections": rejected, "sampled_fk_corner_max_m": max(displacement),
              "contract_sha256": fingerprint(c), "script_sha256": sha256(__file__),
              "training": False, "robot_execution": False, "success_rate": None}
    write_json(args.output/"contract.json", c)
    write_json(args.output/"normalization.json", json.loads(args.normalization.read_text()))
    write_json(args.output/"calibration.json", report)
    write_json(args.output/"windows.json", list(windows.values()))
    write_json(args.output/"relative-candidates.json", candidates)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if mismatch or rejected or len(covered) != len(old):
        raise SystemExit("calibration failed; retained output is not a ready release")


if __name__ == "__main__":
    main()
