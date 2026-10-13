"""Check a learned bundle against recorded contexts, without robot/network I/O."""
# ruff: noqa: PLC0415
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-code", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.source_code/"packages/pld-rl/src"))
    from yam_pld_rl.block_contract import decode
    from yam_pld_rl.block_contract import map_action
    from yam_pld_rl.block_contract import wire_plan
    from yam_pld_rl.data import sha256
    from yam_pld_rl.execution import require_training_runtime
    from yam_pld_rl.runtime import BlockRuntime

    require_training_runtime("thor-offline", cuda_available=torch.cuda.is_available(),
                             capability=torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((args.bundle/"manifest.json").read_text())
    runtime = BlockRuntime(args.bundle, args.encoder, base_weights_sha256=manifest["contract"]["base_weights_sha256"])
    with np.load(args.dataset/"learning-replay-r02-v1/replay.npz", allow_pickle=False) as archive:
        z = {k: archive[k] for k in ("x", "episode", "decision", "scale", "negative_scale")}
    wanted = { (str(ep), str(dec)): i for i, (ep, dec) in enumerate(zip(z["episode"], z["decision"], strict=True)) }
    with torch.inference_mode():
        means = runtime.actor(torch.as_tensor(z["x"], device="cuda"), deterministic=True)[0].cpu().numpy()
    count, arms, joint, gripper, displacement = 0, {"left": 0, "right": 0}, 0., 0., []
    seen = set()
    with (args.dataset/"reward-release-v2/records.jsonl").open() as stream:
        for line in stream:
            proposal = json.loads(line)["proposal"]
            key = (proposal["run_id"]+"/"+proposal["episode_id"], proposal["decision_id"])
            if key not in wanted:
                continue
            if key in seen:
                raise ValueError("duplicate recorded decision")
            seen.add(key)
            i = wanted[key]
            np.testing.assert_array_equal(z["x"][i], np.asarray(proposal["x"], np.float32))
            action = map_action(means[i], z["scale"][i], z["negative_scale"][i])
            base = np.asarray(proposal["physical_base"])
            arm, delay = proposal["arm"], proposal["delay_steps"]
            result = decode(base, action, arm, delay, runtime.contract)
            wire = wire_plan(result, proposal["raw_base"], delay)
            np.testing.assert_array_equal(wire[:delay], np.asarray(proposal["raw_base"])[:delay])
            start = 0 if arm == "left" else 7
            other = slice(7, 14) if arm == "left" else slice(0, 7)
            np.testing.assert_array_equal(result[:, other], base[:, other])
            offset = np.abs(result[:, start:start+7]-base[:, start:start+7])
            joint, gripper = max(joint, float(offset[:, :6].max())), max(gripper, float(offset[:, 6].max()))
            positions = runtime.fk.positions(np.vstack((base[:, start:start+6], result[:, start:start+6])))
            displacement.append(float(np.linalg.norm(positions[50:]-positions[:50], axis=1).max()))
            count += 1
            arms[arm] += 1
    if len(seen) != len(wanted):
        raise ValueError("not all recorded contexts were checked")
    result = {"status": "passed_recorded_contexts_only", "snapshot_sha256": runtime.snapshot,
              "script_sha256": sha256(Path(__file__)), "contexts": count, "arms": arms,
              "full_runtime_bundle_loaded": True, "prefix_and_inactive_arm_exact": True,
              "max_joint_offset_rad": joint, "max_gripper_offset": gripper,
              "target_FK_displacement_m_quantiles": np.quantile(displacement, [0, .5, .95, 1]).tolist(),
              "closed_loop_robot_tested": False, "live_Pi_inference_or_latency_tested": False,
              "encoder_used_for_cached_features_recomputed": False}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
