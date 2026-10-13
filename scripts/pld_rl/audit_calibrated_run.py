"""Read-only Thor checkpoint/export audit; no optimizer or robot commands."""
# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"packages/pld-rl/src"), str(ROOT/"packages/vla-platform/src")]
from train_sac import learner_for
from yam_pld_rl.calibrated import feasible
from yam_pld_rl.data import sha256
from yam_pld_rl.execution import require_training_runtime
from yam_pld_rl.learning_metrics import evaluate
from yam_pld_rl.sac import SquashedActor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require_training_runtime("thor-offline", cuda_available=torch.cuda.is_available(),
                             capability=torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(args.replay/"replay.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    keys = ("x", "nx", "action", "scale", "negative_scale", "next_scale", "next_negative_scale",
            "reward", "terminal", "discount", "mc_return", "attempt_outcome")
    tensors = {k: torch.as_tensor(arrays[k], device="cuda", dtype=torch.float32) for k in keys}
    results = []
    for path in sorted(args.run.glob("checkpoint-*.pt")):
        state = torch.load(path, weights_only=True, map_location="cuda")
        cfg, step = state["policy_contract"]["config"], state["step"]
        learner = learner_for(arrays["x"].shape[1], cfg)
        learner.load_state_dict(state["learner"])
        assert all(torch.isfinite(v).all() for m in (learner.actor, learner.critic, learner.target)
                   for v in m.state_dict().values())
        bundle = args.run/f"bundle-{step:06d}"
        manifest = json.loads((bundle/"manifest.json").read_text())
        assert manifest["provenance"]["checkpoint_sha256"] == sha256(path)
        assert manifest["actor_sha256"] == sha256(bundle/"actor.pt")
        actor = SquashedActor(arrays["x"].shape[1], cfg).cuda()
        actor.load_state_dict(torch.load(bundle/"actor.pt", weights_only=True, map_location="cuda"))
        for k, v in learner.actor.state_dict().items():
            torch.testing.assert_close(v, actor.state_dict()[k], rtol=0, atol=0)
        with torch.no_grad():
            u, _ = actor(tensors["x"], deterministic=True)
            torch.testing.assert_close(u, learner.actor(tensors["x"], deterministic=True)[0], rtol=0, atol=0)
            a = feasible(u, tensors["scale"], tensors["negative_scale"])
            assert torch.all(a <= tensors["scale"]+1e-6)
            assert torch.all(a >= -tensors["negative_scale"]-1e-6)
            radius = np.asarray(manifest["contract"]["radius"]).reshape(2, 7)[arrays["arm"].astype(int)]
            physical = a.cpu().numpy()*radius
        flat, details = evaluate(learner, tensors, arrays)
        saved = json.loads((args.run/f"evaluation-{step:06d}.json").read_text())["metrics"]
        for split in saved:
            for stage in saved[split]:
                for metric, value in saved[split][stage].items():
                    actual = details[split][stage][metric]
                    if value is not None:
                        assert np.isclose(value, actual, rtol=1e-5, atol=1e-6), (step, split, stage, metric)
        results.append({"step": step, "checkpoint_sha256": sha256(path), "actor_sha256": manifest["actor_sha256"],
                        "actor_updates": learner.actor_updates, "parameters_finite": True, "export_exact": True,
                        "recorded_state_feasibility": True, "saved_evaluation_reproduced": True,
                        "max_joint_offset_rad": float(np.abs(physical[:, :6]).max()),
                        "max_gripper_offset": float(np.abs(physical[:, 6]).max()), "metrics": flat})
    args.output.write_text(json.dumps({"run": str(args.run), "checkpoints": results,
                           "new_state_or_robot_validation": False}, indent=2)+"\n")
    print(json.dumps({"run": args.run.name, "checkpoints_checked": len(results), "status": "passed"}))


if __name__ == "__main__":
    main()
