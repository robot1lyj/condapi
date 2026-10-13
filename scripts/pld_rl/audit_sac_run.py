"""Audit saved SAC checkpoints on Thor/Slurm, without training or robot I/O."""

# ruff: noqa: E402

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--code", type=Path, required=True)
p.add_argument("--run", type=Path, required=True)
p.add_argument("--replay", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
sys.path.insert(0, str(args.code / "packages/pld-rl/src"))
from yam_pld_rl.data import sha256
from yam_pld_rl.sac import SacLearner

if args.output.exists():
    raise ValueError("write a new audit, do not replace evidence")
run = json.loads((args.run / "run.json").read_text())
if not torch.cuda.is_available():
    raise RuntimeError("checkpoint audit requires the authorized GPU runtime")
torch.set_num_threads(2)
with np.load(args.replay / "replay.npz", allow_pickle=False) as archive:
    arrays = {k: archive[k] for k in archive.files}
manifest = json.loads((args.replay / "manifest.json").read_text())
assert sha256(args.replay / "replay.npz") == manifest["replay_sha256"]
contract = run["policy_contract"]["contract"]
assert contract == manifest["contract"]
learner = SacLearner(arrays["x"].shape[1], run["policy_contract"]["config"])
x = torch.as_tensor(arrays["x"], device="cuda", dtype=torch.float32)
a = torch.as_tensor(arrays["action"], device="cuda", dtype=torch.float32)
positive, negative = (torch.as_tensor(arrays[k], device="cuda", dtype=torch.float32) for k in ("scale", "negative_scale"))
reports = []
for path in sorted(args.run.glob("checkpoint-*.pt")):
    ckpt = torch.load(path, map_location="cuda", weights_only=True)
    learner.load_state_dict(ckpt["learner"])
    bundle = args.run / ("bundle-" + path.stem.split("-")[-1])
    actor = torch.load(bundle / "actor.pt", map_location="cuda", weights_only=True)
    assert all(torch.equal(v, learner.actor.state_dict()[k]) for k, v in actor.items())
    assert all(torch.isfinite(v).all() for model in (learner.actor, learner.critic, learner.target) for v in model.parameters())
    with torch.no_grad():
        u, _ = learner.actor(x, deterministic=True)
        correction = u * torch.where(u < 0, negative, positive)
        q1, q2 = learner.critic(x, a)
    assert torch.isfinite(u).all()
    assert torch.isfinite(q1).all()
    assert torch.isfinite(q2).all()
    radius = np.asarray(contract["radius"]).reshape(2, 7)[arrays["arm"].astype(int)]
    physical = correction.cpu().numpy() * radius
    result = {"step": ckpt["step"], "actor_updates": learner.actor_updates, "checkpoint_sha256": sha256(path),
              "bundle_actor_sha256": sha256(bundle / "actor.pt"), "bundle_equals_checkpoint": True,
              "finite_parameters_and_outputs": True, "deterministic_u_abs_max": float(u.abs().max()),
              "deterministic_joint_offset_abs_max_rad": float(np.abs(physical[:, :6]).max()),
              "deterministic_gripper_offset_abs_max": float(np.abs(physical[:, 6]).max()), "splits": {}}
    q = torch.minimum(q1, q2).cpu().numpy()
    predicted = correction.cpu().numpy()
    train_terminal = arrays["train"].astype(bool) & arrays["terminal"].astype(bool)
    prior = float(arrays["reward"][train_terminal].mean())
    for split, flag in (("train", True), ("val", False)):
        selected = arrays["train"].astype(bool) == flag
        terminal = selected & arrays["terminal"].astype(bool)
        labels = arrays["reward"][terminal]
        scores = q[terminal]
        pos_scores, neg_scores = scores[labels == 1], scores[labels == 0]
        auc = float(((pos_scores[:, None] > neg_scores[None, :]) + .5 *
                     (pos_scores[:, None] == neg_scores[None, :])).mean()) if len(pos_scores) and len(neg_scores) else None
        control = selected & ((arrays["scale"] + arrays["negative_scale"]).sum(1) > 0)
        ranges = np.quantile(arrays["action"][control], [.01, .99], axis=0)
        outside = (predicted[control] < ranges[0]) | (predicted[control] > ranges[1])
        result["splits"][split] = {"transitions": int(selected.sum()), "attempts": int(terminal.sum()),
            "reward1": int(labels.sum()), "reward0": int((labels == 0).sum()),
            "terminal_q_rmse": float(np.sqrt(np.mean((q[terminal] - labels) ** 2))),
            "constant_0_5_terminal_rmse": .5,
            "train_mean_constant_terminal_rmse": float(np.sqrt(np.mean((prior - labels) ** 2))),
            "terminal_reward_auc": auc,
            "behavior_offset_rms": float(np.sqrt(np.mean(arrays["action"][control] ** 2))),
            "deterministic_actor_offset_rms": float(np.sqrt(np.mean(predicted[control] ** 2))),
            "outside_split_marginal_p01_p99_fraction": float(outside.mean()),
            "terminal_predictions": [{"attempt": str(attempt), "reward": float(label), "q": float(score),
                                      "arm": "right" if arm else "left"}
                                     for attempt, label, score, arm in zip(arrays["attempt"][terminal], labels, scores,
                                                                          arrays["arm"][terminal], strict=True)],
            "arms": {("right" if arm else "left"): {"attempts": int((terminal & (arrays["arm"] == arm)).sum()),
                               "reward1": int(arrays["reward"][terminal & (arrays["arm"] == arm)].sum())}
                     for arm in np.unique(arrays["arm"])}}
    reports.append(result)
metrics = [json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines()]
assert all(np.isfinite(value) for row in metrics for value in row["metrics"].values())
summary = {"checkpoints": reports, "metrics_rows": len(metrics), "metrics_finite": True,
           "training_code_commit": run["code_commit"], "audit_source_sha256": sha256(Path(__file__)),
           "detection_accuracy": None, "robot_success_rate": None, "production_deployed": False,
           "interpretation": "checkpoint consistency and dataset diagnostics only; Q is not evidence of robot improvement"}
args.output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"checkpoints": len(reports), "final": {k: v for k, v in reports[-1].items() if k != "splits"}}, ensure_ascii=False))
