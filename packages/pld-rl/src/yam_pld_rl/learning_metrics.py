"""Fixed recorded-return diagnostics; these are not off-policy success estimates."""

import numpy as np
import torch

from yam_pld_rl.calibrated import feasible


def auc(score, outcome):
    positive, negative = score[outcome == 1], score[outcome == 0]
    if not len(positive) or not len(negative):
        return None
    return float((positive[:, None] > negative).mean() + .5*(positive[:, None] == negative).mean())


@torch.no_grad()
def evaluate(learner, tensors, arrays):
    # Evaluation must never consume training randomness, including TD target noise.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(18791)
        q1, q2 = learner.critic(tensors["x"], tensors["action"])
        q = torch.minimum(q1, q2).cpu().numpy()
        target = learner.backup(tensors)
        bellman = ((q1-target).square()+(q2-target).square()).cpu().numpy()/2
        u, _ = learner.actor(tensors["x"], deterministic=True)
        a = feasible(u, tensors["scale"], tensors["negative_scale"]).cpu().numpy()
        piq = torch.minimum(*learner.critic(tensors["x"], torch.as_tensor(a, device=u.device))).cpu().numpy()
    train = arrays["train"].astype(bool)
    lo, hi = np.quantile(arrays["action"][train], [.01, .99], axis=0)
    mc, outcome = arrays["mc_return"], arrays["attempt_outcome"]
    result, flat = {}, {}
    for split, base in (("train", train), ("val", ~train)):
        result[split] = {}
        for stage, sm in (("all", np.ones(len(train), bool)), ("entry", arrays["is_entry"].astype(bool)),
                          ("last_decision", arrays["terminal"] == 1)):
            for arm, am in (("both", np.ones(len(train), bool)), ("left", arrays["arm"] == 0),
                            ("right", arrays["arm"] == 1)):
                mask = base & sm & am
                if not mask.any():
                    continue
                s = {"count": int(mask.sum()), "mc_rmse": float(np.sqrt(np.mean((q[mask]-mc[mask])**2))),
                     "auc": auc(q[mask], outcome[mask]), "q_mean": float(q[mask].mean()),
                     "q_min": float(q[mask].min()), "q_max": float(q[mask].max()),
                     "recorded_return_mean": float(mc[mask].mean()), "bellman_mse": float(bellman[mask].mean()),
                     "actor_rms": float(np.sqrt(np.mean(a[mask]**2))),
                     "recorded_action_rms": float(np.sqrt(np.mean(arrays["action"][mask]**2))),
                     "outside_train_marginal_1_99": float(((a[mask]<lo)|(a[mask]>hi)).mean()),
                     "self_estimated_policy_q_gain": float(np.mean(piq[mask]-q[mask]))}
                result[split][stage+"/"+arm] = s
                for metric in ("mc_rmse", "auc", "actor_rms", "outside_train_marginal_1_99"):
                    if s[metric] is not None:
                        flat[f"{split}/{stage}_{arm}_{metric}"] = s[metric]
        flat["val_loss" if split == "val" else "train/bellman_mse"] = float(bellman[base].mean())
    return flat, result
