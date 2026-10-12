"""Allocated-server-only Cal-QL-style critic initialization experiments.

Pi, the visual encoder and actor are not trained here. The TD backup evaluates
the logged zero-residual behavior policy, with no SAC entropy term. Cal-QL's
Eq. 5.1 is used with a finite local proposal expectation. These adaptations
are recorded in every checkpoint; this is not full upstream PLD reproduction.
"""

# ruff: noqa: E402

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "packages/pld-rl/src"), str(ROOT / "packages/vla-platform/src")]
from vla_platform.metrics import write_metrics
from yam_pld_rl.data import action_coordinates
from yam_pld_rl.data import sha256
from yam_pld_rl.data import write_json
from yam_pld_rl.networks import TwinCritic
from yam_pld_rl.networks import calibrated_penalty


def observation_tensor(a, features, stats, prefix=""):
    state = a[prefix + "state"]
    sm, ss = np.array(stats["state_mean"]), np.array(stats["state_std"])
    am, acs = np.array(stats["action_mean"]), np.array(stats["action_std"])
    pieces = [
        features[a[prefix + "obs_index"]],
        (state - sm) / ss,
        (a[prefix + "previous_state"] - sm) / ss,
        (action_coordinates(a[prefix + "previous_action"], a[prefix + "previous_state"]) - am) / acs,
        (action_coordinates(a[prefix + "base"], state) - am) / acs,
        a[prefix + "aux"],
    ]
    result = np.concatenate(pieces, axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite observation input")
    return torch.from_numpy(result).cuda()


def auc(scores, labels):
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    delta = positive[:, None] - negative[None, :]
    return float(((delta > 0) + 0.5 * (delta == 0)).mean())


@torch.no_grad()
def evaluate(model, target, tensors, a, indices, entry_indices):
    prediction, td, disagreement = [], [], []
    for chunk in np.array_split(indices, max(1, (len(indices) + 511) // 512)):
        ids = torch.as_tensor(chunk, device="cuda")
        q1, q2 = model(tensors["x"][ids], tensors["action"][ids])
        n1, n2 = target(tensors["nx"][ids], tensors["next_action"][ids])
        y = tensors["reward"][ids] + tensors["discount"][ids] * (1 - tensors["terminal"][ids]) * torch.minimum(n1, n2)
        prediction.append(torch.minimum(q1, q2).cpu().numpy())
        td.append(((q1 - y).square() + (q2 - y).square()).cpu().numpy() / 2)
        disagreement.append((q1 - q2).abs().cpu().numpy())
    scores = np.concatenate(prediction)
    valid = a["mc_valid"][indices]
    result = {
        "val_loss": float(np.concatenate(td).mean()),
        "val/q_mean": float(scores.mean()),
        "val/q_abs_max": float(np.abs(scores).max()),
        "val/q_disagreement": float(np.concatenate(disagreement).mean()),
        "val/transitions": len(indices),
        "val/mc_transitions": int(valid.sum()),
    }
    if valid.any():
        error = scores[valid] - a["mc_return"][indices][valid]
        result["val/mc_rmse"] = float(np.sqrt((error**2).mean()))
        result["val/mc_bias"] = float(error.mean())
    entry_indices = np.array(entry_indices, dtype=int)
    if len(entry_indices):
        ids = torch.as_tensor(entry_indices, device="cuda")
        e1, e2 = model(tensors["x"][ids], tensors["action"][ids])
        es = torch.minimum(e1, e2).cpu().numpy()
        labels = a["outcome"][entry_indices]
        result["val/entry_attempts"] = len(es)
        score_auc = auc(es, labels)
        if score_auc is not None:
            result["val/entry_auc"] = score_auc
        emc = a["mc_valid"][entry_indices]
        if emc.any():
            result["val/entry_mc_rmse"] = float(np.sqrt(np.mean((es[emc] - a["mc_return"][entry_indices][emc]) ** 2)))
        for arm, name in [(0, "left"), (1, "right")]:
            mask = a["arm"][entry_indices] == arm
            result[f"val/{name}/attempts"] = int(mask.sum())
            for outcome, label in [(1, "held"), (0, "empty")]:
                m = mask & (labels == outcome)
                result[f"val/{name}/{label}_n"] = int(m.sum())
                if m.any():
                    result[f"val/{name}/{label}_q"] = float(es[m].mean())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Training requires a Slurm GPU allocation; workstation training is forbidden")
    cfg = json.loads(args.config.read_text())
    required = {
        "id",
        "seed",
        "success_only",
        "calql_coefficient",
        "steps",
        "batch_size",
        "learning_rate",
        "width",
        "tau",
        "proposal_samples",
        "proposal_radius_normalized",
        "gradient_clip",
        "log_interval",
        "eval_interval",
        "save_interval",
    }
    if not required <= set(cfg) or set(cfg) - required - {"mc_coefficient"}:
        raise ValueError("unexpected or missing recipe fields")
    if args.resume is None:
        args.output.mkdir(exist_ok=False)
    elif not args.output.is_dir():
        raise ValueError("resume requires the existing output directory")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(cfg["seed"])
    np_rng = np.random.default_rng(cfg["seed"])
    manifest = json.loads((args.replay / "manifest.json").read_text())
    fm = json.loads((args.replay / "feature-manifest.json").read_text())
    if sha256(args.replay / "replay.npz") != manifest["replay_sha256"]:
        raise ValueError("replay changed")
    if sha256(args.replay / "normalization.json") != manifest["normalization_sha256"]:
        raise ValueError("normalization changed")
    if sha256(args.replay / "features.npy") != fm["features_sha256"]:
        raise ValueError("feature cache changed")
    with np.load(args.replay / "replay.npz") as source:
        a = {k: source[k] for k in source.files}
    stats = json.loads((args.replay / "normalization.json").read_text())
    features = np.load(args.replay / "features.npy", mmap_mode="r")
    tensors = {"x": observation_tensor(a, features, stats), "nx": observation_tensor(a, features, stats, "next_")}
    am, ast = np.array(stats["action_mean"]), np.array(stats["action_std"])
    axes = np.arange(7)[None, :] + a["arm"][:, None] * 7
    for name, data in [("action", a["action_coordinates"]), ("next_action", a["next_action_coordinates"])]:
        normalized = (data - am) / ast
        tensors[name] = torch.from_numpy(np.take_along_axis(normalized, axes, 1).astype(np.float32)).cuda()
    for name in ["reward", "discount", "terminal", "mc_return", "mc_valid"]:
        tensors[name] = torch.from_numpy(a[name].astype(np.float32)).cuda()
    train_mask = (a["episode"] != 1) & ((a["outcome"] == 1) if cfg["success_only"] else True)
    train = np.flatnonzero(train_mask)
    val = np.flatnonzero(a["episode"] == 1)
    if not len(train) or not len(val):
        raise ValueError("missing train/validation data")
    groups = [np.flatnonzero(train_mask & (a["attempt"] == aid)) for aid in np.unique(a["attempt"][train])]
    attempts = json.loads((args.replay / "attempts.json").read_text())
    entries = [x["transition_start"] for x in attempts if x["episode"] == 1 and x["entry_precedes_closure"]]
    model = TwinCritic(tensors["x"].shape[-1], cfg["width"]).cuda()
    target = copy.deepcopy(model).requires_grad_(requires_grad=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    code_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    if code_dirty:
        raise RuntimeError("server training checkout must be clean and pinned")
    identity = {
        "config_sha256": sha256(args.config),
        "replay_sha256": manifest["replay_sha256"],
        "features_sha256": fm["features_sha256"],
        "code_commit": code_commit,
    }
    record = {
        "config": cfg,
        **identity,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "hostname": os.uname().nodename,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "train_transitions": len(train),
        "train_attempts": len(groups),
        "validation_attempts": sum(x["episode"] == 1 for x in attempts),
        "sampling": "uniform attempt, then uniform transition in that attempt",
        "encoder": fm,
        "normalization": stats,
        "actor_updated": False,
        "deployable": False,
        "method": "logged behavior TD; optional Cal-QL Eq5.1 local proposal expectation and project MC-return regression anchor",
        "mc_anchor_is_project_extension": bool(cfg.get("mc_coefficient", 0)),
        "proposal_units": "training action std units; NOT approved physical deployment bounds",
        "phase": "critic-only offline initialization",
        "precision": "FP32, TF32 disabled",
    }
    start_step, best = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cuda", weights_only=True)
        if checkpoint["identity"] != identity:
            raise ValueError("resume data/config/code identity changed")
        model.load_state_dict(checkpoint["critic"])
        target.load_state_dict(checkpoint["target"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu())
        np_rng.bit_generator.state = checkpoint["numpy_rng"]
        start_step, best = checkpoint["step"], checkpoint["best_entry_rmse"]
    write_json(args.output / "run.json", record)
    metrics_file = args.output / "metrics.jsonl"

    def save(name, step):
        payload = {
            "identity": identity,
            "step": step,
            "critic": model.state_dict(),
            "target": target.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "numpy_rng": np_rng.bit_generator.state,
            "best_entry_rmse": best,
            "actor": "zero residual; not trained",
            "deployable": False,
            "observation_dim": tensors["x"].shape[-1],
            "width": cfg["width"],
            "normalization": stats,
        }
        temporary = args.output / (name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(args.output / name)

    @torch.no_grad()
    def save_predictions(step):
        # Per-attempt evidence is retained alongside aggregates for later analysis.
        rows = np.array([item["transition_start"] for item in attempts], dtype=int)
        ids = torch.as_tensor(rows, device="cuda")
        q1, q2 = model(tensors["x"][ids], tensors["action"][ids])
        np.savez_compressed(
            args.output / f"attempt-predictions-{step:06d}.npz",
            candidate_id=np.array([item["candidate_id"] for item in attempts]),
            episode=a["episode"][rows],
            arm=a["arm"][rows],
            outcome=a["outcome"][rows],
            row=a["row"][rows],
            mc_return=a["mc_return"][rows],
            mc_valid=a["mc_valid"][rows],
            entry_precedes_closure=np.array([item["entry_precedes_closure"] for item in attempts]),
            q1=q1.cpu().numpy(),
            q2=q2.cpu().numpy(),
        )

    start = previous_log = time.monotonic()
    accumulated = []
    write_metrics(metrics_file, start_step, evaluate(model, target, tensors, a, val, entries))
    try:
        for step in range(start_step + 1, cfg["steps"] + 1):
            # Sampling never depends on validation losses or future labels as model inputs.
            selected = np_rng.integers(len(groups), size=cfg["batch_size"])
            ids_np = np.array([groups[g][np_rng.integers(len(groups[g]))] for g in selected])
            ids = torch.as_tensor(ids_np, device="cuda")
            x, action = tensors["x"][ids], tensors["action"][ids]
            with torch.no_grad():
                n1, n2 = target(tensors["nx"][ids], tensors["next_action"][ids])
                y = tensors["reward"][ids] + tensors["discount"][ids] * (1 - tensors["terminal"][ids]) * torch.minimum(
                    n1, n2
                )
            q1, q2 = model(x, action)
            td_loss = ((q1 - y).square().mean() + (q2 - y).square().mean()) / 2
            known = tensors["mc_valid"][ids].bool()
            mc_loss = torch.zeros((), device="cuda")
            if cfg.get("mc_coefficient", 0) and known.any():
                mc = tensors["mc_return"][ids][known]
                mc_loss = ((q1[known] - mc).square().mean() + (q2[known] - mc).square().mean()) / 2
            penalty = torch.zeros((), device="cuda")
            proposal_mean = torch.zeros((), device="cuda")
            if cfg["calql_coefficient"]:
                k = cfg["proposal_samples"]
                proposals = action[:, None, :] + cfg["proposal_radius_normalized"] * (
                    2 * torch.rand((len(ids), k, 7), device="cuda") - 1
                )
                xp = x[:, None, :].expand(-1, k, -1).reshape(-1, x.shape[-1])
                s1, s2 = model(xp, proposals.reshape(-1, 7))
                s1, s2 = s1.reshape(-1, k), s2.reshape(-1, k)
                if known.any():
                    mc = tensors["mc_return"][ids][known]
                    penalty = (
                        calibrated_penalty(s1[known], q1[known], mc).mean()
                        + calibrated_penalty(s2[known], q2[known], mc).mean()
                    ) / 2
                proposal_mean = (s1.mean() + s2.mean()) / 2
            loss = td_loss + cfg["calql_coefficient"] * penalty + cfg.get("mc_coefficient", 0) * mc_loss
            if not torch.isfinite(loss) or max(q1.abs().max().item(), q2.abs().max().item()) > 100:
                raise FloatingPointError("nonfinite loss or |Q|>100")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            if not torch.isfinite(gradient):
                raise FloatingPointError("nonfinite critic gradient")
            optimizer.step()
            with torch.no_grad():
                for p_target, p_current in zip(target.parameters(), model.parameters(), strict=True):
                    p_target.lerp_(p_current, cfg["tau"])
            accumulated.append(
                [
                    loss.item(),
                    td_loss.item(),
                    penalty.item(),
                    mc_loss.item(),
                    q1.mean().item(),
                    q2.mean().item(),
                    proposal_mean.item(),
                    gradient.item(),
                    tensors["mc_valid"][ids].mean().item(),
                ]
            )
            if step % cfg["log_interval"] == 0:
                now = time.monotonic()
                average = np.array(accumulated).mean(0)
                log = dict(
                    zip(
                        [
                            "loss",
                            "train/td_loss",
                            "train/calql_penalty",
                            "train/mc_loss",
                            "q/q1",
                            "q/q2",
                            "q/proposal_mean",
                            "grad_norm",
                            "train/mc_valid_fraction",
                        ],
                        average.tolist(),
                        strict=True,
                    )
                )
                log.update(
                    step_seconds=(now - previous_log) / len(accumulated),
                    samples_per_second=len(accumulated) * cfg["batch_size"] / (now - previous_log),
                    gpu_peak_memory_gib=torch.cuda.max_memory_allocated() / 1024**3,
                    learning_rate=cfg["learning_rate"],
                    elapsed_seconds=now - start,
                    actor_updates=0,
                    online_environment_steps=0,
                )
                if not cfg["calql_coefficient"]:
                    log.pop("q/proposal_mean")
                write_metrics(metrics_file, step, log)
                print(json.dumps({"step": step, **log}), flush=True)
                accumulated.clear()
                previous_log = now
            if step % cfg["eval_interval"] == 0:
                validation = evaluate(model, target, tensors, a, val, entries)
                write_metrics(metrics_file, step, validation)
                value = validation.get("val/entry_mc_rmse", validation.get("val/mc_rmse", float("inf")))
                if value < best:
                    best = value
                    save("best.pt", step)
                write_json(args.output / "latest-evaluation.json", {"step": step, **validation})
                save_predictions(step)
            if step % cfg["save_interval"] == 0 or step == cfg["steps"]:
                save(f"step-{step:06d}.pt", step)
        write_json(
            args.output / "status.json",
            {
                "state": "completed",
                "step": cfg["steps"],
                "seconds": time.monotonic() - start,
                "actor_trained": False,
                "deployable": False,
            },
        )
    except BaseException as exc:
        write_json(
            args.output / "status.json",
            {"state": "failed", "error": repr(exc), "seconds": time.monotonic() - start, "deployable": False},
        )
        raise


if __name__ == "__main__":
    main()
