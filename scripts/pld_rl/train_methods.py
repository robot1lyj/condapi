"""Allocated-server-only offline actor/critic comparison, never a deployment entry."""

# ruff: noqa: E402

import argparse
from collections import defaultdict
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
from train_offline import auc
from train_offline import observation_tensor
from vla_platform.metrics import write_metrics
from yam_pld_rl.data import action_coordinates
from yam_pld_rl.data import sha256
from yam_pld_rl.data import write_json
from yam_pld_rl.offline_methods import OfflineLearner


def load_data(root, cfg):
    manifest = json.loads((root / "manifest.json").read_text())
    fm = json.loads((root / "feature-manifest.json").read_text())
    for file, expected in [
        ("replay.npz", manifest["replay_sha256"]),
        ("normalization.json", manifest["normalization_sha256"]),
        ("features.npy", fm["features_sha256"]),
    ]:
        if sha256(root / file) != expected:
            raise ValueError(f"immutable input changed: {file}")
    with np.load(root / "replay.npz") as data:
        a = {key: data[key] for key in data.files}
    stats = json.loads((root / "normalization.json").read_text())
    features = np.load(root / "features.npy", mmap_mode="r")
    tensors = {"x": observation_tensor(a, features, stats), "nx": observation_tensor(a, features, stats, "next_")}
    mean, std = np.array(stats["action_mean"]), np.array(stats["action_std"])
    axes = np.arange(7)[None, :] + a["arm"][:, None] * 7

    def active(value):
        return torch.from_numpy(np.take_along_axis((value - mean) / std, axes, 1).astype(np.float32)).cuda()

    for prefix in ["", "next_"]:
        tensors[prefix + "action"] = active(a[prefix + "action_coordinates"])
        tensors[prefix + "base"] = active(action_coordinates(a[prefix + "base"], a[prefix + "state"]))
        tensors[prefix + "u"] = (tensors[prefix + "action"] - tensors[prefix + "base"]) / cfg["residual_radius_std"]
        if tensors[prefix + "u"].abs().max().item() > 1.00001:
            raise ValueError("dataset residual outside configured action support; never silently clip data")
    for name in ["reward", "discount", "terminal"]:
        tensors[name] = torch.from_numpy(a[name].astype(np.float32)).cuda()
    return a, tensors, stats, manifest, fm


@torch.no_grad()
def evaluate(learner, tensors, a, indices, entries):
    scores, td, disagreement, residual, policy_gain, policy_q = [], [], [], [], [], []
    for rows in np.array_split(indices, max(1, (len(indices) + 511) // 512)):
        ids = torch.as_tensor(rows, device="cuda")
        b = {key: value[ids] for key, value in tensors.items()}
        q1, q2 = learner.critic(b["x"], b["action"])
        prediction = torch.minimum(q1, q2)
        y = learner.backup(b, noise=False)
        u = learner.actor(b["x"])
        pq1, pq2 = learner.critic(b["x"], learner.action(b["x"], b["base"]))
        pq = torch.minimum(pq1, pq2)
        scores.append(prediction.cpu().numpy())
        td.append(((q1 - y).square() + (q2 - y).square()).cpu().numpy() / 2)
        disagreement.append((q1 - q2).abs().cpu().numpy())
        residual.append(u.cpu().numpy())
        policy_gain.append((pq - prediction).cpu().numpy())
        policy_q.append(pq.cpu().numpy())
    scores, u = np.concatenate(scores), np.concatenate(residual)
    result = {
        "val_loss": float(np.concatenate(td).mean()),
        "val/q_mean": float(scores.mean()),
        "val/q_abs_max": float(np.abs(scores).max()),
        "val/q_disagreement": float(np.concatenate(disagreement).mean()),
        "val/actor_residual_rms": float(np.sqrt(np.mean(u**2))),
        "val/actor_saturation": float((np.abs(u) > 0.95).mean()),
        "val/self_estimated_policy_gain": float(np.concatenate(policy_gain).mean()),
        "val/policy_q_abs_max": float(np.abs(np.concatenate(policy_q)).max()),
        "val/transitions": len(indices),
    }
    known = a["mc_valid"][indices]
    if known.any():
        result["val/mc_rmse"] = float(np.sqrt(np.mean((scores[known] - a["mc_return"][indices][known]) ** 2)))
    terminal = a["terminal"][indices].astype(bool)
    if terminal.any():
        result["val/terminal_rmse"] = float(np.sqrt(np.mean((scores[terminal] - a["reward"][indices][terminal]) ** 2)))
    # Equal weight per attempt at the entry; no long-episode domination.
    ids = torch.as_tensor(entries, device="cuda")
    q1, q2 = learner.critic(tensors["x"][ids], tensors["action"][ids])
    es = torch.minimum(q1, q2).cpu().numpy()
    known = a["mc_valid"][entries]
    if known.any():
        result["val/entry_mc_rmse"] = float(np.sqrt(np.mean((es[known] - a["mc_return"][entries][known]) ** 2)))
    for arm, label in [(None, "entry"), (0, "left"), (1, "right")]:
        mask = np.ones(len(entries), dtype=bool) if arm is None else a["arm"][entries] == arm
        result[f"val/{label}_attempts"] = int(mask.sum())
        value = auc(es[mask], a["outcome"][entries][mask])
        if value is not None:
            result[f"val/{label}_auc"] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Slurm GPU required; workstation/login training forbidden")
    cfg = json.loads(args.config.read_text())
    if cfg["method"] not in {"iql", "td3bc", "rebrac"}:
        raise ValueError("unknown paper method")
    if cfg["residual_radius_std"] <= 0 or not 0 < cfg["expectile"] < 1:
        raise ValueError("invalid action radius or expectile")
    if args.resume is None:
        args.output.mkdir(exist_ok=False)
    elif not args.output.is_dir():
        raise ValueError("resume requires existing run")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    load_start = time.monotonic()
    a, tensors, stats, manifest, fm = load_data(args.replay, cfg)
    attempts = json.loads((args.replay / "attempts.json").read_text())
    train_mask = a["episode"] != 1
    groups = [np.flatnonzero(train_mask & (a["attempt"] == aid)) for aid in np.unique(a["attempt"][train_mask])]
    val = np.flatnonzero(~train_mask)
    entries = np.array(
        [item["transition_start"] for item in attempts if item["episode"] == 1 and item["entry_precedes_closure"]],
        dtype=int,
    )
    learner = OfflineLearner(tensors["x"].shape[-1], cfg)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("clean pinned training checkout required")
    identity = {
        "code_commit": commit,
        "config_sha256": sha256(args.config),
        "replay_sha256": manifest["replay_sha256"],
        "features_sha256": fm["features_sha256"],
    }
    record = {
        "config": cfg,
        **identity,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "hostname": os.uname().nodename,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "train_attempts": len(groups),
        "train_transitions": int(train_mask.sum()),
        "validation_attempts": len(entries),
        "phase": "offline actor-critic method comparison",
        "actor_updated": True,
        "pi_updated": False,
        "deployable": False,
        "encoder": fm,
        "normalization": stats,
        "sampling": "uniform attempt then uniform transition",
        "residual_data_abs_max": float(tensors["u"].abs().max()),
        "residual_support": "base + 0.25 training-action std * u; u in [-1,1]; NOT physical deployment bounds",
        "adaptations": [
            "frozen three-view ResNet18",
            "YAM 7D active-arm residual about Pi, zero-head initialization",
            "actual-time discounts from immutable replay",
            "no warmup, no online interaction, no RTC deployment",
        ],
        "evaluation_caveat": "MC is observed behavior return; optimized Q need not equal behavior return. Self-estimated policy gain is not policy improvement evidence.",
        "trainable_parameters": sum(
            p.numel()
            for module in [learner.critic, learner.actor, learner.value]
            if module is not None
            for p in module.parameters()
        ),
        "input_preparation_seconds": time.monotonic() - load_start,
    }
    start_step, best = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cuda", weights_only=True)
        if checkpoint["identity"] != identity:
            raise ValueError("resume identity changed")
        learner.load_state_dict(checkpoint["learner"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu())
        rng.bit_generator.state = checkpoint["numpy_rng"]
        start_step, best = checkpoint["step"], checkpoint["best_entry_rmse"]
    write_json(args.output / "run.json", record)

    def save(name, step):
        payload = {
            "identity": identity,
            "step": step,
            "learner": learner.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "numpy_rng": rng.bit_generator.state,
            "best_entry_rmse": best,
            "config": cfg,
            "normalization": stats,
            "observation_dim": tensors["x"].shape[-1],
            "deployable": False,
        }
        tmp = args.output / (name + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(args.output / name)

    @torch.no_grad()
    def predictions(step):
        rows = np.array([item["transition_start"] for item in attempts], dtype=int)
        ids = torch.as_tensor(rows, device="cuda")
        q1, q2 = learner.critic(tensors["x"][ids], tensors["action"][ids])
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
            actor_u=learner.actor(tensors["x"][ids]).cpu().numpy(),
        )

    start = last_log = time.monotonic()
    metrics = args.output / "metrics.jsonl"
    accumulated = defaultdict(list)
    step = start_step
    try:
        write_metrics(metrics, step, evaluate(learner, tensors, a, val, entries))
        for step in range(start_step + 1, cfg["steps"] + 1):
            selected = rng.integers(len(groups), size=cfg["batch_size"])
            rows = np.array([groups[g][rng.integers(len(groups[g]))] for g in selected])
            ids = torch.as_tensor(rows, device="cuda")
            log = learner.update({key: value[ids] for key, value in tensors.items()}, step)
            for key, value in log.items():
                accumulated[key].append(value)
            if step % cfg["log_interval"] == 0:
                now = time.monotonic()
                log = {key: float(np.mean(value)) for key, value in accumulated.items()}
                log.update(
                    actor_updates=learner.actor_updates,
                    online_environment_steps=0,
                    learning_rate=cfg["learning_rate"],
                    elapsed_seconds=now - start,
                    step_seconds=(now - last_log) / cfg["log_interval"],
                    samples_per_second=cfg["log_interval"] * cfg["batch_size"] / (now - last_log),
                    gpu_peak_memory_gib=torch.cuda.max_memory_allocated() / 1024**3,
                )
                write_metrics(metrics, step, log)
                print(json.dumps({"step": step, **log}), flush=True)
                accumulated.clear()
                last_log = now
            if step % cfg["eval_interval"] == 0 or step == cfg["steps"]:
                validation = evaluate(learner, tensors, a, val, entries)
                write_metrics(metrics, step, validation)
                if validation.get("val/entry_mc_rmse", float("inf")) < best:
                    best = validation["val/entry_mc_rmse"]
                    save("best-diagnostic.pt", step)
                write_json(args.output / "latest-evaluation.json", {"step": step, **validation})
                predictions(step)
            if step % cfg["save_interval"] == 0 or step == cfg["steps"]:
                save(f"step-{step:06d}.pt", step)
        write_json(
            args.output / "status.json",
            {
                "state": "completed",
                "step": step,
                "seconds": time.monotonic() - start,
                "actor_updates": learner.actor_updates,
                "deployable": False,
                "online_environment_steps": 0,
            },
        )
    except BaseException as exc:
        write_json(
            args.output / "status.json",
            {
                "state": "failed",
                "step": step,
                "error": repr(exc),
                "seconds": time.monotonic() - start,
                "deployable": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
