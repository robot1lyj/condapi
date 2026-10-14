"""Explicit Thor/Slurm PLD-Grasp SAC update/export. No physical actuation."""

# ruff: noqa: E402

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "packages/pld-rl/src"), str(ROOT / "packages/vla-platform/src")]
from vla_platform.metrics import write_metrics
from yam_pld_rl.block_contract import feature_dim
from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.block_contract import validate_contract
from yam_pld_rl.calibrated import CalibratedLearner
from yam_pld_rl.collector import native_trajectory_manifest
from yam_pld_rl.data import sha256
from yam_pld_rl.data import write_json
from yam_pld_rl.execution import require_training_runtime
from yam_pld_rl.learning_metrics import evaluate
from yam_pld_rl.sac import SacLearner


def learner_for(dim, cfg, device="cuda"):
    if cfg.get("method") in ("calql", "parts_td3bc"):
        return CalibratedLearner(dim, cfg, device)
    return SacLearner(dim, cfg, device)


def export_bundle(path, learner, contract, normalization, encoder_sha, config, provenance):
    path.mkdir(exist_ok=False)
    shutil.copyfile(ROOT / "third_party/yam-fk-model/yam_linear4310.xml", path / "fk.xml")
    torch.save({k: v.cpu() for k, v in learner.actor.state_dict().items()}, path / "actor.pt")
    manifest = {"schema": "pld_sac_bundle_v2", "contract": contract,
               "normalization": normalization, "encoder_sha256": encoder_sha, "config": config,
               "observation_dim": feature_dim(contract), "actor_sha256": sha256(path / "actor.pt"),
               "provenance": provenance, "robot_validated": False,
               "phase": "candidate_requires_AB" if learner.actor_updates or provenance.get("actor_prior_updates", 0) else "bootstrap_exploration",
               "entropy_semantics": "tanh latent u; critic action is state-feasible normalized physical offset"}
    if contract.get("trajectory"):
        manifest["entropy_semantics"] = "tanh latent target; critic action precedes deterministic recorded trajectory/geometry decoding"
        manifest = native_trajectory_manifest(manifest, behavior_snapshot=config["behavior_snapshot_sha256"])
    write_json(path / "manifest.json", manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--normalization", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--replay", type=Path, help="omit for zero-mean bootstrap export only")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execution", choices=("slurm", "thor-offline"), default="slurm")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--resume", type=Path, help="exact same run/input identity")
    group.add_argument("--initialize-learner", type=Path, help="previous round checkpoint, new replay/output")
    args = p.parse_args()
    execution = require_training_runtime(args.execution, cuda_available=torch.cuda.is_available(),
                                         capability=torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
    cfg, contract, stats = [json.loads(path.read_text()) for path in (args.config, args.contract, args.normalization)]
    validate_contract(contract)
    if not 0 <= cfg.get("actor_prior_steps", 0) <= cfg["critic_warmup_steps"]:
        raise ValueError("coordinate alignment must finish before RL actor updates")
    torch.set_num_threads(2)
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(cfg["seed"])
    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)
    calibrated = cfg.get("method") in ("calql", "parts_td3bc")
    if calibrated and not args.replay:
        raise ValueError("calibrated methods require verified real replay")
    learner = learner_for(feature_dim(contract), cfg)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("clean pinned source required")
    inputs = {str(path): sha256(path) for path in (args.config, args.contract, args.normalization, args.encoder)}
    replay_manifest, arrays, tensors = None, None, None
    groups = {False: [], True: []}
    if args.replay:
        replay_manifest = json.loads((args.replay / "manifest.json").read_text())
        if (replay_manifest["contract"] != contract
                or sha256(args.replay / "replay.npz") != replay_manifest["replay_sha256"]):
            raise ValueError("replay contract/hash mismatch")
        inputs[str(args.replay / "manifest.json")] = sha256(args.replay / "manifest.json")
        for snapshot in replay_manifest["snapshots"].values():
            if snapshot["normalization"] != stats or snapshot["encoder_sha256"] != sha256(args.encoder):
                raise ValueError("feature encoder/normalization changed")
        with np.load(args.replay / "replay.npz", allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        if calibrated and (replay_manifest.get("schema") != "pld_learning_replay_v1"
                           or not replay_manifest.get("discounted_return_preserved")):
            raise ValueError("calibrated comparison requires verified learning view")
        if "feature_group_rms" in cfg:
            provenance = cfg["feature_balance_source"]
            if (provenance["replay_sha256"] != replay_manifest["replay_sha256"]
                    or provenance["training_rows"] != int(arrays["train"].sum())):
                raise ValueError("feature balance statistics belong to another replay/split")
        train = arrays["train"].astype(bool)
        online = arrays["online"].astype(bool)
        if not np.any(train & online & arrays["realized_nonzero"].astype(bool) & (np.abs(arrays["action"]).max(1) > 1e-6)):
            raise ValueError("SAC actor updates require actual nonzero executed corrections")
        for is_online in (False, True):
            mask = train & (online == is_online)
            groups[is_online] = [np.flatnonzero(mask & (arrays["attempt"] == a)) for a in np.unique(arrays["attempt"][mask])]
        tensors = {k: torch.as_tensor(arrays[k], device="cuda", dtype=torch.float32)
                   for k in ("x", "nx", "scale", "next_scale", "negative_scale", "next_negative_scale", "action", "reward", "terminal", "discount")}
        if calibrated:
            tensors.update({k: torch.as_tensor(arrays[k], device="cuda", dtype=torch.float32)
                            for k in ("mc_return", "attempt_outcome")})
        if cfg.get("actor_prior_steps", 0):
            if not calibrated or not contract.get("trajectory"):
                raise ValueError("coordinate initialization requires the calibrated trajectory learner")
            tensors["prior_action"] = torch.as_tensor(arrays["prior_action"], device="cuda", dtype=torch.float32)
    identity = {"code_commit": commit, "inputs": inputs, "execution_target": args.execution}
    policy_contract = {"contract": contract, "normalization": stats, "encoder_sha256": sha256(args.encoder), "config": cfg}
    begin = 0
    if args.resume or args.initialize_learner:
        ckpt = torch.load(args.resume or args.initialize_learner, map_location="cuda", weights_only=True)
        if ckpt["policy_contract"] != policy_contract:
            raise ValueError("learner contract changed; start a new explicit experiment")
        learner.load_state_dict(ckpt["learner"])
        if args.resume:
            if ckpt["identity"] != identity:
                raise ValueError("resume source/data identity mismatch")
            begin = ckpt["step"]
            rng.bit_generator.state = ckpt["numpy_rng"]
            torch.set_rng_state(ckpt["torch_rng"].cpu())
            torch.cuda.set_rng_state(ckpt["cuda_rng"].cpu())
    run = {**identity, "producer": "SAC-Coach", "policy_expert": "PLD-Grasp", "policy_contract": policy_contract,
           "job": execution["job"], "execution": execution,
           "hostname": os.uname().nodename, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
           "replay": replay_manifest, "pi_updated": False, "robot_success_rate": None,
           "parent_checkpoint": str(args.initialize_learner) if args.initialize_learner else None,
           "parent_checkpoint_sha256": sha256(args.initialize_learner) if args.initialize_learner else None,
           "phase": "online replay SAC update" if args.replay else "zero-mean exploration bootstrap",
           "offline_probe_attempts": len(groups[False]), "online_attempts": len(groups[True]),
           "mix": "50% probe / 50% collect when both exist; otherwise collect only",
           "adaptations": ["YAM joint envelope with native RTC cadence rather than EE action", "frozen ResNet18",
                           "no transplant of incompatible historical critics", "no CalQL pretraining in this version"]}
    write_json(args.output / "run.json", run)
    if calibrated:
        successes = [g for g in groups[True] + groups[False] if arrays["attempt_outcome"][g[0]] == 1]
        all_groups = groups[True] + groups[False]
        if not successes:
            raise ValueError("success-prior comparison needs successful training attempts")
        run.update(phase="fixed real replay comparison; no new environment data",
                   method=cfg["method"], new_environment_steps=0,
                   mix="50% successful attempts + 50% all attempts; uniform attempt then decision",
                   adaptations=["return-preserving collapse of mandatory RTC wait suffix", "7D feasible residual with original radius",
                                "frozen Pi/ResNet18", "no entropy in Bellman backup", "offline replay; not complete online paper reproduction",
                                "current-policy + uniform latent proposals for CalQL; MC bound on policy proposals",
                                "optional recorded-MC critic initialization; success prior is corrected collect, not pure Pi demonstrations",
                                "TD3BC mean actor; fixed bootstrap collection std; success BC and optional failed-zero anchor"])
        if "feature_group_rms" in cfg:
            run["adaptations"].append("fixed train-only RMS balance of five observation groups; original state and bounds preserved")
        write_json(args.output / "run.json", run)

    def save(step):
        path = args.output / f"checkpoint-{step:06d}.pt"
        if path.exists():
            return
        tmp = path.with_suffix(".tmp")
        torch.save({"identity": identity, "policy_contract": policy_contract, "step": step,
                    "learner": learner.state_dict(), "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, tmp)
        tmp.replace(path)
        export_bundle(args.output / f"bundle-{step:06d}", learner, contract, stats, sha256(args.encoder), cfg,
                      {**identity, "checkpoint_sha256": sha256(path), "actor_updates": learner.actor_updates,
                       "actor_prior_updates": getattr(learner, "actor_prior_updates", 0)})

    started = time.monotonic()
    if tensors is None:
        if args.resume or args.initialize_learner:
            raise ValueError("bootstrap must start from zero actor, no existing learner")
        save(0)
        print("Exported zero-mean bootstrap. This is untrained exploration, not an improved expert.", flush=True)
        return

    def draw(group, count):
        indices = rng.integers(len(group), size=count)
        return [group[i][rng.integers(len(group[i]))] for i in indices]

    for step in range(begin + 1, cfg["steps"] + 1):
        if calibrated:
            n = cfg["batch_size"] // 2
            ids = draw(successes, n) + draw(all_groups, cfg["batch_size"]-n)
        else:
            n = cfg["batch_size"] // 2 if groups[False] else 0
            ids = draw(groups[False], n) if n else []
            ids += draw(groups[True], cfg["batch_size"] - n)
        batch = {k: v[ids] for k, v in tensors.items()}
        log = learner.update(batch, update_actor=step > cfg["critic_warmup_steps"])
        if step <= cfg.get("actor_prior_steps", 0):
            log["train/actor_prior_alignment_loss"] = learner.align_actor_prior(batch)
        log["actor_prior_updates"] = getattr(learner, "actor_prior_updates", 0)
        if step % cfg["log_interval"] == 0 or step == cfg["steps"]:
            log.update(elapsed_seconds=time.monotonic() - started, learning_rate=cfg["learning_rate"],
                       online_environment_steps=int((arrays["online"] & arrays["train"]).sum()),
                       online_executed_seconds=float(arrays["dt"][arrays["online"] & arrays["train"]].sum()))
            val = np.flatnonzero(~arrays["train"].astype(bool))
            if calibrated:
                metrics, details = evaluate(learner, tensors, arrays)
                log.update(metrics, new_environment_steps=0)
                write_json(args.output / "latest-evaluation.json", {"step": step, "metrics": details,
                           "robot_success_rate": None, "recorded_returns_are_not_policy_value_ground_truth": True})
                if step % cfg["save_interval"] == 0 or step == cfg["steps"]:
                    write_json(args.output / f"evaluation-{step:06d}.json", {"step": step, "metrics": details})
            elif len(val):
                with torch.no_grad():
                    b = {k: v[val] for k, v in tensors.items()}
                    q1, q2 = learner.critic(b["x"], b["action"])
                    y = learner.backup(b)
                    log["val_loss"] = float(((q1 - y).square() + (q2 - y).square()).mean() / 2)
                    terminal = b["terminal"] == 1
                    if terminal.any():
                        log["val/terminal_rmse"] = float((torch.minimum(q1, q2)[terminal] - b["reward"][terminal]).square().mean().sqrt())
            write_metrics(args.output / "metrics.jsonl", step, log)
        if step % cfg["save_interval"] == 0 or step == cfg["steps"]:
            save(step)
    write_json(args.output / "complete.json", {"steps": cfg["steps"], "seconds": time.monotonic() - started,
               "actor_updates": learner.actor_updates, "robot_success_rate": None,
               "actor_prior_updates": getattr(learner, "actor_prior_updates", 0),
               "bundle": f"bundle-{cfg['steps']:06d}", "contract_sha256": fingerprint(contract)})


if __name__ == "__main__":
    main()
