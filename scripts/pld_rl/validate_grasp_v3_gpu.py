"""Explicit Thor/Slurm numerical checks for v3; synthetic data only."""
# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "packages/pld-rl/src"), str(ROOT / "packages/vla-platform/src")]
from grasp_v3_checks import modern_recording
from train_sac import export_bundle
from yam_pld_rl.block_contract import feature_dim
from yam_pld_rl.block_replay import compile_replay
from yam_pld_rl.curator import associate_rewards
from yam_pld_rl.data import sha256
from yam_pld_rl.data import write_json
from yam_pld_rl.execution import require_training_runtime
from yam_pld_rl.sac import SacLearner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--execution", choices=("slurm", "thor-offline"), default="slurm")
    args = p.parse_args()
    execution = require_training_runtime(args.execution, cuda_available=torch.cuda.is_available(),
                                         capability=torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(918)
    from block_fixtures import stats  # noqa: PLC0415

    cfg = json.loads((ROOT / "configs/pld_rl/sac-block-v2.json").read_text())
    cfg.update(steps=40, critic_warmup_steps=4, save_interval=40, log_interval=10)
    all_data, audits = [], []
    for kind in ("held", "empty", "not_lifted", "no_closure", "dropped"):
        c, rows, records, candidates, reviews = modern_recording(kind)
        linked, labels, audit = associate_rewards(records, candidates, reviews, rows, "train")
        assert not audit, audit
        manifest = {"contract": c, "normalization": stats(), "encoder_sha256": sha256(args.encoder)}
        data, audit = compile_replay(linked, labels, c, {"fixture": manifest})
        data["attempt"] = np.asarray([kind+"/"+a for a in data["attempt"]])
        data["episode"] = np.asarray([kind+"/"+a for a in data["episode"]])
        all_data.append(data)
        audits.append({"outcome": kind, **audit, "reward_sum": float(data["reward"].sum())})
    arrays = {k: np.concatenate([d[k] for d in all_data]) for k in all_data[0]}
    replay = args.output/"replay"
    replay.mkdir()
    np.savez_compressed(replay/"replay.npz", **arrays)
    write_json(replay/"manifest.json", {"contract": c, "replay_sha256": sha256(replay/"replay.npz"), "snapshots": {"fixture": manifest}})
    learner = SacLearner(feature_dim(c), cfg)
    b = {k: torch.as_tensor(arrays[k], device="cuda", dtype=torch.float32)
         for k in ("x", "nx", "scale", "next_scale", "negative_scale", "next_negative_scale", "action", "reward", "terminal", "discount")}
    forced = (b["scale"]+b["negative_scale"]).sum(-1) == 0
    assert forced.any()
    # Mandatory queue drainage must not update actor or entropy temperature.
    before = {k: v.clone() for k, v in learner.actor.state_dict().items()}
    for _ in range(2):
        learner.update({k: v[forced] for k, v in b.items()})
    assert learner.actor_updates == 0
    assert all(torch.equal(before[k], v) for k, v in learner.actor.state_dict().items())
    target = learner.backup(b)
    assert torch.isfinite(target).all()
    for _ in range(4):
        learner.update(b)
    assert learner.actor_updates > 0
    export_bundle(args.output/"numerical-bundle", learner, c, stats(), sha256(args.encoder), cfg, {"synthetic": True})
    for name, value in (("config", cfg), ("contract", c), ("normalization", stats())):
        write_json(args.output/f"{name}.json", value)
    subprocess.run([sys.executable, str(ROOT/"scripts/pld_rl/train_sac.py"), "--config", str(args.output/"config.json"),
                    "--contract", str(args.output/"contract.json"), "--normalization", str(args.output/"normalization.json"),
                    "--encoder", str(args.encoder), "--replay", str(replay), "--output", str(args.output/"cli-run"),
                    "--execution", args.execution], check=True)
    checkpoint = torch.load(args.output/"cli-run/checkpoint-000040.pt", map_location="cuda", weights_only=True)
    restored = SacLearner(feature_dim(c), cfg)
    restored.load_state_dict(checkpoint["learner"])
    u, _ = restored.actor(b["x"], deterministic=True)
    assert torch.isfinite(u).all()
    write_json(args.output/"result.json", {"status": "passed_synthetic_only", "job": execution["job"], "execution": execution,
               "feature_dim": feature_dim(c), "outcome_audits": audits, "forced_noop_actor_updates": 0,
               "cli_critic_updates": restored.updates, "cli_actor_updates": restored.actor_updates,
               "robot_success_rate": None, "production_actor_trained": False})
    print((args.output/"result.json").read_text())


if __name__ == "__main__":
    main()
