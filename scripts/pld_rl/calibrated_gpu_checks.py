"""Explicit GPU-only validation of calibrated objectives, export and CLI."""
# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "packages/pld-rl/src"), str(ROOT / "packages/vla-platform/src")]
from block_fixtures import stats
from grasp_v3_checks import modern_recording
from train_sac import export_bundle
from yam_pld_rl.block_contract import feature_dim
from yam_pld_rl.block_replay import compile_replay
from yam_pld_rl.calibrated import CalibratedLearner
from yam_pld_rl.calibrated import calibrated_gap
from yam_pld_rl.calibrated import feasible
from yam_pld_rl.curator import associate_rewards
from yam_pld_rl.data import sha256
from yam_pld_rl.data import write_json
from yam_pld_rl.execution import require_training_runtime
from yam_pld_rl.feature_balance import FeatureBalance
from yam_pld_rl.learning_metrics import evaluate
from yam_pld_rl.learning_view import learning_view
from yam_pld_rl.sac import SquashedActor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--execution", choices=("slurm", "thor-offline"), required=True)
    args = p.parse_args()
    require_training_runtime(args.execution, cuda_available=torch.cuda.is_available(),
                             capability=torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(813)
    balance = FeatureBalance(2287, [1., 2., 3., 4., 5.]).cuda()
    example = torch.ones(2, 2287, device="cuda")
    torch.testing.assert_close(balance(example)[:, 1550:2250], example[:, 1550:2250]/3)
    # MC lower bound must suppress gradients only for below-bound policy proposals.
    pq = torch.tensor([[-.2, .8]], device="cuda", requires_grad=True)
    rq = torch.zeros_like(pq, requires_grad=True)
    dq = torch.zeros(1, device="cuda", requires_grad=True)
    gap = calibrated_gap(rq, pq, dq, torch.zeros_like(pq), torch.tensor([.5], device="cuda"))
    gap.backward()
    assert pq.grad[0, 0] == 0
    assert pq.grad[0, 1] > 0
    assert dq.grad[0] == -1
    data = []
    for i, kind in enumerate(("held", "empty", "held", "empty")):
        c, rows, records, candidates, reviews = modern_recording(kind)
        linked, labels, audit = associate_rewards(records, candidates, reviews, rows, "train")
        assert not audit
        snapshot = {"contract": c, "normalization": stats(), "encoder_sha256": sha256(args.encoder)}
        native, _ = compile_replay(linked, labels, c, {"fixture": snapshot})
        native["attempt"] = np.asarray([str(i)+"/"+a for a in native["attempt"]])
        native["episode"] = np.asarray([str(i)+"/"+a for a in native["episode"]])
        native["train"][:] = i < 2
        learning, _ = learning_view(native)
        data.append(learning)
    arrays = {k: np.concatenate([d[k] for d in data]) for k in data[0]}
    replay = args.output / "replay"
    replay.mkdir()
    np.savez_compressed(replay/"replay.npz", **arrays)
    write_json(replay/"manifest.json", {"schema": "pld_learning_replay_v1", "contract": c,
               "snapshots": {"fixture": snapshot}, "replay_sha256": sha256(replay/"replay.npz"),
               "discounted_return_preserved": True})
    keys = ("x", "nx", "action", "scale", "negative_scale", "next_scale", "next_negative_scale",
            "reward", "terminal", "discount", "mc_return", "attempt_outcome")
    b = {k: torch.as_tensor(arrays[k], device="cuda", dtype=torch.float32) for k in keys}
    write_json(args.output/"contract.json", c)
    write_json(args.output/"normalization.json", stats())
    results = []
    for variant in ("calql", "parts_td3bc", "parts-balanced", "sac", "iql"):
        method = "parts_td3bc" if variant == "parts-balanced" else variant
        base_method = "parts_td3bc" if method in ("sac", "iql") else method
        cfg = json.loads((ROOT/f"configs/pld_rl/collect13-r02/{base_method}-s0.json").read_text())
        cfg.update(method=method, backup_entropy=method == "sac", iql_expectile=0.7, iql_temperature=3.)
        if variant in ("parts-balanced", "sac", "iql"):
            cfg["feature_group_rms"] = [1., 2., 3., 4., 5.]
            cfg["feature_balance_source"] = {"replay_sha256": sha256(replay/"replay.npz"),
                                             "training_rows": int(arrays["train"].sum())}
        cfg.update(steps=40, mc_warmup_steps=4, critic_warmup_steps=8, log_interval=10, save_interval=20)
        learner = CalibratedLearner(feature_dim(c), cfg)
        frozen_q = {k: v.clone() for k, v in learner.critic.state_dict().items()}
        learner.align_actor_prior({**b, "prior_action": b["action"]})
        assert learner.actor_prior_updates == 1
        assert learner.actor_updates == 0
        assert all(torch.equal(frozen_q[k], v) for k, v in learner.critic.state_dict().items())
        for key, value in learner.actor.state_dict().items():
            torch.testing.assert_close(value, learner.actor_target.state_dict()[key], rtol=0, atol=0)
        terminal = b["terminal"] == 1
        torch.testing.assert_close(learner.backup(b)[terminal], b["reward"][terminal])
        if method == "iql":
            # In-sample IQL backup must never query a proposed actor action.
            with patch.object(learner.actor, "forward", side_effect=AssertionError("IQL queried actor in backup")):
                target = learner.backup(b)
            live = ~terminal
            expected = b["reward"][live]+b["discount"][live]*learner.value(b["nx"][live]).squeeze(-1)
            torch.testing.assert_close(target[live], expected)
        if method == "sac":
            torch.manual_seed(291)
            target = learner.backup(b)
            torch.manual_seed(291)
            live = ~terminal
            u, logp = learner.actor(b["nx"][live])
            a = feasible(u, b["next_scale"][live], b["next_negative_scale"][live])
            q = torch.minimum(*learner.target(b["nx"][live], a))
            active = (b["next_scale"][live]+b["next_negative_scale"][live]).sum(-1) > 0
            expected = b["reward"][live]+b["discount"][live]*(q-learner.log_alpha.exp()*logp*active)
            torch.testing.assert_close(target[live], expected)
        original = {k: v.clone() for k, v in learner.actor.state_dict().items()}
        for _ in range(4):
            learner.update(b, update_actor=False)
        assert learner.actor_updates == 0
        assert all(torch.equal(original[k], v) for k, v in learner.actor.state_dict().items())
        for _ in range(4):
            learner.update(b)
        assert learner.actor_updates == 2
        # A collapsed physical action limit cannot generate an actor update.
        no_control = {**b, "scale": b["scale"]*0, "negative_scale": b["negative_scale"]*0}
        learner.update(no_control)
        learner.update(no_control)
        assert learner.actor_updates == 2
        before = torch.cuda.get_rng_state().clone()
        evaluate(learner, b, arrays)
        assert torch.equal(before, torch.cuda.get_rng_state())
        u, _ = learner.actor(b["x"], deterministic=True)
        a = feasible(u, b["scale"], b["negative_scale"])
        assert torch.all(a <= b["scale"]+1e-6)
        assert torch.all(a >= -b["negative_scale"]-1e-6)
        restored = CalibratedLearner(feature_dim(c), cfg)
        restored.load_state_dict(learner.state_dict())
        torch.testing.assert_close(restored.actor(b["x"], deterministic=True)[0], u, rtol=0, atol=0)
        if method == "iql":
            torch.testing.assert_close(restored.value(b["x"]), learner.value(b["x"]), rtol=0, atol=0)
            torch.testing.assert_close(restored.iql_log_std, learner.iql_log_std, rtol=0, atol=0)
        bundle = args.output/(variant+"-bundle")
        export_bundle(bundle, learner, c, stats(), sha256(args.encoder), cfg, {"synthetic": True})
        runtime_actor = SquashedActor(feature_dim(c), cfg).cuda()
        runtime_actor.load_state_dict(torch.load(bundle/"actor.pt", weights_only=True, map_location="cuda"))
        torch.testing.assert_close(runtime_actor(b["x"], deterministic=True)[0], u, rtol=0, atol=0)
        cfgpath = args.output/(variant+".json")
        write_json(cfgpath, cfg)
        subprocess.run([sys.executable, str(ROOT/"scripts/pld_rl/train_sac.py"), "--config", str(cfgpath),
                        "--contract", str(args.output/"contract.json"), "--normalization", str(args.output/"normalization.json"),
                        "--encoder", str(args.encoder), "--replay", str(replay), "--output", str(args.output/(variant+"-cli")),
                        "--execution", args.execution], check=True)
        complete = json.loads((args.output/(variant+"-cli/complete.json")).read_text())
        assert complete["actor_updates"] == 16
        results.append({"method": variant, "status": "passed_synthetic_only", "actor_updates": 16})
    write_json(args.output/"result.json", {"checks": results, "robot_success_rate": None, "production_modified": False})
    print((args.output/"result.json").read_text())


if __name__ == "__main__":
    main()
