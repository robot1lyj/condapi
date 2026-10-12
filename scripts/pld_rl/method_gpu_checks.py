"""Server CUDA checks of paper objectives, action support and exact continuation."""

# ruff: noqa: E402

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/pld-rl/src"))
from yam_pld_rl.data import write_json
from yam_pld_rl.offline_methods import OfflineLearner
from yam_pld_rl.offline_methods import bellman_target
from yam_pld_rl.offline_methods import expectile_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Slurm GPU required")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.testing.assert_close(
        expectile_loss(torch.tensor([2.0, -2.0], device="cuda"), 0.7), torch.tensor([2.8, 1.2], device="cuda")
    )
    r = torch.tensor([1.0, 0.0], device="cuda")
    torch.testing.assert_close(
        bellman_target(
            r,
            torch.ones_like(r) * 0.9,
            torch.tensor([1.0, 0.0], device="cuda"),
            torch.tensor([-100.0, 0.5], device="cuda"),
        ),
        torch.tensor([1.0, 0.45], device="cuda"),
    )
    results = []
    for method in ["iql", "td3bc", "rebrac"]:
        cfg = json.loads((ROOT / f"configs/pld_rl/methods14-pld-{method}-s0.json").read_text())
        cfg.update(width=32)
        learner = OfflineLearner(23, cfg)
        b = {
            "x": torch.randn(16, 23, device="cuda"),
            "nx": torch.randn(16, 23, device="cuda"),
            "base": torch.randn(16, 7, device="cuda"),
            "next_base": torch.randn(16, 7, device="cuda"),
            "u": torch.ones(16, 7, device="cuda") * 0.1,
            "next_u": torch.ones(16, 7, device="cuda") * 0.1,
            "reward": torch.zeros(16, device="cuda"),
            "discount": torch.ones(16, device="cuda") * 0.99,
            "terminal": torch.zeros(16, device="cuda"),
        }
        b["action"] = b["base"] + 0.25 * b["u"]
        torch.testing.assert_close(learner.actor(b["x"]), torch.zeros_like(b["u"]), rtol=0, atol=0)
        if method != "iql":
            # Isolate the next-action penalty from network fitting and noise.
            fixture = OfflineLearner(23, cfg)
            with torch.no_grad():
                for head in [fixture.target.q1, fixture.target.q2]:
                    head[-1].weight.zero_()
                    head[-1].bias.fill_(0.5)
            expected = 0.99 * (0.5 - (0.07 if method == "rebrac" else 0.0))
            torch.testing.assert_close(fixture.backup(b, noise=False), torch.full_like(b["reward"], expected))
        before = copy.deepcopy(learner.actor.state_dict())
        for step in range(1, 5):
            learner.update(b, step)
        if not any(not torch.equal(before[k], v) for k, v in learner.actor.state_dict().items()):
            raise AssertionError("actor failed to update with nonzero residual fixture")
        with tempfile.TemporaryDirectory(prefix="pld-method-check-") as tmp:
            path = Path(tmp) / "resume.pt"
            torch.save(learner.state_dict(), path)
            restored = OfflineLearner(23, cfg)
            restored.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
            state = torch.cuda.get_rng_state()
            learner.update(b, 6)
            torch.cuda.set_rng_state(state)
            restored.update(b, 6)
            for name in ["critic", "actor", "target", "actor_target", "value"]:
                if getattr(learner, name) is not None:
                    for left, right in zip(
                        getattr(learner, name).parameters(), getattr(restored, name).parameters(), strict=True
                    ):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
        with torch.no_grad():
            terminal = dict(b, terminal=torch.ones_like(b["terminal"]), reward=torch.ones_like(b["reward"]))
            torch.testing.assert_close(learner.backup(terminal), terminal["reward"], rtol=0, atol=0)
            if learner.actor(b["x"]).abs().max() > 1:
                raise AssertionError("actor escaped training action support")
        results.append(
            {
                "method": method,
                "actual_cuda_updates": True,
                "actor_updated": True,
                "exact_optimizer_next_update": True,
                "terminal_masks_bootstrap": True,
                "zero_head_initialization": True,
            }
        )
    write_json(
        args.output,
        {
            "passed": True,
            "expectile_numeric_fixture": True,
            "terminal_numeric_fixture": True,
            "methods": results,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
        },
    )


if __name__ == "__main__":
    main()
