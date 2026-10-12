"""Meaningful numerical checks; NEVER run on the workstation or login node."""

# ruff: noqa: E402

import argparse
import copy
import importlib.metadata
import os
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/pld-rl/src"))
from yam_pld_rl.data import write_json
from yam_pld_rl.networks import TwinCritic
from yam_pld_rl.networks import calibrated_penalty


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Slurm GPU required")
    torch.manual_seed(20261012)
    values = torch.tensor([[0.2, 0.4], [0.8, 1.2]], device="cuda", requires_grad=True)
    behavior = torch.tensor([0.5, 1.0], device="cuda", requires_grad=True)
    returns = torch.tensor([0.6, 1.0], device="cuda")
    penalty = calibrated_penalty(values, behavior, returns).sum()
    penalty.backward()
    torch.testing.assert_close(values.grad, torch.tensor([[0.0, 0.0], [0.0, 0.5]], device="cuda"))
    torch.testing.assert_close(behavior.grad, -torch.ones(2, device="cuda"))
    model = TwinCritic(23, width=32).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    x, a = torch.randn(8, 23, device="cuda"), torch.randn(8, 7, device="cuda")
    q1, q2 = model(x, a)
    (q1.square().mean() + q2.square().mean()).backward()
    optimizer.step()
    if not all(torch.isfinite(t).all() for t in model.parameters()):
        raise ValueError("nonfinite actual CUDA optimizer update")
    with tempfile.TemporaryDirectory(prefix="pld-gpu-check-") as tmp:
        path = Path(tmp) / "state.pt"
        torch.save({"critic": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
        restored = TwinCritic(23, width=32).cuda()
        opt2 = torch.optim.Adam(restored.parameters(), lr=3e-4)
        loaded = torch.load(path, map_location="cuda", weights_only=True)
        restored.load_state_dict(loaded["critic"])
        opt2.load_state_dict(loaded["optimizer"])
        for q, r in zip(model(x, a), restored(x, a), strict=True):
            torch.testing.assert_close(q, r, rtol=0, atol=0)
        for network, opt in [(model, optimizer), (restored, opt2)]:
            opt.zero_grad(set_to_none=True)
            u, v = network(x, a)
            (u.square().mean() + v.square().mean()).backward()
            opt.step()
        for original, restored_value in zip(model.parameters(), restored.parameters(), strict=True):
            torch.testing.assert_close(original, restored_value, rtol=0, atol=0)
    target = copy.deepcopy(model).requires_grad_(requires_grad=False)
    before = next(target.parameters()).clone()
    with torch.no_grad():
        next(model.parameters()).add_(1)
        for to, current in zip(target.parameters(), model.parameters(), strict=True):
            to.lerp_(current, 0.005)
    torch.testing.assert_close(next(target.parameters()), before + 0.005)
    write_json(
        args.output,
        {
            "passed": True,
            "calibration_gradient": True,
            "cuda_optimizer_update": True,
            "checkpoint_exact_forward": True,
            "optimizer_exact_next_update": True,
            "target_update": True,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "versions": {
                name: importlib.metadata.version(name)
                for name in ["torch", "torchvision", "numpy", "pillow", "h5py", "av"]
            },
        },
    )


if __name__ == "__main__":
    main()
