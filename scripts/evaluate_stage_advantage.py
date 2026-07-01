"""Evaluate a PyTorch Stage Advantage checkpoint on a labeled LeRobot dataset."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import random
import time
import warnings

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated.*",
)

import jax
import numpy as np
import safetensors.torch
import torch
import tqdm

import openpi.models.pi0_config as pi0_config
import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
import openpi.training.config as _config
import openpi.training.data_loader as _data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Checkpoint directory containing model.safetensors")
    parser.add_argument(
        "--config-name",
        default="ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD",
        help="Stage Advantage TrainConfig name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Labeled validation LeRobot dataset path or repo_id",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=50, help="Finite validation batches to evaluate")
    parser.add_argument(
        "--samples-per-batch",
        type=int,
        default=1,
        help="Average this many stochastic value predictions per batch",
    )
    parser.add_argument("--sign-epsilon", type=float, default=1e-4, help="Ignore near-zero targets for sign accuracy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> _config.TrainConfig:
    config = _config.get_config(args.config_name)
    data = dataclasses.replace(config.data, repo_id=args.dataset)
    model = dataclasses.replace(config.model, dtype=config.pytorch_training_precision)
    return dataclasses.replace(
        config,
        exp_name=f"eval_{args.checkpoint.name}",
        data=data,
        model=model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        wandb_enabled=False,
    )


def _load_model(config: _config.TrainConfig, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    if not isinstance(config.model, pi0_config.AdvantageEstimatorConfig):
        raise TypeError("Stage Advantage evaluation requires AdvantageEstimatorConfig.")
    model = pi0_pytorch.AdvantageEstimator(config.model).to(device)
    model_path = checkpoint / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint model: {model_path}")
    safetensors.torch.load_model(model, model_path, strict=True)
    model.eval()
    return model


def _safe_corrcoef(preds: torch.Tensor, targets: torch.Tensor) -> float | None:
    if preds.numel() < 2:
        return None
    pred_std = torch.std(preds)
    target_std = torch.std(targets)
    if pred_std <= 1e-12 or target_std <= 1e-12:
        return None
    return float(torch.corrcoef(torch.stack([preds, targets]))[0, 1].item())


def evaluate(args: argparse.Namespace) -> dict:
    if args.max_batches <= 0:
        raise ValueError("--max-batches must be positive because the OpenPI torch loader loops indefinitely.")
    if args.samples_per_batch <= 0:
        raise ValueError("--samples-per-batch must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    config = _build_config(args)
    loader = _data.create_data_loader(
        config,
        framework="pytorch",
        shuffle=False,
        num_batches=args.max_batches,
        skip_norm_stats=config.skip_norm_stats,
    )
    model = _load_model(config, args.checkpoint, device)

    pred_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    loss_sum = 0.0
    abs_sum = 0.0
    sign_correct = 0
    sign_total = 0
    sample_count = 0
    start = time.time()

    with torch.inference_mode():
        for batch_idx, (observation, actions) in enumerate(tqdm.tqdm(loader, total=args.max_batches, desc="Eval")):
            del actions
            observation = jax.tree.map(lambda x: x.to(device), observation)
            target = observation.progress.to(device=device, dtype=torch.float32).reshape(-1)

            pred_accum = torch.zeros_like(target)
            for sample_idx in range(args.samples_per_batch):
                torch.manual_seed(args.seed + batch_idx * args.samples_per_batch + sample_idx)
                pred_accum += model.sample_values(device, observation).to(torch.float32).reshape(-1)
            pred = pred_accum / args.samples_per_batch

            error = pred - target
            loss_sum += float(torch.sum(error.square()).item())
            abs_sum += float(torch.sum(error.abs()).item())
            sample_count += int(target.numel())

            sign_mask = target.abs() >= args.sign_epsilon
            if torch.any(sign_mask):
                sign_correct += int(((pred[sign_mask] >= 0) == (target[sign_mask] >= 0)).sum().item())
                sign_total += int(sign_mask.sum().item())

            pred_chunks.append(pred.detach().cpu())
            target_chunks.append(target.detach().cpu())

    preds = torch.cat(pred_chunks)
    targets = torch.cat(target_chunks)
    mse = loss_sum / sample_count
    mae = abs_sum / sample_count
    ss_res = float(torch.sum((preds - targets).square()).item())
    ss_tot = float(torch.sum((targets - targets.mean()).square()).item())

    report = {
        "checkpoint": str(args.checkpoint),
        "config_name": args.config_name,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "samples_per_batch": args.samples_per_batch,
        "num_pairs": sample_count,
        "mse": mse,
        "rmse": mse**0.5,
        "mae": mae,
        "sign_accuracy": (sign_correct / sign_total) if sign_total else None,
        "sign_total": sign_total,
        "corrcoef": _safe_corrcoef(preds, targets),
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else None,
        "pred_mean": float(preds.mean().item()),
        "pred_std": float(preds.std().item()),
        "target_mean": float(targets.mean().item()),
        "target_std": float(targets.std().item()),
        "elapsed_sec": time.time() - start,
        "device": str(device),
    }
    return report


def main() -> None:
    args = _parse_args()
    report = evaluate(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
