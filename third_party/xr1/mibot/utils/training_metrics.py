"""Observable optimizer-step metrics for the native Lightning/DeepSpeed loop."""

import json
import math
from pathlib import Path
import time

from lightning.pytorch.callbacks import Callback
import torch


class TrainingMetrics(Callback):
    def __init__(self, directory, save_interval):
        self.path = Path(directory) / "metrics.jsonl"
        self.save_interval = save_interval
        self.last_step = 0
        self.last_time = None

    def on_train_start(self, trainer, pl_module):
        self.last_step = trainer.global_step

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # XR-1 consumes action with pop(); capture its true packed batch first.
        self.local_batch = int(batch["action"].shape[0])

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if trainer.global_step >= 3:
            return
        partitions = trainer.strategy.model.optimizer.single_partition_of_fp32_groups
        bad = []
        for index, parameter in enumerate(partitions):
            if parameter.grad is None:
                continue
            count = sum(int((~torch.isfinite(chunk)).sum()) for chunk in
                        parameter.grad.detach().reshape(-1).split(32 * 1024 * 1024))
            if count:
                bad.append({"partition": index, "nonfinite_gradients": count})
        flag = torch.tensor(bool(bad), device=pl_module.device, dtype=torch.int32)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        print(f"XR1_GRADIENT_AUDIT rank={trainer.global_rank} step={trainer.global_step} bad={bad}", flush=True)
        if flag.item():
            raise RuntimeError(f"Non-finite gradients before optimizer update: {bad}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step <= self.last_step:
            return
        engine = trainer.strategy.model
        if engine.global_steps != trainer.global_step:
            raise RuntimeError("Lightning step differs from the actual DeepSpeed optimizer step")
        local_batch = self.local_batch
        if local_batch != engine.train_micro_batch_size_per_gpu():
            raise RuntimeError("Native token packing dropped samples; the declared global batch is no longer valid")
        self.last_step = trainer.global_step
        values = {name: float(value.detach().float().cpu()) if isinstance(value, torch.Tensor) else float(value)
                  for name, value in trainer.callback_metrics.items()}
        if isinstance(outputs, dict) and isinstance(outputs.get("loss"), torch.Tensor):
            values["backward_loss"] = float(outputs["loss"].detach().float().cpu())
        if "backward_loss" not in values or any(not math.isfinite(value) for value in values.values()):
            raise RuntimeError(f"Missing or non-finite native training metrics; refusing checkpoint publication: {values}")
        finite_at_save = None
        if trainer.global_step % self.save_interval == 0 or trainer.global_step == trainer.max_steps:
            with torch.no_grad():
                finite = torch.ones((), dtype=torch.bool, device=pl_module.device)
                for parameter in pl_module.parameters():
                    for chunk in parameter.detach().reshape(-1).split(32 * 1024 * 1024):
                        finite.logical_and_(torch.isfinite(chunk).all())
                finite_at_save = bool(finite)
            if not finite_at_save:
                raise RuntimeError("Non-finite model parameters; refusing checkpoint publication")
        if trainer.is_global_zero:
            now = time.time()
            record = {"step": trainer.global_step, "time": now,
                      "step_seconds": now - self.last_time if self.last_time is not None else None,
                      "global_batch": engine.train_batch_size(), "local_batch": local_batch, "metrics": values,
                      "parameters_finite_at_save": finite_at_save,
                      "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            self.last_time = now
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
            print("XR1_OPTIMIZER_STEP " + json.dumps(record, allow_nan=False), flush=True)
