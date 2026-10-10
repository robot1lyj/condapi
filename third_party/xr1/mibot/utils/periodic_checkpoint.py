"""Condapi retention around Lightning's native full-state checkpoint callback."""

import os
from pathlib import Path
import uuid

from lightning.pytorch.callbacks import ModelCheckpoint

from mibot.utils.checkpoint_contract import checkpoint_step
from mibot.utils.checkpoint_contract import commit_checkpoint
from mibot.utils.checkpoint_contract import receipt_path
from mibot.utils.checkpoint_contract import validate_checkpoint


class PeriodicModelCheckpoint(ModelCheckpoint):
    def __init__(self, dirpath, save_interval, keep_period):
        if (type(save_interval) is not int or type(keep_period) is not int
                or save_interval < 1 or keep_period < 1 or keep_period % save_interval):
            raise ValueError("keep_period must be a positive multiple of save_interval")
        self.keep_period = keep_period
        super().__init__(
            dirpath=dirpath, filename="step_{step:09d}", auto_insert_metric_name=False,
            every_n_train_steps=save_interval, save_on_train_epoch_end=False,
            save_top_k=1, save_last="link", save_weights_only=False, enable_version_counter=False,
        )

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        trainer.strategy.barrier()
        error = None
        if trainer.is_global_zero:
            try:
                commit_checkpoint(filepath, trainer.global_step, trainer.world_size)
            except Exception as exc:
                error = str(exc)
        error = trainer.strategy.broadcast(error)
        if error is not None:
            raise RuntimeError(f"Checkpoint commit failed; previous recovery point retained: {error}")
        # Native top-k pruning runs before save_last. Publish the new committed
        # link first so a crash during pruning cannot leave last.ckpt dangling.
        self._link_checkpoint(trainer, filepath, str(Path(self.dirpath) / "last.ckpt"))

    def _should_remove_checkpoint(self, trainer, previous, current):
        previous, current = Path(previous).resolve(), Path(current).resolve()
        step = checkpoint_step(previous)
        if previous == current or previous.parent != Path(self.dirpath).resolve() or step is None:
            return False
        if step % self.keep_period == 0:
            return False
        # A failed/partial replacement must never retire the previous recovery point.
        error = None
        if trainer.is_global_zero:
            try:
                validate_checkpoint(current, trainer.world_size)
                validate_checkpoint(previous, trainer.world_size)
            except Exception as exc:
                error = str(exc)
        error = trainer.strategy.broadcast(error)
        if error is not None:
            raise RuntimeError(f"Refusing checkpoint pruning: {error}")
        return True

    def _remove_checkpoint(self, trainer, filepath):
        trainer.strategy.barrier()
        error = None
        try:
            super()._remove_checkpoint(trainer, filepath)
            if trainer.is_global_zero:
                receipt_path(filepath).unlink(missing_ok=True)
        except Exception as exc:
            error = str(exc)
        error = trainer.strategy.broadcast(error)
        if error is not None:
            raise RuntimeError(f"Checkpoint pruning failed; new recovery point is committed: {error}")

    @staticmethod
    def _link_checkpoint(trainer, filepath, linkpath):
        error = None
        if trainer.is_global_zero:
            try:
                link = Path(linkpath)
                if link.exists() and not link.is_symlink():
                    raise ValueError("Refusing to replace a non-link last.ckpt")
                temporary = link.with_name(f".{link.name}.{uuid.uuid4().hex}")
                os.symlink(os.path.relpath(filepath, link.parent), temporary)
                os.replace(temporary, link)
            except Exception as exc:
                error = str(exc)
        error = trainer.strategy.broadcast(error)
        if error is not None:
            raise RuntimeError(f"Checkpoint recovery link failed: {error}")

    def on_train_end(self, trainer, pl_module):
        # Native on_train_end does not save a non-interval final step after an
        # earlier checkpoint. Always commit that final full state before pruning.
        if not trainer.fast_dev_run and trainer.global_step > 0 and self._last_global_step_saved != trainer.global_step:
            candidates = self._monitor_candidates(trainer)
            self._save_topk_checkpoint(trainer, candidates)
            self._save_last_checkpoint(trainer, candidates)
        else:
            super().on_train_end(trainer, pl_module)
