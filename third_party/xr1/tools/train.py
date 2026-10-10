# Copyright (C) 2026 Xiaomi Corporation.
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from mmengine import Config, DATASETS
from omegaconf import DictConfig
from mibot.models import MIMODEL

from mibot.utils.cfg_utils import helper
from mibot.utils.periodic_checkpoint import PeriodicModelCheckpoint
from mibot.utils.training_metrics import TrainingMetrics

import mibot.data

warnings.filterwarnings("ignore")


def prepare(cfg: Dict[str, Any]) -> Tuple[Config, LightningDataModule, LightningModule, List[Any]]:
    cfg: Config = helper(cfg)
    datamodule: LightningDataModule = DATASETS.build(cfg.data)
    model: LightningModule = MIMODEL.build(cfg.model)
    save_interval = cfg.trainer.pop("save_interval", 10000)
    keep_period = cfg.trainer.pop("keep_period", None)
    checkpoint = (
        PeriodicModelCheckpoint(cfg.trainer.default_root_dir, save_interval, keep_period)
        if keep_period is not None else ModelCheckpoint(
            save_top_k=-1, save_last=True, every_n_train_steps=save_interval,
            dirpath=cfg.trainer.default_root_dir, enable_version_counter=False,
        )
    )
    cfg.trainer["callbacks"] = [
        ModelSummary(max_depth=2),
        checkpoint,
    ]

    backend = cfg.trainer.pop("logger_backend", "wandb")
    project, experiment = cfg.trainer.pop("project"), cfg.trainer.pop("exp_name")
    if backend == "csv":
        logger = [CSVLogger(save_dir=cfg.trainer.default_root_dir, name="metrics", version="", flush_logs_every_n_steps=1)]
        cfg.trainer["callbacks"].insert(0, TrainingMetrics(cfg.trainer.default_root_dir, save_interval))
    elif backend == "wandb":
        logger = [WandbLogger(project=project, name=experiment, config=cfg)]
    else:
        raise ValueError(f"Unknown logger backend: {backend}")

    logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

    return (cfg, datamodule, model, logger)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg, datamodule, model, logger = prepare(cfg)
    ckpt_path: Optional[str] = cfg.trainer.pop("ckpt_path", None)
    trainer = Trainer(**cfg.trainer, logger=logger)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
