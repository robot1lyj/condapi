# Copyright (C) 2026 Xiaomi Corporation.
from copy import deepcopy
from itertools import islice

from lightning import LightningDataModule
from mmengine import DATASETS
from mmengine import Config
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from mibot.data.collate.custom_collate import CustomCollate
from mibot.data.datasets.json_dataset import JsonDataset


class ResumeDistributedSampler(DistributedSampler):
    """Keep native seeded ordering, skip samples in committed optimizer steps."""

    def __init__(self, dataset, trainer, batch_size):
        super().__init__(dataset, shuffle=True, seed=42)
        self.trainer = trainer
        self.batch_size = batch_size

    def __iter__(self):
        accumulation = int(self.trainer.accumulate_grad_batches)
        samples_per_step = self.batch_size * accumulation
        if self.num_samples % samples_per_step:
            raise ValueError("Resumable XR-1 sampler requires complete optimizer batches")
        offset = (int(self.trainer.global_step) * samples_per_step) % self.num_samples
        print(f"XR1_SAMPLER rank={self.rank} epoch={self.epoch} step={self.trainer.global_step} skip_samples={offset}", flush=True)
        return islice(super().__iter__(), offset, None)


@DATASETS.register_module()
class BaseDataModule(LightningDataModule):
    def __init__(self, params: Config) -> None:
        super().__init__()
        self.params: Config = params
        self.batch_size: int = params.train_datasets.get("batch_size", 16)
        self.collate_fn = CustomCollate()
        self.train_set = None

    def setup(self, stage=None) -> None:
        if stage in (None, "fit") and self.train_set is None:
            self.train_set = JsonDataset(deepcopy(self.params))

    def train_dataloader(self) -> DataLoader:
        if self.train_set is None:
            self.setup("fit")
        sampler = (ResumeDistributedSampler(self.train_set, self.trainer, self.batch_size)
                   if self.params.get("resume_skip_samples", False)
                   else DistributedSampler(self.train_set, shuffle=True, seed=42))
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=8,
            prefetch_factor=4,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )
