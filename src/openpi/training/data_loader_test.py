import dataclasses

import jax
import numpy as np

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class _IndexDataset:
    def __init__(self, size: int):
        self._size = size

    def __getitem__(self, index):
        return {"index": np.asarray(index, dtype=np.int64)}

    def __len__(self):
        return self._size


class _ShortVideoDataset:
    def __init__(self, *, error: str = "Requested next frame while there are no more frames left to decode"):
        self.video_backend = "torchcodec"
        self.error = error
        self.calls = []

    def __getitem__(self, index):
        self.calls.append((index, self.video_backend))
        if self.video_backend == "torchcodec":
            raise RuntimeError(self.error)
        return {"index": index}

    def __len__(self):
        return 1


def test_torchcodec_tail_fallback_retries_only_end_of_stream_with_pyav():
    dataset = _ShortVideoDataset()
    wrapped = _data_loader.TorchCodecTailFallbackDataset(dataset)

    assert wrapped[7] == {"index": 7}
    assert dataset.calls == [(7, "torchcodec"), (7, "pyav")]
    assert dataset.video_backend == "torchcodec"


def test_torchcodec_tail_fallback_does_not_hide_other_decode_failures():
    dataset = _ShortVideoDataset(error="corrupt video stream")
    wrapped = _data_loader.TorchCodecTailFallbackDataset(dataset)

    with np.testing.assert_raises_regex(RuntimeError, "corrupt video stream"):
        wrapped[3]
    assert dataset.calls == [(3, "torchcodec")]
    assert dataset.video_backend == "torchcodec"


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_resume_position_and_epoch_shuffle():
    dataset = _IndexDataset(10)
    sampler = _data_loader.EpochRandomSampler(dataset, seed=7)
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=2,
        sampler=sampler,
        num_batches=4,
        framework="pytorch",
    )
    loader.set_start_step(2)

    batches = [batch["index"].numpy().tolist() for batch in loader]
    epoch0 = list(_data_loader.EpochRandomSampler(dataset, seed=7))
    epoch1_sampler = _data_loader.EpochRandomSampler(dataset, seed=7)
    epoch1_sampler.set_epoch(1)
    epoch1 = list(epoch1_sampler)

    assert batches == [epoch0[4:6], epoch0[6:8], epoch0[8:10], epoch1[0:2]]
    assert epoch0 != epoch1


def test_distributed_sampler_resume_is_rank_disjoint():
    dataset = _IndexDataset(12)
    rank0 = _data_loader.ResumableDistributedSampler(
        dataset, num_replicas=2, rank=0, shuffle=True, seed=11, drop_last=True
    )
    rank1 = _data_loader.ResumableDistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=True, seed=11, drop_last=True
    )
    full_rank0 = list(rank0)
    full_rank1 = list(rank1)
    assert set(full_rank0).isdisjoint(full_rank1)
    assert set(full_rank0) | set(full_rank1) == set(range(12))

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=2,
        sampler=rank0,
        num_batches=2,
        framework="pytorch",
    )
    loader.set_start_step(2)
    batches = [batch["index"].numpy().tolist() for batch in loader]
    epoch1 = _data_loader.ResumableDistributedSampler(
        dataset, num_replicas=2, rank=0, shuffle=True, seed=11, drop_last=True
    )
    epoch1.set_epoch(1)

    assert batches == [full_rank0[4:6], list(epoch1)[0:2]]


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
