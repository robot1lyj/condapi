import argparse

import numpy as np
import pandas as pd

from scripts import evaluate_openarm_checkpoint_sweep as sweep


def test_sampling_signature_versions_critical_selector() -> None:
    args = argparse.Namespace(
        uniform_frames=3,
        critical_frames=3,
        include_adjacent=True,
        train_max_episodes=20,
        val_max_episodes=10,
        prompt="Fold the T-shirt properly, Advantage: positive",
    )

    assert sweep._sampling_signature(args)["critical_selector"] == sweep.CRITICAL_SELECTOR_VERSION  # noqa: SLF001


def test_subsample_decodable_episodes_replaces_invalid_initial_sample(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sweep, "_subsample_episodes", lambda *_args, **_kwargs: [0, 1, 2])
    monkeypatch.setattr(
        sweep,
        "_episode_video_error",
        lambda _dataset, episode: "broken video" if episode == 1 else None,
    )

    selected, rejected = sweep._subsample_decodable_episodes(  # noqa: SLF001
        tmp_path,
        list(range(6)),
        3,
        seed=7,
    )

    assert len(selected) == 3
    assert 1 not in selected
    assert rejected == [{"episode_index": 1, "reason": "broken video"}]


def test_prompt_override_for_awbc_evaluation() -> None:
    item = {
        "observation.state": np.zeros(16, dtype=np.float32),
        "task_index": np.asarray(0, dtype=np.int64),
    }

    default = sweep._build_observation(item, {0: "Fold the T-shirt properly"}, None)  # noqa: SLF001
    positive = sweep._build_observation(  # noqa: SLF001
        item,
        {0: "Fold the T-shirt properly"},
        "Fold the T-shirt properly, Advantage: positive",
    )

    assert default["prompt"] == "Fold the T-shirt properly"
    assert positive["prompt"] == "Fold the T-shirt properly, Advantage: positive"


def test_cached_report_requires_matching_positive_prompt(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoints/5000"
    dataset = tmp_path / "dataset"
    report_path = tmp_path / "checkpoint_5000.json"
    checkpoint.mkdir(parents=True)
    dataset.mkdir()
    prompt = "Fold the T-shirt properly, Advantage: positive"
    report = {
        "schema_version": sweep.REPORT_SCHEMA_VERSION,
        "checkpoint": str(checkpoint),
        "step": 5000,
        "config": "pi05_openarm_kai0_awbc_v1",
        "dataset": str(dataset),
        "train_episodes": [0, 1],
        "val_episodes": [2],
        "sampling": {"prompt_override": prompt},
    }
    sweep._write_json_atomic(report_path, report)  # noqa: SLF001

    cached = sweep._load_cached_report(  # noqa: SLF001
        report_path,
        checkpoint_dir=checkpoint,
        config_name="pi05_openarm_kai0_awbc_v1",
        dataset_dir=dataset,
        train_episodes=[0, 1],
        val_episodes=[2],
        sampling={"prompt_override": prompt},
    )
    wrong_prompt = sweep._load_cached_report(  # noqa: SLF001
        report_path,
        checkpoint_dir=checkpoint,
        config_name="pi05_openarm_kai0_awbc_v1",
        dataset_dir=dataset,
        train_episodes=[0, 1],
        val_episodes=[2],
        sampling={"prompt_override": "Fold the T-shirt properly"},
    )

    assert cached == report
    assert wrong_prompt is None


def test_critical_offsets_follow_motion_not_absolute_joint_pose(monkeypatch, tmp_path) -> None:
    actions = np.zeros((20, 16), dtype=np.float32)
    actions[:, 0] = 100.0
    actions[10:, 1] = 25.0
    states = np.zeros((20, 16), dtype=np.float32)
    frame = pd.DataFrame({"action": list(actions), "observation.state": list(states)})
    monkeypatch.setattr(sweep, "_episode_parquet_path", lambda *_args: tmp_path / "episode.parquet")
    monkeypatch.setattr(sweep.pd, "read_parquet", lambda *_args, **_kwargs: frame)

    offsets = sweep._critical_offsets(  # noqa: SLF001
        tmp_path,
        0,
        count=1,
        min_separation=1,
        max_length=20,
    )

    assert offsets == [10]


def test_critical_offsets_include_commanded_gripper_change(monkeypatch, tmp_path) -> None:
    actions = np.zeros((20, 16), dtype=np.float32)
    actions[7:, 7] = -66.0
    states = np.zeros((20, 16), dtype=np.float32)
    frame = pd.DataFrame({"action": list(actions), "observation.state": list(states)})
    monkeypatch.setattr(sweep, "_episode_parquet_path", lambda *_args: tmp_path / "episode.parquet")
    monkeypatch.setattr(sweep.pd, "read_parquet", lambda *_args, **_kwargs: frame)

    offsets = sweep._critical_offsets(tmp_path, 0, count=1, min_separation=1, max_length=20)  # noqa: SLF001

    assert offsets == [7]
