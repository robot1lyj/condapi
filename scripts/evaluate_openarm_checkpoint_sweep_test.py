import numpy as np

from scripts import evaluate_openarm_checkpoint_sweep as sweep


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
