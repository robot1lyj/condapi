import json

import pandas as pd
import pytest

from scripts import audit_openarm_kai0_training_data as audit


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _data_path(root, info, episode_index):
    return root / info["data_path"].format(episode_chunk=0, episode_index=episode_index)


def test_audit_dataset_structure_accepts_binary_k_data(tmp_path):
    info = {
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "total_episodes": 2,
        "total_frames": 4,
    }
    _write_json(tmp_path / "meta/info.json", info)
    _write_jsonl(tmp_path / "meta/tasks.jsonl", audit.TASKS)
    _write_jsonl(
        tmp_path / "meta/episodes.jsonl",
        [
            {"episode_index": 0, "source_kind": "HQ"},
            {"episode_index": 1, "source_kind": "Site"},
        ],
    )
    _write_jsonl(
        tmp_path / "meta/episodes_stats.jsonl",
        [{"episode_index": 0, "stats": {}}, {"episode_index": 1, "stats": {}}],
    )
    _write_json(
        tmp_path / "kai0_awbc_build_report.json",
        {"total_episodes": 2, "tasks": list(audit.TASKS), "advantage_source": "absolute_advantage"},
    )
    for episode_index, labels in enumerate(((0, 1), (1, 0))):
        path = _data_path(tmp_path, info, episode_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "index": [episode_index * 2, episode_index * 2 + 1],
                "episode_index": [episode_index, episode_index],
                "frame_index": [0, 1],
                "task_index": labels,
            }
        ).to_parquet(path, index=False)

    result = audit.audit_dataset_structure(
        tmp_path,
        expected_episodes=2,
        expected_source_counts={"HQ": 1, "Site": 1},
    )

    assert result["label_counts"] == {"0": 2, "1": 2}
    assert result["selected_global_indices"] == {"0": 0, "1": 1}
    assert result["positive_ratio"] == 0.5


def test_audit_dataset_structure_rejects_non_binary_label(tmp_path):
    info = {
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "total_episodes": 1,
        "total_frames": 2,
    }
    _write_json(tmp_path / "meta/info.json", info)
    _write_jsonl(tmp_path / "meta/tasks.jsonl", audit.TASKS)
    _write_jsonl(tmp_path / "meta/episodes.jsonl", [{"episode_index": 0, "source_kind": "HQ"}])
    _write_jsonl(tmp_path / "meta/episodes_stats.jsonl", [{"episode_index": 0, "stats": {}}])
    _write_json(
        tmp_path / "kai0_awbc_build_report.json",
        {"total_episodes": 1, "tasks": list(audit.TASKS), "advantage_source": "absolute_advantage"},
    )
    path = _data_path(tmp_path, info, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": [0, 1], "episode_index": [0, 0], "frame_index": [0, 1], "task_index": [0, 2]}).to_parquet(
        path, index=False
    )

    with pytest.raises(ValueError, match="non-binary"):
        audit.audit_dataset_structure(tmp_path, expected_episodes=1, expected_source_counts={"HQ": 1})


def test_audit_dataset_structure_rejects_relative_advantage_contract(tmp_path):
    _write_json(
        tmp_path / "meta/info.json",
        {
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "total_episodes": 1,
            "total_frames": 2,
        },
    )
    _write_jsonl(tmp_path / "meta/tasks.jsonl", audit.TASKS)
    _write_jsonl(tmp_path / "meta/episodes.jsonl", [{"episode_index": 0, "source_kind": "HQ"}])
    _write_jsonl(tmp_path / "meta/episodes_stats.jsonl", [{"episode_index": 0, "stats": {}}])
    _write_json(
        tmp_path / "kai0_awbc_build_report.json",
        {"total_episodes": 1, "tasks": list(audit.TASKS), "advantage_source": "relative_advantage"},
    )

    with pytest.raises(ValueError, match="absolute_advantage"):
        audit.audit_dataset_structure(tmp_path, expected_episodes=1, expected_source_counts={"HQ": 1})
