import json

import numpy as np
import pandas as pd

from scripts import audit_openarm_site_stage_scores as audit


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _info():
    features = {
        "observation.images.base": {"dtype": "video", "shape": [4, 6, 3]},
        "observation.images.left_wrist": {"dtype": "video", "shape": [4, 8, 3]},
        "observation.images.right_wrist": {"dtype": "video", "shape": [4, 8, 3]},
    }
    return {
        "total_episodes": 1,
        "chunks_size": 1000,
        "fps": 30,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def test_audit_site_scores_passes_exact_stage_predictions(tmp_path):
    source = tmp_path / "site"
    score_root = tmp_path / "scores"
    annotations = source / "annotations/openarm_stage_v1.jsonl"
    _write_json(source / "meta/info.json", _info())
    _write_jsonl(source / "meta/episodes.jsonl", [{"episode_index": 0, "length": 4}])
    _write_jsonl(
        annotations,
        [{"episode_index": 0, "quality": "success", "flatten_done": 1}],
    )
    for key in audit.VIDEO_KEYS:
        path = source / f"videos/chunk-000/{key}/episode_000000.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    _write_json(score_root / "meta/info.json", _info())
    _write_jsonl(
        score_root / "meta/episodes.jsonl",
        [{"episode_index": 0, "source_episode_index": 0, "length": 4}],
    )
    progress = np.asarray([0.0, 0.5, 0.5, 1.0], dtype=np.float32)
    relative = np.asarray([0.5, 0.5, 1.0, 0.0], dtype=np.float32)
    frame = pd.DataFrame(
        {
            "relative_advantage": relative,
            "absolute_value": progress,
            "absolute_advantage": relative,
        }
    )
    parquet = score_root / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    frame.to_parquet(parquet, index=False)

    result = audit.audit_site_scores(
        source,
        annotations,
        [score_root],
        tmp_path / "review",
        expected_episodes=1,
        relative_interval=2,
        overwrite=False,
    )

    assert result["passed"] is True
    assert result["metrics"]["absolute_mse"] == 0.0
    assert result["metrics"]["boundary_error_median"] == 0.0
    assert (tmp_path / "review/site_score_report/index.html").exists()


def test_adapted_profile_keeps_relative_and_boundary_metrics_diagnostic_only():
    metrics = {
        "absolute_mse": 0.0155,
        "absolute_mae": 0.0830,
        "relative_mse": 0.0089,
        "relative_mae": 0.0543,
        "direction_accuracy": 0.668,
        "corrcoef": 0.908,
        "r2": 0.811,
        "boundary_error_median": 0.271,
        "boundary_error_p90": 0.498,
    }

    assert all(audit._quality_gates(metrics, "adapted").values())  # noqa: SLF001
    assert not all(audit._quality_gates(metrics, "direct").values())  # noqa: SLF001
