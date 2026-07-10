import json

import pytest

from scripts import openarm_advantage_report as report


def _payload() -> dict:
    return {
        "episode_index": 7,
        "length": 3,
        "fps": 30,
        "duration_s": 2 / 30,
        "quality": "predicted",
        "eligible_for_k_data": True,
        "flatten_done_frame": 2,
        "source_dataset": "HQ",
        "source_episode_index": 7,
        "progress": [0.0, 0.3, 0.6],
        "stage_id": [0, 0, 1],
        "advantage": [0.2, -0.1, 0.0],
        "advantage_min": -0.1,
        "advantage_mean": 0.033,
        "advantage_max": 0.2,
        "negative_fraction": 1 / 3,
        "videos": {"base": "../videos/base/episode_000007.mp4"},
        "score_source": "HQ-Stage",
    }


def test_write_stage_report_renders_reusable_viewer(tmp_path):
    output_dir = tmp_path / "report"
    index_path = report.write_stage_report(
        output_dir,
        [_payload()],
        {
            "completed_episodes": 1,
            "completed_frames": 3,
            "negative_frame_fraction": 1 / 3,
            "generated_at": "2026-07-10T00:00:00+08:00",
        },
        report.StageReportConfig(
            title="Stage report",
            subtitle="Test report",
            score_source="Test-Stage",
        ),
    )

    page = index_path.read_text()
    payload = json.loads((output_dir / "data/episode_000007.json").read_text())
    assert "Stage report" in page
    assert "video-aligned" not in page.lower()
    assert "__REPORT_" not in page
    assert 'data-signal="neutral"' in page
    assert payload["episode_index"] == 7
    assert payload["advantage"] == [0.2, -0.1, 0.0]


def test_write_stage_report_rejects_misaligned_series(tmp_path):
    payload = _payload()
    payload["advantage"] = [0.1]
    with pytest.raises(ValueError, match="advantage length"):
        report.write_stage_report(
            tmp_path / "report",
            [payload],
            {},
            report.StageReportConfig(title="Invalid", subtitle="Invalid", score_source="test"),
        )
