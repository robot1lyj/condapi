import numpy as np

from scripts import build_openarm_hq_score_report as report


def test_stage_aware_progress_offsets_folding_only_episodes() -> None:
    raw = np.asarray([0.0, 0.2, 0.4], dtype=np.float32)

    full_progress, full_stages, full_boundary = report._stage_aware_progress(raw, 535)  # noqa: SLF001
    folding_progress, folding_stages, folding_boundary = report._stage_aware_progress(raw, 536)  # noqa: SLF001

    np.testing.assert_allclose(full_progress, [0.0, 0.2, 0.4])
    assert full_stages.tolist() == [0, 0, 0]
    assert full_boundary is None
    np.testing.assert_allclose(folding_progress, [0.5, 0.7, 0.9])
    assert folding_stages.tolist() == [1, 1, 1]
    assert folding_boundary == 0
