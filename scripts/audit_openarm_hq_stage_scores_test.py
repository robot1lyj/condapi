import numpy as np

from scripts import audit_openarm_hq_stage_scores as audit


def test_hq_stage_quality_gate_distinguishes_full_and_folding_only() -> None:
    curves = []
    relative = np.linspace(-0.1, 0.1, 100, dtype=np.float32)
    for episode in range(4):
        peak = 0.9 if episode < 2 else 0.4
        curves.append((episode, np.linspace(0.0, peak, 100, dtype=np.float32), relative, relative))

    result = audit.audit_curves(curves, expected_episodes=4, folding_only_start=2)

    assert result["passed"] is True
    assert result["full_task"]["peak_crossing_fraction"] == 1.0
    assert result["folding_only"]["peak_above_0.20_fraction"] == 1.0


def test_hq_stage_quality_gate_rejects_collapsed_folding_scores() -> None:
    relative = np.linspace(-0.1, 0.1, 100, dtype=np.float32)
    curves = [
        (0, np.linspace(0.0, 0.9, 100, dtype=np.float32), relative, relative),
        (1, np.linspace(0.0, 0.9, 100, dtype=np.float32), relative, relative),
        (2, np.linspace(0.0, 0.05, 100, dtype=np.float32), relative, relative),
        (3, np.linspace(0.0, 0.05, 100, dtype=np.float32), relative, relative),
    ]

    result = audit.audit_curves(curves, expected_episodes=4, folding_only_start=2)

    assert result["passed"] is False
    assert result["gates"]["folding_only_peak_p10>=0.25"] is False
