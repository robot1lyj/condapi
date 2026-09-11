"""Check report values against recorded evidence, without a model runtime."""

from pathlib import Path

from scripts.thor.render_realtime_vla import render


def test_report_uses_measured_values_and_excludes_invalid_run():
    root = Path(__file__).resolve().parents[2]
    report = render(root / "docs/reports/thor/evidence/20260911/realtime-vla")
    assert "124.16" in report
    assert "105.36" in report
    assert "0.005956" in report
    assert "0.002522" in report
    assert "0.166369" not in report
    assert report.count("<tr") == 6
    assert "realtime-vla-upstream-20260911-r1" in report
