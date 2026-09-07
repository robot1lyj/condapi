from scripts.thor.render_report import render


def test_missing_measurements_are_not_fabricated():
    page = render({"updated_at": "2026-09-07", "experiments": [{"name": "A", "weights": "FP32", "compute": "FP32"}]})
    assert "待测试" in page
    assert "<td>—</td>" in page
    assert "0.000" not in page


def test_report_escapes_untrusted_labels():
    page = render({"updated_at": "<script>alert(1)</script>", "experiments": []})
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_report_formats_real_measurements():
    page = render(
        {
            "updated_at": "2026-09-07",
            "experiments": [{"name": "A", "weights": "FP32", "compute": "FP32", "p50_ms": 12.5}],
        }
    )
    assert "<td>12.500</td>" in page
