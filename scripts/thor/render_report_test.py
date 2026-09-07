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


def test_small_nonzero_precision_error_is_not_displayed_as_zero():
    page = render(
        {
            "updated_at": "test",
            "experiments": [{"name": "D", "weights": "FP32", "compute": "FP32", "max_abs_error": 0.0000152587890625}],
        }
    )
    assert "<td>1.52588e-05</td>" in page
    assert "<td>0.000</td>" not in page


def test_extra_sections_escape_cells_and_do_not_expand_data_placeholders():
    page = render(
        {
            "updated_at": "SOURCE",
            "experiments": [],
            "detail_tables": [{"title": "样本", "columns": ["维度"], "rows": [["<script>FACTS</script>"]]}],
            "sources": [{"title": "unsafe", "url": "javascript:alert(1)", "note": "bad"}],
        }
    )
    assert "更新于 SOURCE" in page
    assert "&lt;script&gt;FACTS&lt;/script&gt;" in page
    assert 'href="javascript:' not in page
