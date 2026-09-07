import json

from build_suite_report import attach_engine_next_steps
from build_suite_report import attach_exports
from build_suite_report import attach_front_runner


def test_highlight_is_measured_latency_not_accuracy_approval():
    data = {
        "facts": [{}, {}],
        "experiments": [
            {"name": "failed", "p50_ms": None},
            {"name": "slow", "p50_ms": 177, "p95_ms": 178, "max_abs_error": 0.02},
            {"name": "fast", "p50_ms": 99, "p95_ms": 101, "max_abs_error": 0.03},
        ],
    }
    attach_front_runner(data)
    assert data["latency_front_runner"]["name"] == "fast"
    assert data["latency_front_runner"]["accuracy_approved"] is False
    assert data["latency_front_runner"]["p95_ms"] == 101
    assert data["facts"][1]["value"] == "99.00 ms"


def test_export_preparation_pass_does_not_hide_failed_export(tmp_path):
    path = tmp_path / "export"
    path.mkdir()
    report = {
        "run_id": "export-r1",
        "status": "started",
        "compute_dtype": "bfloat16",
        "wrapper_comparisons": [{"exact": True, "finite": True}],
    }
    (path / "export_report.json").write_text(json.dumps(report))
    (tmp_path / "export-r1.exit.json").write_text(json.dumps({"exit_code": 1}))
    (tmp_path / "export-r1.manifest.json").write_text("{}")
    data = {"detail_tables": []}
    attach_exports(data, [path], tmp_path)
    assert "失败" in data["detail_tables"][0]["rows"][0][3]
    assert data["detail_tables"][0]["rows"][0][4] == "1/1 输入完全一致"
    assert "experiments" not in data

def test_engine_plan_uses_actual_result_without_accuracy_claim():
    data = {"measured_records": [], "recommendations": ["original"]}
    attach_engine_next_steps(data)
    assert data["recommendations"] == ["original"]
    data["measured_records"] = [{"backend": "tensorrt", "p50_ms": 127.37, "p95_ms": 127.99}]
    attach_engine_next_steps(data)
    assert "127.37" in data["recommendations"][0]["detail"]
    assert "不是任务精度保证" in data["recommendations"][1]["detail"]
