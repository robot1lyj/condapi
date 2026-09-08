# ruff: noqa: RUF001
import copy
import json
from pathlib import Path

import pytest

from scripts.thor.render_report import number
from scripts.thor.render_report import render
from scripts.thor.render_report import select_configurations


@pytest.fixture
def data():
    return json.loads((Path(__file__).resolve().parents[2] / "docs/reports/thor/status.json").read_text())


def test_seven_meaningful_configurations_without_deleting_evidence(data):
    original = copy.deepcopy(data)
    selected = select_configurations(data)
    assert [row["mode"] for row in selected] == list("ABCDIVW")
    page = render(data)
    assert page.count('data-config="') == 7
    assert page.count("<table>") == 1
    assert "不是 7 个不同模型" in page
    assert "pi05-L-20260907-r1" not in page
    assert "pi05-F-20260907-r1" not in page
    assert 'href="status.json"' in page
    assert "<pre>" not in page
    assert "逐层分析已完成" not in page
    assert data == original


def test_labels_distinguish_storage_compute_and_actual_mixed_precision(data):
    selected = {row["mode"]: row for row in select_configurations(data)}
    assert selected["B"]["weights"] == "FP32"
    assert selected["B"]["compute"] == "BF16 为主"
    assert selected["C"]["weights"] == "BF16"
    for mode in "IVW":
        assert selected[mode]["weights"] == "BF16 + FP32"
        assert selected[mode]["compute"] == "BF16 + FP32"
    page = render(data)
    assert "不是原磁盘文件" in page
    assert "不是 FP16" in page
    assert "不是三种模型" in page


def test_values_come_from_matching_original_jax_comparison(data):
    selected = {row["mode"]: row for row in select_configurations(data)}
    comparison = next(c for c in data["additional_comparisons"] if c["candidate"] == selected["W"]["record"]["run_id"])
    assert selected["W"]["error"] == comparison["physical_dataset_units"]
    page = render(data)
    assert number(selected["W"]["record"]["p50_ms"], latency=True) in page
    assert number(selected["W"]["error"]["mae"]) in page
    assert number(selected["D"]["error"]["max_abs"]) in page
    assert "0.00001526" in page
    assert "0.00000109" in page


def test_missing_values_are_not_replaced_by_other_results(data):
    data["measured_records"] = [r for r in data["measured_records"] if not r["run_id"].startswith("pi05-W-")]
    page = render(data)
    assert "尚无当前候选的完整延迟数据" in page
    assert 'class="value">—' in page
    assert "104.25" not in page
    assert number(None) == "—"
    assert number(1e-12) != "0"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_metrics_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        number(value)


def test_report_escapes_content_and_never_reinterprets_placeholders(data):
    data["updated_at"] = "<script>{{P50}}</script>"
    page = render(data)
    assert "<script>" not in page
    assert "&lt;script&gt;{{P50}}&lt;/script&gt;" in page
    assert "https://" not in page


def test_precision_or_reference_mismatch_cannot_be_mislabeled(data):
    record = next(r for r in data["measured_records"] if r["run_id"].startswith("pi05-B-"))
    record["loaded_param_dtypes"] = {"bfloat16": 51}
    with pytest.raises(ValueError, match="Weight precision"):
        select_configurations(data)
    record["loaded_param_dtypes"] = {"float32": 51}
    data["additional_comparisons"][-1]["reference"] = "some-other-reference"
    with pytest.raises(ValueError, match="original JAX"):
        select_configurations(data)


def test_report_explains_metrics_and_does_not_claim_task_accuracy(data):
    page = render(data)
    for phrase in (
        "它不是平均值",
        "剩余约 5%",
        "取绝对值，再求平均",
        "0.50 与 0.51",
        "不是错误率或百分比",
        "尚未校准成弧度",
        "不是最坏耗时",
        "不能单独判定方案好坏",
        "误差 P95",
        "延迟 P95",
        "闭环执行成功率",
        "首次编译",
        "3588 相机采集或网线传输",
        "不把重复测速当成更多独立样本",
        "原严格诊断为 8/9",
        "尚无自动切换",
        "未微调",
    ):
        assert phrase in page


def test_duplicate_configuration_and_confounded_suite_are_rejected(data):
    data["measured_records"].append(copy.deepcopy(data["measured_records"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        render(data)
    data["measured_records"].pop()
    data["measured_records"][0]["suite_sha256"] = "different-suite"
    with pytest.raises(ValueError, match="same suite"):
        render(data)
