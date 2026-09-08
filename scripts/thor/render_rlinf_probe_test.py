import json
from pathlib import Path
import shutil

import pytest

from scripts.thor.render_rlinf_probe import render

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/reports/thor/evidence/20260908/rlinf"


@pytest.fixture
def status():
    return json.loads((ROOT / "docs/reports/thor/status.json").read_text())


def test_report_keeps_original_and_gelu_ablation_separate(status):
    page = render(EVIDENCE, status)
    assert page.count('<th scope="row">') == 6
    for text in ("0.000555", "0.00000111", "0.002428", "0.025485", "104.25", "667", "没有进行微调"):
        assert text in page
    assert "非完整 RLinf 训练框架" in page
    assert "不能用未优化的 RLinf 延迟断言框架的性能上限" in page
    assert "{{ROWS}}" not in page


def test_report_requires_same_workload(status):
    reference = next(r for r in status["measured_records"] if r["run_id"].startswith("pi05-A-"))
    reference["noise_sha256"] = "wrong"
    with pytest.raises(ValueError, match="mismatch"):
        render(EVIDENCE, status)


def test_failed_mapping_audit_cannot_support_weight_equivalence(tmp_path, status):
    copied = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, copied)
    path = copied / "pi05-rlinf-mapping-20260908-r1.json"
    record = json.loads(path.read_text())
    record["all_tensors_bit_equal"] = False
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Weight equivalence"):
        render(copied, status)
