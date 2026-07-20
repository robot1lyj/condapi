import argparse
import json

from scripts import build_openarm_kai0_policy_report as report


def _write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def test_builds_self_contained_kai0_policy_report(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text('{"step":0,"loss":1.0,"grad_norm":2.0}\n{"step":20,"loss":0.5,"grad_norm":1.0}\n')
    selection = tmp_path / "selection.json"
    selected_checkpoint = str(tmp_path / "checkpoints/5000")
    _write_json(
        selection,
        {
            "selected_step": 5000,
            "selected_checkpoint": selected_checkpoint,
            "candidates": [
                {
                    "step": 5000,
                    "rank_score": 0.0,
                    "site_mae": 0.1,
                    "site_critical_mae": 0.2,
                    "hq_mae": 0.1,
                    "hq_critical_mae": 0.2,
                    "hq_gap_ratio": 1.1,
                }
            ],
        },
    )
    deployment = tmp_path / "deployment.json"
    _write_json(
        deployment,
        {
            "checkpoint": selected_checkpoint,
            "host": "gpu25",
            "port": 6666,
            "prompt": "Fold the T-shirt properly, Advantage: positive",
            "inference_smoke": {
                "passed": True,
                "elapsed_ms": 1234,
                "actions": {"shape": [50, 16]},
                "contract": {"passed": True},
            },
        },
    )
    hq_audit = tmp_path / "hq_audit.json"
    _write_json(hq_audit, {"passed": True})
    site_selection = tmp_path / "site_selection.json"
    _write_json(site_selection, {"mode": "hq_stage_direct_transfer", "checkpoint": "/stage/10000"})
    k_data_report = tmp_path / "k_data.json"
    _write_json(
        k_data_report,
        {
            "total_episodes": 1719,
            "materialized_episode_counts": {"HQ": 999, "Site": 420, "TDA": 300},
            "materialized_label_counts": {
                kind: {
                    "0": {"positive": 10, "positive_ratio": 0.3},
                    "1": {"positive": 10, "positive_ratio": 0.3},
                }
                for kind in ("HQ", "Site", "TDA")
            },
        },
    )
    k_data_audit = tmp_path / "k_data_audit.json"
    _write_json(k_data_audit, {"passed": True})
    output = tmp_path / "policy_report/index.html"

    result = report.build_report(
        argparse.Namespace(
            metrics=metrics,
            selection=selection,
            deployment=deployment,
            hq_audit=hq_audit,
            site_selection=site_selection,
            k_data_report=k_data_report,
            k_data_audit=k_data_audit,
            output=output,
        )
    )

    html = output.read_text()
    assert result["selected_step"] == 5000
    assert "OpenArm K-Policy 训练报告" in html
    assert '"schema_version": "openarm_kai0_policy_report_v1"' in html
    assert "gpu25 inference smoke" in html
    assert "https://" not in html
    assert json.loads((output.parent / "report.json").read_text())["selection"]["selected_step"] == 5000
    assert not list(output.parent.glob(".index.html.tmp-*"))
