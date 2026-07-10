from scripts import monitor_openarm_kai0_pipeline as pipeline


def _report(mae: float, critical: float, overlap: float, gap: float = 1.2) -> dict:
    return {
        "val": {
            "overall": {
                "mae": mae,
                "overlap_consistency_mae": overlap,
                "gripper_mae": mae,
            },
            "by_kind": {"critical": {"mae": critical}},
        },
        "gaps": {"mae_val_over_train": gap},
    }


def test_policy_selection_prefers_site_and_hq_joint_rank(tmp_path) -> None:
    hq = {
        5000: _report(0.10, 0.10, 0.10),
        10000: _report(0.08, 0.08, 0.08),
    }
    site = {
        5000: _report(0.05, 0.05, 0.05),
        10000: _report(0.09, 0.09, 0.09),
    }

    result = pipeline._select_policy_reports(hq, site, [5000, 10000], tmp_path)  # noqa: SLF001

    assert result["selected_step"] == 5000
    assert result["positive_prompt"] == "Fold the T-shirt properly, Advantage: positive"
