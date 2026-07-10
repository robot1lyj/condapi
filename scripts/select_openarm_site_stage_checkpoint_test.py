import json

from scripts import select_openarm_site_stage_checkpoint as selector


def _write(path, **metrics):
    path.write_text(json.dumps(metrics))


def test_selects_lowest_site_mse_that_preserves_hq(tmp_path) -> None:
    reports = tmp_path / "reports"
    checkpoints = tmp_path / "checkpoints"
    reports.mkdir()
    for step in (1000, 2000):
        (checkpoints / str(step)).mkdir(parents=True)
    _write(reports / "site_1000.json", mse=0.012, mae=0.08, sign_accuracy=0.91, corrcoef=0.95, r2=0.88)
    _write(reports / "hq_1000.json", mse=0.013, mae=0.09, sign_accuracy=0.90, corrcoef=0.95, r2=0.87)
    _write(reports / "site_2000.json", mse=0.010, mae=0.07, sign_accuracy=0.93, corrcoef=0.96, r2=0.90)
    _write(reports / "hq_2000.json", mse=0.030, mae=0.12, sign_accuracy=0.80, corrcoef=0.80, r2=0.70)

    result = selector.select_checkpoint(reports, checkpoints, [1000, 2000])

    assert result["selected_step"] == 1000
    assert result["candidates"][0]["passed"] is True
    assert result["candidates"][1]["passed"] is False
