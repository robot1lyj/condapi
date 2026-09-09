import json
from pathlib import Path
import subprocess
import threading
import urllib.error
import urllib.request

import pytest

from scripts import training_dashboard as dashboard


def test_waiting_and_partial_append(tmp_path):
    path = tmp_path / "metrics.jsonl"
    assert dashboard.read_metrics(path, 64)["rows"] == []
    path.write_text('{"step":0,"loss":0.8}\n{"step":1,"loss":', encoding="utf-8")
    assert len(dashboard.read_metrics(path, 64)["rows"]) == 1
    with path.open("a") as stream:
        stream.write("0.5}\n")
    assert dashboard.read_metrics(path, 64)["rows"][-1]["loss"] == 0.5


def test_invalid_and_nonfinite_values(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('broken\n[]\n{"step":0,"loss":NaN}\n{"step":1,"seconds":4}\n', encoding="utf-8")
    result = dashboard.read_metrics(path, 64)
    assert result["invalid_lines"] == 3
    assert "loss" not in result["rows"][0]
    assert result["rows"][1]["samples_per_second"] == 16
    json.dumps(result, allow_nan=False)


def test_resume_discards_abandoned_future(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text("".join(json.dumps({"step": step, "loss": i}) + "\n" for i, step in enumerate([0, 10, 20, 10, 15])))
    assert [r["step"] for r in dashboard.read_metrics(path, 64)["rows"]] == [0, 10, 15]
    path.write_text('{"step":0,"loss":99}\n')
    assert dashboard.read_metrics(path, 64)["rows"] == [{"step": 0, "loss": 99}]


@pytest.mark.parametrize(
    "extra",
    [
        ["--batch-size", "0"],
        ["--interval", "0"],
        ["--stage-steps", "200000", "--total-steps", "162097"],
        ["--remote", "-oProxyCommand=evil", "--remote-metrics", "/a"],
        ["--remote", "yam-server", "--remote-metrics", "/a;touch /b"],
        ["--remote", "yam-server"],
    ],
)
def test_bad_arguments_rejected(extra):
    with pytest.raises(SystemExit):
        dashboard.parse_args(["--metrics", "/tmp/test-dashboard.jsonl", *extra])


def test_size_and_history_limits(tmp_path, monkeypatch):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step":0}\n{"step":1}\n{"step":2}\n')
    monkeypatch.setattr(dashboard, "MAX_ROWS", 2)
    result = dashboard.read_metrics(path, 64)
    assert result["trimmed"]
    assert len(result["rows"]) == 2
    monkeypatch.setattr(dashboard, "MAX_BYTES", 4)
    with pytest.raises(ValueError, match="32 MiB"):
        dashboard.read_metrics(path, 64)


def test_http_live_refresh_and_no_directory_access(tmp_path):
    path = tmp_path / "metrics.jsonl"
    app = dashboard.Dashboard(dashboard.parse_args(["--metrics", str(path)]))
    server = dashboard.make_server(app, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(url + "/api/metrics") as response:
            assert json.load(response)["rows"] == []
        path.write_text('{"step":50,"loss":0.7}\n')
        with urllib.request.urlopen(url + "/api/metrics") as response:
            assert json.load(response)["rows"][0]["loss"] == 0.7
            assert response.headers["Cache-Control"] == "no-store"
        with urllib.request.urlopen(url) as response:
            assert b"<canvas" in response.read()
        with pytest.raises(urllib.error.HTTPError, match="404"):
            urllib.request.urlopen(url + "/../AGENTS.md")
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_failed_mirror_preserves_last_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step":1}\n')
    args = dashboard.parse_args(["--metrics", str(path), "--remote", "yam-server", "--remote-metrics", "/a"])
    app = dashboard.Dashboard(args)
    stop = threading.Event()

    def fail(*_args, **_kwargs):
        stop.set()
        raise subprocess.CalledProcessError(44, "ssh")

    monkeypatch.setattr(subprocess, "run", fail)
    app.mirror(stop)
    assert path.read_text() == '{"step":1}\n'
    assert "等待" in app.sync["error"]


def test_successful_mirror_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "metrics.jsonl"
    args = dashboard.parse_args(["--metrics", str(path), "--remote", "yam-server", "--remote-metrics", "/a"])
    app = dashboard.Dashboard(args)
    stop = threading.Event()

    def success(*_args, **_kwargs):
        stop.set()
        return subprocess.CompletedProcess([], 0, stdout=b'1788842923\n{"step":2,"loss":0.1}\n')

    monkeypatch.setattr(subprocess, "run", success)
    app.mirror(stop)
    assert app.snapshot()["rows"][0]["step"] == 2
    assert app.sync["source_modified_at"] == 1788842923
    assert not Path(str(path.with_suffix(".pending"))).exists()


def test_formats_and_custom_metrics(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step":2,"metrics":{"loss":0.125,"action":{"mse":0.002}}}\n{"step":2,"eval_loss":0.25}\n')
    row = dashboard.read_metrics(path)["rows"][0]
    assert row["loss"] == 0.125
    assert row["val_loss"] == 0.25
    assert row["action/mse"] == 0.002
    csv = tmp_path / "metrics.csv"
    csv.write_text("iteration,objective,time_ms\n2,0.125,123\n")
    row = dashboard.read_metrics(csv, field_map={"step": "iteration", "loss": "objective"})["rows"][0]
    assert row["loss"] == 0.125
    assert "step_seconds" not in row
    state = tmp_path / "trainer_state.json"
    state.write_text(json.dumps({"log_history": [{"step": 2, "loss": 0.125}, {"step": 2, "eval_loss": 0.25}]}))
    assert dashboard.read_metrics(state)["rows"][0]["val_loss"] == 0.25


def test_multirun_isolation_and_metadata(tmp_path):
    (tmp_path / "pi.jsonl").write_text('{"step":10,"loss":1}\n')
    (tmp_path / "custom.jsonl").write_text('{"global_step":3,"custom_score":0.95}\n')
    config = tmp_path / "runs.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": [
                    {"id": "pi", "metrics": "pi.jsonl", "model": "pi05"},
                    {"id": "custom", "metrics": "custom.jsonl", "model": "anything"},
                ],
            }
        )
    )
    app = dashboard.Dashboard(dashboard.parse_args(["--runs-config", str(config)]))
    assert app.snapshot("pi")["rows"][0]["step"] == 10
    assert app.snapshot("custom")["rows"][0]["custom_score"] == 0.95
    assert "batch_size" not in app.snapshot("custom")["run"]
    with pytest.raises(KeyError):
        app.snapshot("../../AGENTS.md")


def test_unknown_batch_does_not_invent_throughput(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step":0,"step_seconds":2}\n')
    assert "samples_per_second" not in dashboard.read_metrics(path)["rows"][0]
