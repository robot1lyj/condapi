import shlex
import subprocess

from scripts import monitor_openarm_site_stage_scores as monitor


def test_ssh_argv_preserves_site_tmux_command() -> None:
    remote_args = ["bash", "-s", "--", "kai0_site_s4", "cd /shared/repo && bash scripts/score.sh >>log 2>&1"]

    argv = monitor._ssh_argv("gpu28", remote_args)  # noqa: SLF001

    assert argv[-2] == "gpu28"
    assert shlex.split(argv[-1]) == remote_args


def test_finalize_backfills_site_stage_episode_stats(monkeypatch, tmp_path) -> None:
    commands = []
    audit_root = tmp_path / "review"
    stage_data = tmp_path / "stage"
    checkpoint = tmp_path / "checkpoint"
    audit_root.mkdir()
    (audit_root / "site_stage_audit.json").write_text('{"passed": false, "metrics": {}}')
    (audit_root / "hq_stage_site_val10_pairs.json").write_text(
        '{"mse": 1, "mae": 1, "sign_accuracy": 0, "corrcoef": 0, "r2": 0}'
    )

    monkeypatch.setattr(monitor, "AUDIT_ROOT", audit_root)
    monkeypatch.setattr(monitor, "SITE_STAGE_DATA", stage_data)
    monkeypatch.setattr(monitor, "PAIR_EVAL_PATH", audit_root / "hq_stage_site_val10_pairs.json")
    monkeypatch.setattr(monitor, "DECISION_PATH", audit_root / "site_stage_decision.json")
    monkeypatch.setattr(monitor, "COMPLETE_MARKER", tmp_path / "complete")
    monkeypatch.setattr(monitor, "CHECKPOINT", checkpoint)

    def fake_ssh(_host, remote_args, *, input_text=None, timeout=30):
        commands.append((remote_args, input_text, timeout))
        return ""

    monkeypatch.setattr(monitor, "_ssh", fake_ssh)

    result = monitor._finalize_site_audit()  # noqa: SLF001

    audit_script = commands[0][1]
    assert "write_lerobot_episode_stats.py" in audit_script
    assert str(stage_data / "meta/episodes_stats.jsonl") in audit_script
    assert result["action"] == "train_site_stage"
    assert monitor.DECISION_PATH.exists()


def test_monitor_counts_failed_starts_and_stops_after_limit(monkeypatch) -> None:
    slot = {"host": "gpu28", "gpu_id": 0, "hq_shard": "hq", "hq_expected": 1}
    shard = {"shard_index": 0, "episode_count": 1, "total_frames": 10}
    state = {"jobs": {}}
    starts = []

    monkeypatch.setattr(monitor, "SLOTS", (slot,))
    monkeypatch.setattr(
        monitor,
        "_load_json",
        lambda path, default=None: (
            {"shards": [shard], "selected_episode_count": 1} if path == monitor.MANIFEST_PATH else state
        ),
    )
    monkeypatch.setattr(monitor, "_write_json_atomic", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(monitor, "_append_jsonl", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        monitor,
        "_remote_status",
        lambda *_args, **_kwargs: {"state": "ready", "completed": 0, "metadata_rows": 0, "hq_completed": 1},
    )

    def fail_start(*_args, **_kwargs):
        starts.append(1)
        raise subprocess.SubprocessError("launch failed")

    monkeypatch.setattr(monitor, "_start_job", fail_start)

    statuses = [monitor.monitor_once(max_restarts=3, stall_seconds=60, auto_start=True) for _ in range(4)]

    assert [status["jobs"][0]["state"] for status in statuses] == [
        "start_failed",
        "start_failed",
        "start_failed",
        "restart_exhausted",
    ]
    assert len(starts) == 3
    assert statuses[-1]["jobs"][0]["restarts"] == 3
