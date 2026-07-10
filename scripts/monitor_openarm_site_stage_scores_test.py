import shlex
import subprocess

from scripts import monitor_openarm_site_stage_scores as monitor


def test_ssh_argv_preserves_site_tmux_command() -> None:
    remote_args = ["bash", "-s", "--", "kai0_site_s4", "cd /shared/repo && bash scripts/score.sh >>log 2>&1"]

    argv = monitor._ssh_argv("gpu28", remote_args)  # noqa: SLF001

    assert argv[-2] == "gpu28"
    assert shlex.split(argv[-1]) == remote_args


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
