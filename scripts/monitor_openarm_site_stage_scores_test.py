import shlex

from scripts import monitor_openarm_site_stage_scores as monitor


def test_ssh_argv_preserves_site_tmux_command() -> None:
    remote_args = ["bash", "-s", "--", "kai0_site_s4", "cd /shared/repo && bash scripts/score.sh >>log 2>&1"]

    argv = monitor._ssh_argv("gpu28", remote_args)  # noqa: SLF001

    assert argv[-2] == "gpu28"
    assert shlex.split(argv[-1]) == remote_args
