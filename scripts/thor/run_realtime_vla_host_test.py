"""Lightweight command-contract tests; no model loading or local training."""

from pathlib import Path

import pytest

from scripts.thor.run_realtime_vla_host import docker_command


def test_isolation_and_original_defaults():
    command = docker_command(Path("/data/pi"), Path("/probe"), image="pi:test", label="replay-1")
    assert "/data/pi:/pi:ro" in command
    assert "/data/pi/results:/pi/results" in command
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--position-offset") + 1] == "-1"
    assert command[command.index("--text-capacity") + 1] == "200"
    assert "--fix-time-broadcast" not in command


def test_diagnostics_are_explicit():
    command = docker_command(
        Path("/data/pi"),
        Path("/probe"),
        image="pi:test",
        label="replay-2",
        position_offset=0,
        fix_time_broadcast=True,
        text_capacity=80,
    )
    assert command[command.index("--position-offset") + 1] == "0"
    assert "--fix-time-broadcast" in command
    assert command[command.index("--text-capacity") + 1] == "80"


@pytest.mark.parametrize("label", ["../old", "x/y", "x;rm", ""])
def test_reject_unsafe_run_label(label):
    with pytest.raises(ValueError, match="Unsafe run label"):
        docker_command(Path("/data/pi"), Path("/probe"), image="pi:test", label=label)
