import shlex

import pytest

from scripts import launch_openarm_jax_multinode as launcher


def test_ssh_argv_preserves_multiline_tmux_command() -> None:
    remote_args = ["bash", "-s", "--", "session", "cd /shared/repo\npython scripts/train.py"]

    argv = launcher._ssh_argv("gpu12", remote_args)  # noqa: SLF001

    assert argv[-2] == "gpu12"
    assert shlex.split(argv[-1]) == remote_args


def test_builds_two_process_four_gpu_command() -> None:
    report = launcher.launch(
        config="pi05_openarm_kai0_awbc_v1",
        exp_name="smoke",
        num_train_steps=20,
        batch_size=128,
        num_workers=0,
        log_interval=1,
        mode="overwrite",
        session_prefix="kai0_smoke",
        coordinator_address="172.31.11.112:12365",
        xla_memory_fraction=0.9,
        dry_run=True,
    )

    assert report["global_batch_size"] == 128
    assert [job["host"] for job in report["jobs"]] == ["gpu12", "gpu14"]
    assert "JAX_PROCESS_ID=0" in report["jobs"][0]["command"]
    assert "JAX_PROCESS_ID=1" in report["jobs"][1]["command"]
    assert "JAX_COORDINATOR_BIND_ADDRESS=0.0.0.0:12365" in report["jobs"][0]["command"]
    assert "--overwrite" in report["jobs"][0]["command"]


def test_rejects_batch_not_divisible_by_global_device_count() -> None:
    with pytest.raises(ValueError, match="divisible by four"):
        launcher.launch(
            config="config",
            exp_name="bad",
            num_train_steps=1,
            batch_size=126,
            num_workers=0,
            log_interval=1,
            mode="overwrite",
            session_prefix="bad",
            coordinator_address="172.31.11.112:12365",
            xla_memory_fraction=0.9,
            dry_run=True,
        )
