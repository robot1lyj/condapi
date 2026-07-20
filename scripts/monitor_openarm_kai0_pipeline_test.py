import shlex

import pytest

from scripts import monitor_openarm_kai0_pipeline as pipeline


def test_ssh_argv_preserves_remote_shell_argument_boundaries() -> None:
    remote_args = ["bash", "-s", "--", "session", "cd /shared/repo && echo 'ready now'"]

    argv = pipeline._ssh_argv("gpu28", remote_args)  # noqa: SLF001

    assert argv[-2] == "gpu28"
    assert shlex.split(argv[-1]) == remote_args


def _report(mae: float, critical: float, overlap: float, gap: float = 1.2) -> dict:
    return {
        "schema_version": pipeline.SWEEP_SCHEMA_VERSION,
        "sampling": {"prompt_override": pipeline.POSITIVE_PROMPT},
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


def test_policy_sweep_selects_every_openpi_5k_checkpoint() -> None:
    candidates = [*range(5_000, 80_000, 5_000), 79_999]
    selected = [step for step in candidates if pipeline._is_policy_sweep_step(step)]  # noqa: SLF001

    assert selected == candidates
    assert not pipeline._is_policy_sweep_step(4_999)  # noqa: SLF001


def test_checkpoint_ready_requires_orbax_completion_metadata(tmp_path) -> None:
    checkpoint = tmp_path / "79999"
    (checkpoint / "params").mkdir(parents=True)

    assert not pipeline._checkpoint_ready(checkpoint)  # noqa: SLF001

    (checkpoint / "_CHECKPOINT_METADATA").write_text("{}")
    (checkpoint / "params/_METADATA").write_text("{}")

    assert pipeline._checkpoint_ready(checkpoint)  # noqa: SLF001


def test_policy_selection_rejects_non_positive_sweep_prompt(tmp_path) -> None:
    hq = {5_000: _report(0.1, 0.1, 0.1)}
    site_report = _report(0.1, 0.1, 0.1)
    site_report["sampling"]["prompt_override"] = "Fold the T-shirt properly"

    with pytest.raises(RuntimeError, match="positive AWBC prompt"):
        pipeline._select_policy_reports(hq, {5_000: site_report}, [5_000], tmp_path)  # noqa: SLF001


def test_deployment_evidence_contract_binds_checkpoint_prompt_and_action_shape() -> None:
    checkpoint = "/checkpoints/5000"
    smoke = {
        "passed": True,
        "checkpoint": checkpoint,
        "prompt": pipeline.POSITIVE_PROMPT,
        "actions": {"shape": [50, 16]},
        "contract": {"passed": True},
    }
    deployment = {"checkpoint": checkpoint, "prompt": pipeline.POSITIVE_PROMPT}

    assert pipeline._deployment_evidence_current(deployment, smoke, checkpoint)  # noqa: SLF001
    smoke["actions"]["shape"] = [50, 32]
    assert not pipeline._deployment_evidence_current(deployment, smoke, checkpoint)  # noqa: SLF001
    assert not pipeline._deployment_evidence_current(None, None, checkpoint)  # noqa: SLF001
