import argparse
import json

from scripts import audit_openarm_kai0_pipeline_completion as completion


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _args(tmp_path):
    paths = {
        name: tmp_path / f"{name}.json"
        for name in (
            "hq_audit",
            "site_selection",
            "k_data_report",
            "k_data_audit",
            "norm_stats",
            "policy_selection",
            "deployment",
            "report_payload",
        )
    }
    return argparse.Namespace(
        **paths,
        checkpoint_root=tmp_path / "checkpoints",
        report_html=tmp_path / "report/index.html",
        output=tmp_path / "completion.json",
    )


def _complete_fixture(args):
    _write(args.hq_audit, {"passed": True, "expected_episodes": 999, "actual_episodes": 999})
    _write(
        args.site_selection,
        {
            "mode": "adapted_site_stage",
            "checkpoint": "/stage/4000",
            "adapted_audit": {"passed": True, "metrics": {"episode_count": 150}},
        },
    )
    _write(
        args.k_data_report,
        {
            "total_episodes": 1719,
            "unique_episode_counts": {"HQ": 999, "Site": 140, "TDA": 300},
            "materialized_episode_counts": {"HQ": 999, "Site": 420, "TDA": 300},
        },
    )
    _write(args.k_data_audit, {"passed": True})
    _write(args.norm_stats, {"norm_stats": {"state": {}}})
    for step in completion.EXPECTED_STEPS:
        (args.checkpoint_root / str(step) / "params").mkdir(parents=True)
        (args.checkpoint_root / str(step) / "_CHECKPOINT_METADATA").write_text("{}")
        (args.checkpoint_root / str(step) / "params/_METADATA").write_text("{}")
    selected = str(args.checkpoint_root / "5000")
    _write(
        args.policy_selection,
        {
            "selected_step": 5000,
            "selected_checkpoint": selected,
            "positive_prompt": completion.POSITIVE_PROMPT,
            "candidates": [{"step": step} for step in completion.EXPECTED_STEPS],
        },
    )
    smoke = {
        "passed": True,
        "checkpoint": selected,
        "prompt": completion.POSITIVE_PROMPT,
        "actions": {"shape": [50, 16]},
        "contract": {"passed": True},
        "elapsed_ms": 123.0,
    }
    _write(
        args.deployment,
        {
            "checkpoint": selected,
            "prompt": completion.POSITIVE_PROMPT,
            "host": "gpu25",
            "port": 6666,
            "inference_smoke": smoke,
        },
    )
    _write(
        args.report_payload, {"selection": {"selected_checkpoint": selected}, "deployment": {"checkpoint": selected}}
    )
    args.report_html.parent.mkdir(parents=True)
    args.report_html.write_text("<html></html>")


def test_completion_audit_requires_every_route_k_artifact(tmp_path) -> None:
    args = _args(tmp_path)
    _complete_fixture(args)

    result = completion.audit(args)

    assert result["passed"]
    assert all(result["gates"].values())


def test_completion_audit_rejects_smoke_for_wrong_checkpoint(tmp_path) -> None:
    args = _args(tmp_path)
    _complete_fixture(args)
    deployment = json.loads(args.deployment.read_text())
    deployment["inference_smoke"]["checkpoint"] = "/wrong/checkpoint"
    _write(args.deployment, deployment)

    result = completion.audit(args)

    assert not result["passed"]
    assert not result["gates"]["deployment_inference_smoke"]
