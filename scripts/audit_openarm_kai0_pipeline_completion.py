"""Audit every required artifact of the completed OpenArm KAI0 K route."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from typing import Any

POSITIVE_PROMPT = "Fold the T-shirt properly, Advantage: positive"
EXPECTED_STEPS = [*range(5_000, 80_000, 5_000), 79_999]


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _checkpoint_ready(path: pathlib.Path) -> bool:
    return (path / "_CHECKPOINT_METADATA").is_file() and (path / "params/_METADATA").is_file()


def _site_evidence(site: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if site.get("mode") == "adapted_site_stage":
        evidence = site.get("adapted_audit", {})
        metrics = evidence.get("metrics", {})
        return evidence.get("passed") is True and metrics.get("episode_count") == 150, evidence
    decision = site.get("direct_decision", {})
    metrics = decision.get("curve_metrics", {})
    return decision.get("direct_transfer_passed") is True and metrics.get("episode_count") == 150, decision


def audit(args: argparse.Namespace) -> dict[str, Any]:
    hq = _load_json(args.hq_audit)
    site = _load_json(args.site_selection)
    k_report = _load_json(args.k_data_report)
    k_audit = _load_json(args.k_data_audit)
    selection = _load_json(args.policy_selection)
    deployment = _load_json(args.deployment)
    report_payload = _load_json(args.report_payload)
    norm_stats = _load_json(args.norm_stats)

    site_passed, site_evidence = _site_evidence(site)
    unique_counts = k_report.get("unique_episode_counts", {})
    materialized_counts = k_report.get("materialized_episode_counts", {})
    selected_checkpoint = selection.get("selected_checkpoint")
    selected_step = selection.get("selected_step")
    candidates = selection.get("candidates", [])
    candidate_steps = sorted(row.get("step") for row in candidates if isinstance(row, dict) and "step" in row)
    smoke = deployment.get("inference_smoke", {})

    gates = {
        "hq999_stage": hq.get("passed") is True
        and hq.get("expected_episodes") == 999
        and hq.get("actual_episodes") == 999,
        "site150_stage": site_passed,
        "k_data_build": k_report.get("total_episodes") == 1719
        and unique_counts == {"HQ": 999, "Site": 140, "TDA": 300}
        and materialized_counts == {"HQ": 999, "Site": 420, "TDA": 300},
        "k_data_norm": bool(norm_stats),
        "k_data_loader_audit": k_audit.get("passed") is True,
        "final_checkpoint_79999": _checkpoint_ready(args.checkpoint_root / "79999"),
        "all_sweep_checkpoints_ready": all(
            _checkpoint_ready(args.checkpoint_root / str(step)) for step in EXPECTED_STEPS
        ),
        "selection_has_16_candidates": candidate_steps == EXPECTED_STEPS,
        "selection_is_candidate": selected_step in EXPECTED_STEPS
        and selected_checkpoint == str(args.checkpoint_root / str(selected_step)),
        "selection_positive_prompt": selection.get("positive_prompt") == POSITIVE_PROMPT,
        "deployment_matches_selection": bool(selected_checkpoint)
        and deployment.get("checkpoint") == selected_checkpoint,
        "deployment_positive_prompt": deployment.get("prompt") == POSITIVE_PROMPT,
        "deployment_inference_smoke": smoke.get("passed") is True
        and smoke.get("checkpoint") == selected_checkpoint
        and smoke.get("prompt") == POSITIVE_PROMPT
        and smoke.get("actions", {}).get("shape") == [50, 16]
        and smoke.get("contract", {}).get("passed") is True,
        "report_matches_selection": args.report_html.is_file()
        and report_payload.get("selection", {}).get("selected_checkpoint") == selected_checkpoint
        and report_payload.get("deployment", {}).get("checkpoint") == selected_checkpoint,
    }
    return {
        "schema_version": "openarm_kai0_completion_audit_v1",
        "passed": all(gates.values()),
        "gates": gates,
        "summary": {
            "site_mode": site.get("mode"),
            "site_checkpoint": site.get("checkpoint"),
            "site_episode_count": site_evidence.get("metrics", site_evidence.get("curve_metrics", {})).get(
                "episode_count"
            ),
            "k_data_episodes": k_report.get("total_episodes"),
            "candidate_steps": candidate_steps,
            "selected_step": selected_step,
            "selected_checkpoint": selected_checkpoint,
            "deployment_host": deployment.get("host"),
            "deployment_port": deployment.get("port"),
            "smoke_elapsed_ms": smoke.get("elapsed_ms"),
        },
    }


def _write_json_atomic(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hq-audit", type=pathlib.Path, required=True)
    parser.add_argument("--site-selection", type=pathlib.Path, required=True)
    parser.add_argument("--k-data-report", type=pathlib.Path, required=True)
    parser.add_argument("--k-data-audit", type=pathlib.Path, required=True)
    parser.add_argument("--norm-stats", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-root", type=pathlib.Path, required=True)
    parser.add_argument("--policy-selection", type=pathlib.Path, required=True)
    parser.add_argument("--deployment", type=pathlib.Path, required=True)
    parser.add_argument("--report-html", type=pathlib.Path, required=True)
    parser.add_argument("--report-payload", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    _write_json_atomic(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
