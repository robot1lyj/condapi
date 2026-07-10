"""Start and supervise six Site Stage score shards as their HQ GPU slots become free."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import time
from typing import Any

REPO_ROOT = pathlib.Path("/share/home/linyongjia/conda-pi/openpi")
PYTHON = "/share/home/linyongjia/miniconda3/envs/pi-conda/bin/python"
DATASETS_ROOT = pathlib.Path("/share/home/linyongjia/datasets")
SOURCE = DATASETS_ROOT / "openarm_site_align_v1_deg"
ANNOTATIONS = SOURCE / "annotations/openarm_stage_v1.jsonl"
DESTINATION_PREFIX = DATASETS_ROOT / "openarm_kai0_site_scores_direct_v1"
CHECKPOINT = pathlib.Path(
    "/share/home/linyongjia/output/openpi/ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD/"
    "openarm_stage_v1_train180_bs32_no_ckpt_10k_20260701/10000"
)
OUTPUT_ROOT = pathlib.Path("/share/home/linyongjia/output/openpi/logs/openarm_kai0_site_scores_direct_v1")
MANIFEST_PATH = OUTPUT_ROOT / "shards.json"
LATEST_STATUS = OUTPUT_ROOT / "monitor_latest.json"
HISTORY_LOG = OUTPUT_ROOT / "monitor.jsonl"
ALERT_LOG = OUTPUT_ROOT / "monitor_alerts.jsonl"
STATE_PATH = OUTPUT_ROOT / "monitor_state.json"
COMPLETE_MARKER = OUTPUT_ROOT / "site150_scoring_complete"
AUDIT_ROOT = DATASETS_ROOT / "openarm_site_score_review_v1"
SITE_STAGE_DATA = DATASETS_ROOT / "openarm_site_stage_v1"
PAIR_EVAL_PATH = AUDIT_ROOT / "hq_stage_site_val10_pairs.json"
DECISION_PATH = AUDIT_ROOT / "site_stage_decision.json"

SLOTS = (
    {"host": "gpu12", "gpu_id": 0, "hq_shard": "s0_000_167", "hq_expected": 167},
    {"host": "gpu12", "gpu_id": 1, "hq_shard": "s1_167_334", "hq_expected": 167},
    {"host": "gpu14", "gpu_id": 0, "hq_shard": "s2_334_501", "hq_expected": 167},
    {"host": "gpu14", "gpu_id": 1, "hq_shard": "s3_501_668", "hq_expected": 167},
    {"host": "gpu28", "gpu_id": 0, "hq_shard": "s4_668_835", "hq_expected": 167},
    {"host": "gpu28", "gpu_id": 1, "hq_shard": "s5_835_999", "hq_expected": 164},
)


def _load_json(path: pathlib.Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _write_json_atomic(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")


def _ssh_argv(host: str, remote_args: list[str]) -> list[str]:
    # OpenSSH joins trailing argv with spaces before invoking the remote shell, so quote the remote argv ourselves.
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, shlex.join(remote_args)]


def _ssh(host: str, remote_args: list[str], *, input_text: str | None = None, timeout: int = 30) -> str:
    result = subprocess.run(
        _ssh_argv(host, remote_args),
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
    )
    return result.stdout.strip()


def _remote_status(slot: dict[str, Any], shard_name: str, expected: int, stall_seconds: int) -> dict[str, Any]:
    session = f"kai0_site_{shard_name}"
    log_path = OUTPUT_ROOT / f"{shard_name}.log"
    script = r"""
session="$1"
site_root="$2"
expected="$3"
log_path="$4"
hq_root="$5"
hq_expected="$6"
stall_seconds="$7"
site_count="$(find "$site_root/data" -name '*.parquet' 2>/dev/null | wc -l)"
meta_count="$(wc -l < "$site_root/meta/episodes.jsonl" 2>/dev/null || echo 0)"
hq_count="$(find "$hq_root/data" -name '*.parquet' 2>/dev/null | wc -l)"
if [[ "$site_count" -ge "$expected" && "$meta_count" -ge "$expected" ]]; then
    state="complete"
elif tmux has-session -t "$session" 2>/dev/null; then
    now="$(date +%s)"
    log_mtime="$(stat -c %Y "$log_path" 2>/dev/null || echo 0)"
    if (( log_mtime > 0 && now - log_mtime > stall_seconds )); then state="stalled"; else state="running"; fi
elif [[ "$hq_count" -ge "$hq_expected" ]]; then
    state="ready"
else
    state="waiting_hq"
fi
printf '%s %s %s %s\n' "$state" "$site_count" "$meta_count" "$hq_count"
"""
    site_root = f"{DESTINATION_PREFIX}_{shard_name}"
    hq_root = DATASETS_ROOT / f"openarm_kai0_stage_scores_hq_v1_{slot['hq_shard']}"
    output = _ssh(
        str(slot["host"]),
        [
            "bash",
            "-s",
            "--",
            session,
            site_root,
            str(expected),
            str(log_path),
            str(hq_root),
            str(slot["hq_expected"]),
            str(stall_seconds),
        ],
        input_text=script,
    )
    state, count, meta_count, hq_count = output.split()
    return {
        "state": state,
        "completed": int(count),
        "metadata_rows": int(meta_count),
        "hq_completed": int(hq_count),
    }


def _start_job(slot: dict[str, Any], shard: dict[str, Any], *, resume: bool) -> None:
    shard_name = f"s{int(shard['shard_index'])}"
    session = f"kai0_site_{shard_name}"
    log_path = OUTPUT_ROOT / f"{shard_name}.log"
    command = [
        "bash",
        "scripts/run_openarm_stage_score_shard.sh",
        "--gpu-id",
        str(slot["gpu_id"]),
        "--episodes",
        str(shard["episode_spec"]),
        "--shard-name",
        shard_name,
        "--batch-size",
        "32",
        "--source",
        str(SOURCE),
        "--destination-prefix",
        str(DESTINATION_PREFIX),
        "--checkpoint",
        str(CHECKPOINT),
    ]
    if resume:
        command.append("--resume")
    remote_command = (
        f"cd {shlex.quote(str(REPO_ROOT))} && exec {shlex.join(command)} >>{shlex.quote(str(log_path))} 2>&1"
    )
    kill_script = 'tmux kill-session -t "$1" 2>/dev/null || true\ntmux new-session -d -s "$1" "$2"\n'
    _ssh(str(slot["host"]), ["bash", "-s", "--", session, remote_command], input_text=kill_script)


def _finalize_site_audit() -> dict[str, Any]:
    score_roots = [f"{DESTINATION_PREFIX}_s{index}" for index in range(6)]
    command = [
        PYTHON,
        "scripts/audit_openarm_site_stage_scores.py",
        "--source",
        str(SOURCE),
        "--annotations",
        str(ANNOTATIONS),
        "--output-root",
        str(AUDIT_ROOT),
        "--expected-episodes",
        "150",
        "--relative-interval",
        "50",
        "--overwrite",
    ]
    for score_root in score_roots:
        command.extend(("--score-root", score_root))
    pair_command = [
        PYTHON,
        "scripts/evaluate_stage_advantage.py",
        str(CHECKPOINT),
        "--config-name",
        "ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD",
        "--dataset",
        str(SITE_STAGE_DATA),
        "--episodes",
        "140:150",
        "--batch-size",
        "32",
        "--num-workers",
        "2",
        "--max-batches",
        "100",
        "--samples-per-batch",
        "1",
        "--sign-epsilon",
        "0.005",
        "--device",
        "cuda:0",
        "--output",
        str(PAIR_EVAL_PATH),
    ]
    audit_script = (
        f"cd {shlex.quote(str(REPO_ROOT))}\n{shlex.join(command)}\nCUDA_VISIBLE_DEVICES=0 {shlex.join(pair_command)}\n"
    )
    output = _ssh("gpu28", ["bash", "-s"], input_text=audit_script, timeout=7200)

    serve_command = (
        f"cd {shlex.quote(str(REPO_ROOT))} && exec {shlex.quote(PYTHON)} "
        "scripts/serve_openarm_advantage_report.py "
        f"--dataset {shlex.quote(str(AUDIT_ROOT))} --index-path site_score_report/index.html "
        "--host 0.0.0.0 --port 8768"
    )
    serve_script = 'tmux kill-session -t "$1" 2>/dev/null || true\ntmux new-session -d -s "$1" "$2"\n'
    _ssh(
        "gpu28",
        ["bash", "-s", "--", "openarm_site_score_report_v1", serve_command],
        input_text=serve_script,
    )
    audit = _load_json(AUDIT_ROOT / "site_stage_audit.json", {})
    pair_eval = _load_json(PAIR_EVAL_PATH, {})

    def metric(name: str, default: float) -> float:
        value = pair_eval.get(name)
        return default if value is None else float(value)

    pair_gates = {
        "mse<=0.020": metric("mse", float("inf")) <= 0.020,
        "mae<=0.110": metric("mae", float("inf")) <= 0.110,
        "sign_accuracy>=0.88": metric("sign_accuracy", 0.0) >= 0.88,
        "corrcoef>=0.92": metric("corrcoef", 0.0) >= 0.92,
        "r2>=0.80": metric("r2", 0.0) >= 0.80,
    }
    decision = {
        "direct_transfer_passed": bool(audit.get("passed")) and all(pair_gates.values()),
        "curve_audit_passed": bool(audit.get("passed")),
        "curve_metrics": audit.get("metrics", {}),
        "pair_eval": pair_eval,
        "pair_gates": pair_gates,
        "action": "promote_hq_stage_scores"
        if bool(audit.get("passed")) and all(pair_gates.values())
        else "train_site_stage",
    }
    _write_json_atomic(DECISION_PATH, decision)
    COMPLETE_MARKER.touch()
    return {"command_output": output, **decision}


def monitor_once(*, max_restarts: int, stall_seconds: int, auto_start: bool) -> dict[str, Any]:
    manifest = _load_json(MANIFEST_PATH)
    if not manifest or len(manifest.get("shards", [])) != len(SLOTS):
        raise ValueError(f"Missing or invalid Site shard manifest: {MANIFEST_PATH}")
    state = _load_json(STATE_PATH, {"jobs": {}})
    status: dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "jobs": []}
    all_complete = True

    for slot, shard in zip(SLOTS, manifest["shards"], strict=True):
        shard_name = f"s{int(shard['shard_index'])}"
        expected = int(shard["episode_count"])
        job_state = state["jobs"].setdefault(shard_name, {"last_completed": 0, "restarts": 0})
        try:
            remote = _remote_status(slot, shard_name, expected, stall_seconds)
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            remote = {"state": "unreachable", "completed": 0, "metadata_rows": 0, "hq_completed": 0}
            _append_jsonl(ALERT_LOG, {"timestamp": status["timestamp"], "shard": shard_name, "error": str(error)})

        if int(remote["completed"]) > int(job_state["last_completed"]):
            job_state["restarts"] = 0
        remote_state = str(remote["state"])
        should_start = remote_state == "ready" or remote_state in {"stalled", "unreachable"}
        if remote_state == "unreachable":
            should_start = False
        if should_start and auto_start:
            if int(job_state["restarts"]) >= max_restarts:
                remote_state = "restart_exhausted"
            else:
                job_state["restarts"] = int(job_state["restarts"]) + 1
                try:
                    _start_job(slot, shard, resume=True)
                    remote_state = "started" if remote_state == "ready" else "restarted"
                except (subprocess.SubprocessError, OSError) as error:
                    remote_state = "start_failed"
                    _append_jsonl(
                        ALERT_LOG,
                        {"timestamp": status["timestamp"], "shard": shard_name, "error": str(error)},
                    )
        if remote_state != "complete":
            all_complete = False
        job_state["last_completed"] = int(remote["completed"])
        status["jobs"].append(
            {
                "shard": shard_name,
                "host": slot["host"],
                "gpu_id": slot["gpu_id"],
                "state": remote_state,
                "completed": remote["completed"],
                "expected": expected,
                "hq_completed": remote["hq_completed"],
                "hq_expected": slot["hq_expected"],
                "restarts": job_state["restarts"],
                "planned_frames": shard["total_frames"],
            }
        )

    status["completed_episodes"] = sum(int(job["completed"]) for job in status["jobs"])
    status["expected_episodes"] = int(manifest["selected_episode_count"])
    status["all_complete"] = all_complete
    _write_json_atomic(STATE_PATH, state)
    _write_json_atomic(LATEST_STATUS, status)
    _append_jsonl(HISTORY_LOG, status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--stall-seconds", type=int, default=1200)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-auto-start", action="store_true")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    while True:
        status = monitor_once(
            max_restarts=args.max_restarts,
            stall_seconds=args.stall_seconds,
            auto_start=not args.no_auto_start,
        )
        print(json.dumps(status, ensure_ascii=False), flush=True)
        if status["all_complete"]:
            if not COMPLETE_MARKER.exists():
                final = _finalize_site_audit()
                print(json.dumps({"site_audit": final}, ensure_ascii=False), flush=True)
            break
        if args.once:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
