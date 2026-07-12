"""Supervise the OpenArm KAI0 path from Site audit through four-GPU policy training."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import time
from typing import Any

try:
    from scripts import launch_openarm_jax_multinode as jax_launcher
    from scripts import select_openarm_site_stage_checkpoint as stage_selector
    from scripts.openarm_kai0_contract import HQ_FOLDING_ONLY_START
    from scripts.openarm_kai0_contract import HQ_LAYOUT_TASK_START
except ImportError:
    import launch_openarm_jax_multinode as jax_launcher
    from openarm_kai0_contract import HQ_FOLDING_ONLY_START
    from openarm_kai0_contract import HQ_LAYOUT_TASK_START
    import select_openarm_site_stage_checkpoint as stage_selector


REPO_ROOT = pathlib.Path("/share/home/linyongjia/conda-pi/openpi")
PYTHON = pathlib.Path("/share/home/linyongjia/miniconda3/envs/pi-conda/bin/python")
DATASETS = pathlib.Path("/share/home/linyongjia/datasets")
DATA = pathlib.Path("/share/home/linyongjia/data")
OUTPUT = pathlib.Path("/share/home/linyongjia/output/openpi")
PIPELINE_ROOT = OUTPUT / "logs/openarm_kai0_pipeline_v1"
STATUS_PATH = PIPELINE_ROOT / "status.json"
HISTORY_PATH = PIPELINE_ROOT / "history.jsonl"
STATE_PATH = PIPELINE_ROOT / "state.json"

SITE_DECISION = DATASETS / "openarm_site_score_review_v1/site_stage_decision.json"
SITE_DIRECT_PREFIX = DATASETS / "openarm_kai0_site_scores_direct_v1"
SITE_FORMAL_PREFIX = DATASETS / "openarm_kai0_site_scores_v1"
SITE_SELECTION = PIPELINE_ROOT / "site_score_selection.json"
SITE_MANIFEST = OUTPUT / "logs/openarm_kai0_site_scores_direct_v1/shards.json"
SITE_SOURCE = DATASETS / "openarm_site_align_v1_deg"
SITE_ANNOTATIONS = SITE_SOURCE / "annotations/openarm_stage_v1.jsonl"
SITE_STAGE_DATA = DATASETS / "openarm_site_stage_v1"
SITE_STAGE_MIX = DATA / "openarm_stage_mix_site_v1"
HQ_STAGE_TRAIN = DATA / "high_quality_folding_v2p1_stage_train180"
HQ_STAGE_VAL = DATA / "high_quality_folding_v2p1_stage_val20"
HQ_STAGE_CHECKPOINT = (
    OUTPUT / "ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD" / "openarm_stage_v1_train180_bs32_no_ckpt_10k_20260701" / "10000"
)
SITE_STAGE_CONFIG = "ADVANTAGE_TORCH_OPENARM_SITE_FOLD"
SITE_STAGE_EXP = "openarm_site_stage_v1_hq180_site140x3_bs32_5k_20260712"
SITE_STAGE_CHECKPOINT_ROOT = OUTPUT / SITE_STAGE_CONFIG / SITE_STAGE_EXP
SITE_STAGE_FINAL = SITE_STAGE_CHECKPOINT_ROOT / "4999/model.safetensors"
SITE_STAGE_EVAL = PIPELINE_ROOT / "site_stage_eval"
SITE_STAGE_SELECTION = SITE_STAGE_EVAL / "selection.json"
SITE_ADAPTED_AUDIT = DATASETS / "openarm_site_score_review_adapted_v1"

HQ_SCORE_ROOTS = tuple(
    DATASETS / f"openarm_kai0_stage_scores_hq_v1_{name}"
    for name in ("s0_000_167", "s1_167_334", "s2_334_501", "s3_501_668", "s4_668_835", "s5_835_999")
)
HQ_SCORE_AUDIT = PIPELINE_ROOT / "hq999_stage_audit.json"
TDA_DATA = DATASETS / "openarm_hq_tda_aug_v1"
K_DATA = DATASETS / "openarm_kai0_awbc_v1"
K_DATA_REPORT = K_DATA / "kai0_awbc_build_report.json"
K_DATA_AUDIT = PIPELINE_ROOT / "k_data_training_audit.json"
K_CONFIG = "pi05_openarm_kai0_awbc_v1"
K_SMOKE_EXP = "openarm_kai0_awbc_v1_4gpu_smoke20_20260710"
K_FULL_EXP = "openarm_kai0_awbc_v1_4gpu_80k_20260710"
K_SMOKE_CHECKPOINT = OUTPUT / K_CONFIG / K_SMOKE_EXP / "19"
K_FULL_ROOT = OUTPUT / K_CONFIG / K_FULL_EXP
K_FULL_CHECKPOINT = K_FULL_ROOT / "79999"
K_SWEEP_ROOT = PIPELINE_ROOT / "checkpoint_sweep"
K_POLICY_SELECTION = K_SWEEP_ROOT / "selection.json"
K_DEPLOYMENT = PIPELINE_ROOT / "gpu25_deployment.json"
K_POLICY_REPORT = PIPELINE_ROOT / "policy_report/index.html"
K_POLICY_REPORT_PAYLOAD = PIPELINE_ROOT / "policy_report/report.json"
POSITIVE_PROMPT = "Fold the T-shirt properly, Advantage: positive"
SWEEP_SCHEMA_VERSION = "openarm_checkpoint_sweep_v2"

SLOTS = (
    ("gpu12", 0),
    ("gpu12", 1),
    ("gpu14", 0),
    ("gpu14", 1),
    ("gpu28", 0),
    ("gpu28", 1),
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


def _ssh_argv(host: str, args: list[str]) -> list[str]:
    # OpenSSH joins trailing argv with spaces before invoking the remote shell, so quote the remote argv ourselves.
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, shlex.join(args)]


def _ssh(host: str, args: list[str], *, input_text: str | None = None, timeout: int = 30) -> str:
    result = subprocess.run(
        _ssh_argv(host, args),
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
    )
    return result.stdout.strip()


def _session_exists(host: str, session: str) -> bool:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, "tmux", "has-session", "-t", session],
        capture_output=True,
        timeout=20,
        check=False,
    )
    return result.returncode == 0


def _kill_session(host: str, session: str) -> None:
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, "tmux", "kill-session", "-t", session],
        capture_output=True,
        timeout=20,
        check=False,
    )


def _exit_code(path: pathlib.Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return -999


def _checkpoint_ready(path: pathlib.Path) -> bool:
    return path.is_dir() and (path / "_CHECKPOINT_METADATA").is_file() and (path / "params/_METADATA").is_file()


def _start_tmux(host: str, session: str, command: str, exit_marker: pathlib.Path) -> None:
    wrapped = "\n".join(
        [
            "set +e",
            f"rm -f {shlex.quote(str(exit_marker))}",
            "(",
            command,
            ")",
            "rc=$?",
            f"printf '%s\\n' \"$rc\" > {shlex.quote(str(exit_marker))}",
            'exit "$rc"',
        ]
    )
    script = r"""
session="$1"
command="$2"
tmux kill-session -t "$session" 2>/dev/null || true
tmux new-session -d -s "$session" "$command"
"""
    _ssh(host, ["bash", "-s", "--", session, wrapped], input_text=script)


def _dataset_episode_count(root: pathlib.Path) -> int:
    info = _load_json(root / "meta/info.json", {})
    return int(info.get("total_episodes", 0))


def _score_roots(prefix: pathlib.Path) -> list[pathlib.Path]:
    return [pathlib.Path(f"{prefix}_s{index}") for index in range(6)]


def _completed_score_episodes(root: pathlib.Path) -> int:
    episodes_path = root / "meta/episodes.jsonl"
    if not episodes_path.exists():
        return 0
    with episodes_path.open() as file:
        return sum(1 for line in file if line.strip())


def _site_selection_direct() -> dict[str, Any]:
    selection = {
        "mode": "hq_stage_direct_transfer",
        "checkpoint": str(HQ_STAGE_CHECKPOINT),
        "config": "ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD",
        "score_roots": [str(path) for path in _score_roots(SITE_DIRECT_PREFIX)],
        "decision": _load_json(SITE_DECISION),
    }
    _write_json_atomic(SITE_SELECTION, selection)
    return selection


def _ensure_site_stage_mix() -> None:
    report = _load_json(SITE_STAGE_MIX / "merge_report.json", {})
    if int(report.get("total_episodes", 0)) == 600:
        return
    command = [
        str(PYTHON),
        "scripts/merge_openarm_lerobot_v21.py",
        "--source",
        f"hq,{HQ_STAGE_TRAIN},0:180,1",
        "--source",
        f"site,{SITE_STAGE_DATA},0:140,3",
        "--dst",
        str(SITE_STAGE_MIX),
        "--task",
        "Fold the T-shirt properly",
        "--copy-mode",
        "hardlink",
        "--overwrite",
    ]
    _ssh("gpu28", ["bash", "-lc", f"cd {shlex.quote(str(REPO_ROOT))} && {shlex.join(command)}"], timeout=7200)
    if _dataset_episode_count(SITE_STAGE_MIX) != 600:
        raise RuntimeError("Site-Stage mixed dataset did not materialize 600 episodes")


def _ensure_site_stage_training(state: dict[str, Any]) -> bool:
    if SITE_STAGE_FINAL.exists():
        return True
    _ensure_site_stage_mix()
    session = "kai0_site_stage_train"
    exit_marker = PIPELINE_ROOT / "site_stage_train.exit"
    if _session_exists("gpu28", session):
        return False
    retries = int(state.setdefault("restarts", {}).get("site_stage_train", 0))
    if retries >= 3:
        raise RuntimeError("Site-Stage training exhausted three launch attempts")
    has_checkpoint = any(path.is_dir() and path.name.isdigit() for path in SITE_STAGE_CHECKPOINT_ROOT.glob("*"))
    mode = "--resume" if has_checkpoint else "--overwrite"
    log = PIPELINE_ROOT / "site_stage_train.log"
    train = (
        f"cd {shlex.quote(str(REPO_ROOT))} && unset WANDB_DISABLED && "
        "export WANDB_MODE=offline WANDB_SILENT=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
        "HF_DATASETS_OFFLINE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false && "
        "CUDA_VISIBLE_DEVICES=0,1 "
        f"{shlex.quote(str(PYTHON.parent / 'torchrun'))} --standalone --nnodes=1 --nproc_per_node=2 "
        f"scripts/train_pytorch.py {SITE_STAGE_CONFIG} --exp-name {shlex.quote(SITE_STAGE_EXP)} "
        f"--checkpoint-base-dir {shlex.quote(str(OUTPUT))} {mode} >>{shlex.quote(str(log))} 2>&1"
    )
    _start_tmux("gpu28", session, train, exit_marker)
    state["restarts"]["site_stage_train"] = retries + 1
    return False


def _stage_eval_commands() -> str:
    SITE_STAGE_EVAL.mkdir(parents=True, exist_ok=True)
    commands = [f"cd {shlex.quote(str(REPO_ROOT))}", "set -e"]
    for step in (1000, 2000, 3000, 4000, 4999):
        checkpoint = SITE_STAGE_CHECKPOINT_ROOT / str(step)
        site_output = SITE_STAGE_EVAL / f"site_{step}.json"
        hq_output = SITE_STAGE_EVAL / f"hq_{step}.json"
        site_command = [
            str(PYTHON),
            "scripts/evaluate_stage_advantage.py",
            str(checkpoint),
            "--config-name",
            SITE_STAGE_CONFIG,
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
            str(site_output),
        ]
        hq_command = [
            str(PYTHON),
            "scripts/evaluate_stage_advantage.py",
            str(checkpoint),
            "--config-name",
            SITE_STAGE_CONFIG,
            "--dataset",
            str(HQ_STAGE_VAL),
            "--episodes",
            "0:20",
            "--batch-size",
            "32",
            "--num-workers",
            "2",
            "--max-batches",
            "25",
            "--samples-per-batch",
            "1",
            "--sign-epsilon",
            "0.005",
            "--device",
            "cuda:0",
            "--output",
            str(hq_output),
        ]
        commands.extend(
            [
                f"CUDA_VISIBLE_DEVICES=0 {shlex.join(site_command)} >{shlex.quote(str(site_output) + '.log')} 2>&1 &",
                "site_pid=$!",
                f"CUDA_VISIBLE_DEVICES=1 {shlex.join(hq_command)} >{shlex.quote(str(hq_output) + '.log')} 2>&1 &",
                "hq_pid=$!",
                'set +e; wait "$site_pid"; site_rc=$?; wait "$hq_pid"; hq_rc=$?; set -e',
                'test "$site_rc" -eq 0 -a "$hq_rc" -eq 0',
            ]
        )
    return "\n".join(commands)


def _ensure_site_stage_selection(state: dict[str, Any]) -> pathlib.Path | None:
    if SITE_STAGE_SELECTION.exists():
        return pathlib.Path(_load_json(SITE_STAGE_SELECTION)["selected_checkpoint"])
    expected_reports = [
        SITE_STAGE_EVAL / f"{kind}_{step}.json" for step in (1000, 2000, 3000, 4000, 4999) for kind in ("site", "hq")
    ]
    if all(path.exists() for path in expected_reports):
        selection = stage_selector.select_checkpoint(
            SITE_STAGE_EVAL,
            SITE_STAGE_CHECKPOINT_ROOT,
            [1000, 2000, 3000, 4000, 4999],
        )
        _write_json_atomic(SITE_STAGE_SELECTION, selection)
        return pathlib.Path(selection["selected_checkpoint"])
    session = "kai0_site_stage_eval"
    if _session_exists("gpu28", session):
        return None
    retries = int(state.setdefault("restarts", {}).get("site_stage_eval", 0))
    if retries >= 3:
        raise RuntimeError("Site-Stage evaluation exhausted three attempts")
    _start_tmux("gpu28", session, _stage_eval_commands(), PIPELINE_ROOT / "site_stage_eval.exit")
    state["restarts"]["site_stage_eval"] = retries + 1
    return None


def _start_adapted_score_shard(host: str, gpu_id: int, shard: dict[str, Any], checkpoint: pathlib.Path) -> None:
    shard_name = f"s{int(shard['shard_index'])}"
    session = f"kai0_site_adapt_{shard_name}"
    log = PIPELINE_ROOT / f"site_adapt_{shard_name}.log"
    command = [
        "bash",
        "scripts/run_openarm_stage_score_shard.sh",
        "--gpu-id",
        str(gpu_id),
        "--episodes",
        str(shard["episode_spec"]),
        "--shard-name",
        shard_name,
        "--batch-size",
        "32",
        "--source",
        str(SITE_SOURCE),
        "--destination-prefix",
        str(SITE_FORMAL_PREFIX),
        "--checkpoint",
        str(checkpoint),
        "--config-name",
        SITE_STAGE_CONFIG,
        "--resume",
    ]
    remote_command = f"cd {shlex.quote(str(REPO_ROOT))} && {shlex.join(command)} >>{shlex.quote(str(log))} 2>&1"
    _start_tmux(host, session, remote_command, PIPELINE_ROOT / f"site_adapt_{shard_name}.exit")


def _ensure_adapted_site_scores(checkpoint: pathlib.Path, state: dict[str, Any]) -> dict[str, Any] | None:
    manifest = _load_json(SITE_MANIFEST)
    if not manifest or len(manifest.get("shards", [])) != 6:
        raise ValueError(f"Invalid Site shard manifest: {SITE_MANIFEST}")
    roots = _score_roots(SITE_FORMAL_PREFIX)
    all_complete = True
    progress = []
    for (host, gpu_id), shard, root in zip(SLOTS, manifest["shards"], roots, strict=True):
        expected = int(shard["episode_count"])
        completed = _completed_score_episodes(root)
        shard_name = f"s{int(shard['shard_index'])}"
        session = f"kai0_site_adapt_{shard_name}"
        if completed < expected:
            all_complete = False
            if not _session_exists(host, session):
                key = f"site_adapt_{shard_name}"
                retries = int(state.setdefault("restarts", {}).get(key, 0))
                if retries >= 3:
                    raise RuntimeError(f"Adapted Site scoring {shard_name} exhausted three attempts")
                _start_adapted_score_shard(host, gpu_id, shard, checkpoint)
                state["restarts"][key] = retries + 1
        progress.append({"shard": shard_name, "completed": completed, "expected": expected})
    state["adapted_site_progress"] = progress
    if not all_complete:
        return None

    audit_path = SITE_ADAPTED_AUDIT / "site_stage_audit.json"
    if not audit_path.exists():
        command = [
            str(PYTHON),
            "scripts/audit_openarm_site_stage_scores.py",
            "--source",
            str(SITE_SOURCE),
            "--annotations",
            str(SITE_ANNOTATIONS),
            "--output-root",
            str(SITE_ADAPTED_AUDIT),
            "--expected-episodes",
            "150",
            "--relative-interval",
            "50",
            "--overwrite",
        ]
        for root in roots:
            command.extend(("--score-root", str(root)))
        _ssh("gpu28", ["bash", "-lc", f"cd {shlex.quote(str(REPO_ROOT))} && {shlex.join(command)}"], timeout=7200)
    audit = _load_json(audit_path, {})
    if not audit.get("passed"):
        raise RuntimeError("Adapted Site-Stage scores failed the curve quality gate")
    selection = {
        "mode": "adapted_site_stage",
        "checkpoint": str(checkpoint),
        "config": SITE_STAGE_CONFIG,
        "score_roots": [str(path) for path in roots],
        "direct_decision": _load_json(SITE_DECISION),
        "adapted_audit": audit,
        "checkpoint_selection": _load_json(SITE_STAGE_SELECTION),
    }
    _write_json_atomic(SITE_SELECTION, selection)
    return selection


def _ensure_site_selection(state: dict[str, Any]) -> dict[str, Any] | None:
    selection = _load_json(SITE_SELECTION)
    if selection:
        return selection
    decision = _load_json(SITE_DECISION)
    if not decision:
        return None
    if decision.get("direct_transfer_passed"):
        return _site_selection_direct()
    if not _ensure_site_stage_training(state):
        return None
    checkpoint = _ensure_site_stage_selection(state)
    if checkpoint is None:
        return None
    return _ensure_adapted_site_scores(checkpoint, state)


def _build_k_data_command(selection: dict[str, Any]) -> str:
    command = [
        str(PYTHON),
        "scripts/build_openarm_kai0_awbc_dataset.py",
        "--tda-augmented",
        str(TDA_DATA),
        "--destination",
        str(K_DATA),
        "--site-repeat",
        "3",
        "--tda-time-count",
        "150",
        "--tda-mirror-count",
        "150",
        "--positive-ratio",
        "0.30",
        "--relative-interval",
        "50",
        "--hq-folding-only-start",
        str(HQ_FOLDING_ONLY_START),
        "--hq-layout-task-start",
        str(HQ_LAYOUT_TASK_START),
        "--site-annotations",
        str(SITE_ANNOTATIONS),
        "--overwrite",
    ]
    for root in HQ_SCORE_ROOTS:
        command.extend(("--hq-score-root", str(root)))
    for root in selection["score_roots"]:
        command.extend(("--site-score-root", str(root)))
    log = PIPELINE_ROOT / "build_k_data.log"
    return f"cd {shlex.quote(str(REPO_ROOT))} && {shlex.join(command)} >>{shlex.quote(str(log))} 2>&1"


def _ensure_hq_score_audit() -> bool:
    report = _load_json(HQ_SCORE_AUDIT)
    if report:
        if not report.get("passed"):
            raise RuntimeError("HQ999 Stage scores failed the final quality gate")
        return True
    command = [
        str(PYTHON),
        "scripts/audit_openarm_hq_stage_scores.py",
        "--expected-episodes",
        "999",
        "--folding-only-start",
        str(HQ_FOLDING_ONLY_START),
        "--output",
        str(HQ_SCORE_AUDIT),
    ]
    for root in HQ_SCORE_ROOTS:
        command.extend(("--score-root", str(root)))
    _ssh("gpu25", ["bash", "-lc", f"cd {shlex.quote(str(REPO_ROOT))} && {shlex.join(command)}"], timeout=7200)
    report = _load_json(HQ_SCORE_AUDIT, {})
    if not report.get("passed"):
        raise RuntimeError("HQ999 Stage audit did not produce a passing report")
    return True


def _ensure_k_data(selection: dict[str, Any], state: dict[str, Any]) -> bool:
    report = _load_json(K_DATA_REPORT, {})
    if int(report.get("total_episodes", 0)) == 1719:
        return True
    session = "kai0_build_k_data"
    if _session_exists("gpu28", session):
        return False
    retries = int(state.setdefault("restarts", {}).get("build_k_data", 0))
    previous_exit = _exit_code(PIPELINE_ROOT / "build_k_data.exit")
    if previous_exit not in (None, 0) and retries >= 2:
        raise RuntimeError(f"K-Data build failed with exit code {previous_exit}")
    _start_tmux("gpu28", session, _build_k_data_command(selection), PIPELINE_ROOT / "build_k_data.exit")
    state["restarts"]["build_k_data"] = retries + 1
    return False


def _ensure_norm_stats(state: dict[str, Any]) -> bool:
    norm_path = K_DATA / "norm_stats.json"
    if norm_path.exists():
        norm = _load_json(norm_path, {})
        if set(norm.get("norm_stats", {})) >= {"state", "actions"}:
            return True
    session = "kai0_k_data_norm"
    if _session_exists("gpu28", session):
        return False
    retries = int(state.setdefault("restarts", {}).get("k_data_norm", 0))
    previous_exit = _exit_code(PIPELINE_ROOT / "k_data_norm.exit")
    if previous_exit not in (None, 0) and retries >= 2:
        raise RuntimeError(f"K-Data norm stats failed with exit code {previous_exit}")
    log = PIPELINE_ROOT / "k_data_norm.log"
    command = (
        f"cd {shlex.quote(str(REPO_ROOT))} && {shlex.quote(str(PYTHON))} "
        f"scripts/compute_openarm_parquet_norm_stats.py --dataset {shlex.quote(str(K_DATA))} "
        f"--episodes 0:1719 >>{shlex.quote(str(log))} 2>&1"
    )
    _start_tmux("gpu28", session, command, PIPELINE_ROOT / "k_data_norm.exit")
    state["restarts"]["k_data_norm"] = retries + 1
    return False


def _ensure_k_data_audit() -> bool:
    report = _load_json(K_DATA_AUDIT)
    if report:
        if not report.get("passed"):
            raise RuntimeError("K-Data failed the formal OpenPI training-data audit")
        return True
    command = [
        str(PYTHON),
        "scripts/audit_openarm_kai0_training_data.py",
        "--dataset",
        str(K_DATA),
        "--config",
        K_CONFIG,
        "--expected-episodes",
        "1719",
        "--expected-hq",
        "999",
        "--expected-site",
        "420",
        "--expected-tda",
        "300",
        "--output",
        str(K_DATA_AUDIT),
    ]
    _ssh("gpu28", ["bash", "-lc", f"cd {shlex.quote(str(REPO_ROOT))} && {shlex.join(command)}"], timeout=7200)
    report = _load_json(K_DATA_AUDIT, {})
    if not report.get("passed"):
        raise RuntimeError("K-Data audit did not produce a passing report")
    return True


def _jax_job_state(exp_name: str, session_prefix: str, final_checkpoint: pathlib.Path) -> dict[str, Any]:
    sessions = {host: _session_exists(host, f"{session_prefix}_{host}") for host in ("gpu12", "gpu14")}
    exits = {host: _exit_code(OUTPUT / "logs" / K_CONFIG / f"{exp_name}_{host}.exit") for host in ("gpu12", "gpu14")}
    checkpoint_root = OUTPUT / K_CONFIG / exp_name
    steps = sorted(int(path.name) for path in checkpoint_root.glob("*") if path.is_dir() and path.name.isdigit())
    return {
        "complete": _checkpoint_ready(final_checkpoint),
        "sessions": sessions,
        "exit_codes": exits,
        "latest_checkpoint": steps[-1] if steps else None,
    }


def _ensure_jax_job(
    state: dict[str, Any],
    *,
    key: str,
    exp_name: str,
    session_prefix: str,
    num_train_steps: int,
    num_workers: int,
    final_checkpoint: pathlib.Path,
    max_restarts: int,
) -> bool:
    job = _jax_job_state(exp_name, session_prefix, final_checkpoint)
    state[f"{key}_status"] = job
    if job["complete"]:
        return True
    if all(job["sessions"].values()):
        return False
    if any(job["sessions"].values()):
        for host, running in job["sessions"].items():
            if running:
                _kill_session(host, f"{session_prefix}_{host}")
    retries = int(state.setdefault("restarts", {}).get(key, 0))
    if retries >= max_restarts:
        raise RuntimeError(f"{key} exhausted {max_restarts} launches: {job}")
    mode = "resume" if job["latest_checkpoint"] is not None else "overwrite"
    if key == "k_smoke":
        mode = "overwrite"
    jax_launcher.launch(
        config=K_CONFIG,
        exp_name=exp_name,
        num_train_steps=num_train_steps,
        batch_size=128,
        num_workers=num_workers,
        log_interval=1 if key == "k_smoke" else 20,
        mode=mode,
        session_prefix=session_prefix,
        coordinator_address="172.31.11.112:12365",
        xla_memory_fraction=0.90,
        dry_run=False,
    )
    state["restarts"][key] = retries + 1
    return False


def _policy_checkpoint_steps() -> list[int]:
    return sorted(int(path.name) for path in K_FULL_ROOT.glob("*") if path.name.isdigit() and _checkpoint_ready(path))


def _is_policy_sweep_step(step: int) -> bool:
    return step % 5_000 == 0 or step == 79_999


def _sweep_command() -> str:
    K_SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    steps = _policy_checkpoint_steps()
    checkpoints = [K_FULL_ROOT / str(step) for step in steps if _is_policy_sweep_step(step)]
    if not checkpoints:
        raise RuntimeError("No policy checkpoints found for sweep")

    def command(
        dataset: pathlib.Path, output: pathlib.Path, train_split: str, val_split: str, val_max: int
    ) -> list[str]:
        result = [
            str(PYTHON),
            "scripts/evaluate_openarm_checkpoint_sweep.py",
            "--config",
            K_CONFIG,
            "--dataset",
            str(dataset),
            "--train-split",
            train_split,
            "--val-split",
            val_split,
            "--train-max-episodes",
            "20",
            "--val-max-episodes",
            str(val_max),
            "--uniform-frames",
            "3",
            "--critical-frames",
            "3",
            "--include-adjacent",
            "--prompt",
            POSITIVE_PROMPT,
            "--resume",
            "--output",
            str(output),
        ]
        for checkpoint in checkpoints:
            result.extend(("--checkpoint", str(checkpoint)))
        return result

    hq_output = K_SWEEP_ROOT / "hq"
    site_output = K_SWEEP_ROOT / "site"
    hq = command(DATASETS / "high_quality_folding", hq_output, "0:999", "999:1199", 40)
    site = command(SITE_SOURCE, site_output, "0:141", "141:151", 10)
    return "\n".join(
        [
            f"cd {shlex.quote(str(REPO_ROOT))}",
            "set -e",
            f"CUDA_VISIBLE_DEVICES=0 {shlex.join(hq)} >{shlex.quote(str(hq_output) + '.log')} 2>&1 &",
            "hq_pid=$!",
            f"CUDA_VISIBLE_DEVICES=1 {shlex.join(site)} >{shlex.quote(str(site_output) + '.log')} 2>&1 &",
            "site_pid=$!",
            'set +e; wait "$hq_pid"; hq_rc=$?; wait "$site_pid"; site_rc=$?; set -e',
            'test "$hq_rc" -eq 0 -a "$site_rc" -eq 0',
        ]
    )


def _select_policy_reports(
    hq_reports: dict[int, dict[str, Any]],
    site_reports: dict[int, dict[str, Any]],
    expected: list[int],
    checkpoint_root: pathlib.Path,
) -> dict[str, Any]:
    steps = sorted(set(hq_reports) & set(site_reports))
    if steps != expected:
        raise RuntimeError(f"Checkpoint sweep incomplete: expected={expected}, actual={steps}")

    rows = []
    for step in steps:
        hq = hq_reports[step]
        site = site_reports[step]
        for domain, report in (("HQ", hq), ("Site", site)):
            if report.get("schema_version") != SWEEP_SCHEMA_VERSION:
                raise RuntimeError(f"{domain} checkpoint {step} uses an incompatible sweep report schema")
            if report.get("sampling", {}).get("prompt_override") != POSITIVE_PROMPT:
                raise RuntimeError(f"{domain} checkpoint {step} was not evaluated with the positive AWBC prompt")
        rows.append(
            {
                "step": step,
                "checkpoint": str(checkpoint_root / str(step)),
                "site_mae": float(site["val"]["overall"]["mae"]),
                "site_critical_mae": float(site["val"]["by_kind"]["critical"]["mae"]),
                "site_overlap_mae": float(site["val"]["overall"]["overlap_consistency_mae"]),
                "site_gripper_mae": float(site["val"]["overall"]["gripper_mae"]),
                "hq_mae": float(hq["val"]["overall"]["mae"]),
                "hq_critical_mae": float(hq["val"]["by_kind"]["critical"]["mae"]),
                "hq_gap_ratio": float(hq["gaps"]["mae_val_over_train"]),
            }
        )
    weights = {
        "site_critical_mae": 0.30,
        "site_mae": 0.25,
        "hq_critical_mae": 0.20,
        "hq_mae": 0.15,
        "site_overlap_mae": 0.10,
    }
    for metric, weight in weights.items():
        ordered = sorted(rows, key=lambda row: row[metric])
        for rank, row in enumerate(ordered):
            row["rank_score"] = row.get("rank_score", 0.0) + weight * rank
    eligible = [row for row in rows if row["hq_gap_ratio"] <= 2.0]
    if not eligible:
        eligible = rows
    selected = min(eligible, key=lambda row: (row["rank_score"], row["site_critical_mae"], row["step"]))
    return {
        "selected_step": selected["step"],
        "selected_checkpoint": selected["checkpoint"],
        "selection_rule": "weighted rank: Site critical 30%, Site MAE 25%, HQ critical 20%, HQ MAE 15%, Site overlap 10%",
        "hq_gap_guard": "prefer checkpoints with HQ val/train MAE ratio <= 2.0",
        "positive_prompt": POSITIVE_PROMPT,
        "candidates": sorted(rows, key=lambda row: row["step"]),
    }


def _rank_policy_checkpoints() -> dict[str, Any]:
    hq_reports = {
        int(path.stem.split("_")[-1]): _load_json(path) for path in (K_SWEEP_ROOT / "hq").glob("checkpoint_*.json")
    }
    site_reports = {
        int(path.stem.split("_")[-1]): _load_json(path) for path in (K_SWEEP_ROOT / "site").glob("checkpoint_*.json")
    }
    expected = [step for step in _policy_checkpoint_steps() if _is_policy_sweep_step(step)]
    return _select_policy_reports(hq_reports, site_reports, expected, K_FULL_ROOT)


def _ensure_policy_selection(state: dict[str, Any]) -> dict[str, Any] | None:
    selection = _load_json(K_POLICY_SELECTION)
    if selection:
        return selection
    hq_summary = K_SWEEP_ROOT / "hq/summary.csv"
    site_summary = K_SWEEP_ROOT / "site/summary.csv"
    if hq_summary.exists() and site_summary.exists():
        selection = _rank_policy_checkpoints()
        _write_json_atomic(K_POLICY_SELECTION, selection)
        return selection
    session = "kai0_policy_sweep"
    if _session_exists("gpu28", session):
        return None
    retries = int(state.setdefault("restarts", {}).get("policy_sweep", 0))
    previous_exit = _exit_code(PIPELINE_ROOT / "policy_sweep.exit")
    if previous_exit not in (None, 0) and retries >= 2:
        raise RuntimeError(f"Policy checkpoint sweep failed with exit code {previous_exit}")
    _start_tmux("gpu28", session, _sweep_command(), PIPELINE_ROOT / "policy_sweep.exit")
    state["restarts"]["policy_sweep"] = retries + 1
    return None


def _ensure_gpu25_deployment(selection: dict[str, Any]) -> bool:
    deployment = _load_json(K_DEPLOYMENT)
    session = "openarm_kai0_policy_v1"
    if deployment and _session_exists("gpu25", session):
        return True
    checkpoint = pathlib.Path(selection["selected_checkpoint"])
    log = PIPELINE_ROOT / "gpu25_serve.log"
    command = (
        f"cd {shlex.quote(str(REPO_ROOT))} && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 "
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && "
        f"CUDA_VISIBLE_DEVICES=0 {shlex.quote(str(PYTHON))} scripts/serve_policy.py --port 6666 "
        f"--force-prompt {shlex.quote(POSITIVE_PROMPT)} policy:checkpoint "
        f"--policy.config={K_CONFIG} --policy.dir={shlex.quote(str(checkpoint))} "
        f">>{shlex.quote(str(log))} 2>&1"
    )
    _ssh("gpu25", ["bash", "-lc", "fuser -k 6666/tcp >/dev/null 2>&1 || true"])
    _start_tmux("gpu25", session, command, PIPELINE_ROOT / "gpu25_serve.exit")
    listening = ""
    for _ in range(60):
        time.sleep(10)
        if not _session_exists("gpu25", session):
            raise RuntimeError("gpu25 K-Policy server exited during startup")
        listening = _ssh(
            "gpu25",
            ["bash", "-lc", "if ss -ltn | grep -q ':6666 '; then echo listening; fi"],
        )
        if listening == "listening":
            break
    if listening != "listening":
        raise RuntimeError("gpu25 K-Policy server did not listen on port 6666 within ten minutes")
    deployment = {
        "host": "gpu25",
        "port": 6666,
        "session": session,
        "config": K_CONFIG,
        "checkpoint": str(checkpoint),
        "prompt": POSITIVE_PROMPT,
        "deployed_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    _write_json_atomic(K_DEPLOYMENT, deployment)
    return True


def _ensure_policy_report(selection: dict[str, Any]) -> bool:
    payload = _load_json(K_POLICY_REPORT_PAYLOAD, {})
    selected_checkpoint = selection["selected_checkpoint"]
    report_current = (
        K_POLICY_REPORT.exists()
        and payload.get("selection", {}).get("selected_checkpoint") == selected_checkpoint
        and payload.get("deployment", {}).get("checkpoint") == selected_checkpoint
    )
    if not report_current:
        command = [
            str(PYTHON),
            "scripts/build_openarm_kai0_policy_report.py",
            "--metrics",
            str(K_FULL_ROOT / "metrics/metrics.jsonl"),
            "--selection",
            str(K_POLICY_SELECTION),
            "--deployment",
            str(K_DEPLOYMENT),
            "--hq-audit",
            str(HQ_SCORE_AUDIT),
            "--site-selection",
            str(SITE_SELECTION),
            "--k-data-report",
            str(K_DATA_REPORT),
            "--k-data-audit",
            str(K_DATA_AUDIT),
            "--output",
            str(K_POLICY_REPORT),
        ]
        _ssh("gpu28", ["bash", "-lc", f"cd {shlex.quote(str(REPO_ROOT))} && {shlex.join(command)}"], timeout=300)
        payload = _load_json(K_POLICY_REPORT_PAYLOAD, {})
        if payload.get("selection", {}).get("selected_checkpoint") != selected_checkpoint:
            raise RuntimeError("K-Policy report did not capture the selected checkpoint")

    session = "openarm_kai0_policy_report_v1"
    if not _session_exists("gpu28", session):
        _ssh("gpu28", ["bash", "-lc", "fuser -k 8769/tcp >/dev/null 2>&1 || true"])
        serve = (
            f"cd {shlex.quote(str(REPO_ROOT))} && {shlex.quote(str(PYTHON))} "
            "scripts/serve_openarm_advantage_report.py "
            f"--dataset {shlex.quote(str(PIPELINE_ROOT))} --index-path policy_report/index.html "
            "--host 0.0.0.0 --port 8769"
        )
        _start_tmux("gpu28", session, serve, PIPELINE_ROOT / "policy_report_server.exit")
    for _ in range(30):
        listening = _ssh("gpu28", ["bash", "-lc", "if ss -ltn | grep -q ':8769 '; then echo yes; fi"])
        if listening == "yes":
            return True
        if not _session_exists("gpu28", session):
            raise RuntimeError("K-Policy report server exited during startup")
        time.sleep(1)
    raise RuntimeError("K-Policy report server did not listen on port 8769")


def monitor_once(state: dict[str, Any]) -> dict[str, Any]:
    status: dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    selection = _ensure_site_selection(state)
    if selection is None:
        direct_decision = _load_json(SITE_DECISION, {})
        status["phase"] = (
            "training_or_evaluating_site_stage"
            if direct_decision and not direct_decision.get("direct_transfer_passed")
            else "waiting_site_score_selection"
        )
    elif not _ensure_hq_score_audit():
        status["phase"] = "auditing_hq999_stage_scores"
    elif not _ensure_k_data(selection, state):
        status["phase"] = "building_k_data"
    elif not _ensure_norm_stats(state):
        status["phase"] = "computing_k_data_norm_stats"
    elif not _ensure_k_data_audit():
        status["phase"] = "auditing_k_data_loader"
    elif not _ensure_jax_job(
        state,
        key="k_smoke",
        exp_name=K_SMOKE_EXP,
        session_prefix="kai0_k_smoke",
        num_train_steps=20,
        num_workers=0,
        final_checkpoint=K_SMOKE_CHECKPOINT,
        max_restarts=2,
    ):
        status["phase"] = "training_k_policy_smoke20"
    elif not _ensure_jax_job(
        state,
        key="k_full",
        exp_name=K_FULL_EXP,
        session_prefix="kai0_k_full",
        num_train_steps=80_000,
        num_workers=2,
        final_checkpoint=K_FULL_CHECKPOINT,
        max_restarts=5,
    ):
        status["phase"] = "training_k_policy_80k"
    else:
        policy_selection = _ensure_policy_selection(state)
        if policy_selection is None:
            status["phase"] = "sweeping_k_policy_checkpoints"
        elif not _ensure_gpu25_deployment(policy_selection):
            status["phase"] = "deploying_gpu25"
        elif not _ensure_policy_report(policy_selection):
            status["phase"] = "building_k_policy_report"
        else:
            status["phase"] = "complete"
            status["deployment"] = _load_json(K_DEPLOYMENT)
            status["policy_report"] = {
                "path": str(K_POLICY_REPORT),
                "url": "http://gpu28:8769/policy_report/index.html",
            }
    status["site_selection"] = _load_json(SITE_SELECTION)
    status["hq_score_audit_passed"] = bool((_load_json(HQ_SCORE_AUDIT, {}) or {}).get("passed"))
    status["k_data_ready"] = K_DATA_REPORT.exists()
    status["norm_stats_ready"] = (K_DATA / "norm_stats.json").exists()
    status["k_data_audit_passed"] = bool(_load_json(K_DATA_AUDIT, {}).get("passed"))
    status["smoke_checkpoint_ready"] = _checkpoint_ready(K_SMOKE_CHECKPOINT)
    status["full_latest_checkpoint"] = _policy_checkpoint_steps()[-1] if _policy_checkpoint_steps() else None
    status["state"] = state
    _write_json_atomic(STATE_PATH, state)
    _write_json_atomic(STATUS_PATH, status)
    _append_jsonl(HISTORY_PATH, status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
    state = _load_json(STATE_PATH, {"restarts": {}})
    while True:
        try:
            status = monitor_once(state)
        except Exception as error:
            status = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "phase": "error",
                "error": f"{type(error).__name__}: {error}",
                "state": state,
            }
            _write_json_atomic(STATE_PATH, state)
            _write_json_atomic(STATUS_PATH, status)
            _append_jsonl(HISTORY_PATH, status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        if status["phase"] == "complete" or args.once:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
