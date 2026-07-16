"""Launch one OpenPI JAX job across configurable two-GPU cluster nodes."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import shlex
import subprocess

REPO_ROOT = pathlib.Path("/share/home/linyongjia/conda-pi/openpi")
PYTHON = pathlib.Path("/share/home/linyongjia/miniconda3/envs/pi-conda/bin/python")
OUTPUT_ROOT = pathlib.Path("/share/home/linyongjia/output/openpi")
DATASETS_ROOT = pathlib.Path("/share/home/linyongjia/datasets")
OPENPI_DATA_HOME = pathlib.Path("/share/home/linyongjia/.cache/openpi")
HF_HOME = pathlib.Path("/share/home/linyongjia/.cache/huggingface")


@dataclasses.dataclass(frozen=True)
class Node:
    host: str
    process_id: int


DEFAULT_HOSTS = ("gpu12", "gpu14", "gpu28")


def _ssh_argv(host: str, args: list[str]) -> list[str]:
    # OpenSSH joins trailing argv with spaces before invoking the remote shell, so quote the remote argv ourselves.
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, shlex.join(args)]


def _ssh(host: str, args: list[str], *, input_text: str | None = None) -> str:
    result = subprocess.run(
        _ssh_argv(host, args),
        input=input_text,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    return result.stdout.strip()


def build_node_command(
    node: Node,
    *,
    config: str,
    exp_name: str,
    num_train_steps: int,
    batch_size: int,
    num_workers: int,
    log_interval: int,
    mode: str,
    coordinator_address: str,
    xla_memory_fraction: float,
    num_processes: int,
) -> tuple[str, pathlib.Path]:
    log_dir = OUTPUT_ROOT / "logs" / config
    log_path = log_dir / f"{exp_name}_{node.host}.log"
    exit_path = log_dir / f"{exp_name}_{node.host}.exit"
    command = [
        str(PYTHON),
        "scripts/train.py",
        config,
        "--exp-name",
        exp_name,
        "--checkpoint-base-dir",
        str(OUTPUT_ROOT),
        "--num-train-steps",
        str(num_train_steps),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--log-interval",
        str(log_interval),
        f"--{mode}",
    ]
    exports = {
        "WANDB_MODE": "offline",
        "WANDB_SILENT": "true",
        "HF_HUB_OFFLINE": "1",
        "HUGGINGFACE_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "OPENPI_DATA_HOME": str(OPENPI_DATA_HOME),
        "HF_HOME": str(HF_HOME),
        "HUGGINGFACE_HUB_CACHE": str(HF_HOME / "hub"),
        "HF_DATASETS_CACHE": str(HF_HOME / "datasets"),
        "TRANSFORMERS_CACHE": str(HF_HOME / "transformers"),
        "HF_LEROBOT_HOME": str(DATASETS_ROOT),
        "CUDA_VISIBLE_DEVICES": "0,1",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": str(xla_memory_fraction),
        "JAX_COORDINATOR_ADDRESS": coordinator_address,
        "JAX_NUM_PROCESSES": str(num_processes),
        "JAX_PROCESS_ID": str(node.process_id),
        "JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT": "900",
    }
    if node.process_id == 0:
        coordinator_port = coordinator_address.rsplit(":", 1)[-1]
        exports["JAX_COORDINATOR_BIND_ADDRESS"] = f"0.0.0.0:{coordinator_port}"
    export_lines = [f"export {key}={shlex.quote(value)}" for key, value in exports.items()]
    script = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(REPO_ROOT))}",
            f"mkdir -p {shlex.quote(str(log_dir))}",
            f"rm -f {shlex.quote(str(exit_path))}",
            "unset WANDB_DISABLED",
            *export_lines,
            "set +e",
            f"{shlex.join(command)} 2>&1 | tee -a {shlex.quote(str(log_path))}",
            "rc=${PIPESTATUS[0]}",
            f"printf '%s\\n' \"$rc\" > {shlex.quote(str(exit_path))}",
            'exit "$rc"',
        ]
    )
    return script, log_path


def launch(
    *,
    config: str,
    exp_name: str,
    num_train_steps: int,
    batch_size: int,
    num_workers: int,
    log_interval: int,
    mode: str,
    session_prefix: str,
    coordinator_address: str,
    xla_memory_fraction: float,
    dry_run: bool,
    hosts: tuple[str, ...] = DEFAULT_HOSTS,
) -> dict:
    if mode not in {"overwrite", "resume"}:
        raise ValueError("mode must be overwrite or resume")
    if not hosts or len(set(hosts)) != len(hosts):
        raise ValueError("Training hosts must be non-empty and unique")
    nodes = tuple(Node(host, process_id) for process_id, host in enumerate(hosts))
    global_device_count = len(nodes) * 2
    if batch_size % global_device_count != 0:
        raise ValueError(f"Global batch size must be divisible by {global_device_count} devices")
    if num_train_steps <= 0 or num_workers < 0:
        raise ValueError("Training steps must be positive and workers non-negative")

    jobs = []
    for node in nodes:
        command, log_path = build_node_command(
            node,
            config=config,
            exp_name=exp_name,
            num_train_steps=num_train_steps,
            batch_size=batch_size,
            num_workers=num_workers,
            log_interval=log_interval,
            mode=mode,
            coordinator_address=coordinator_address,
            xla_memory_fraction=xla_memory_fraction,
            num_processes=len(nodes),
        )
        jobs.append(
            {
                "host": node.host,
                "process_id": node.process_id,
                "session": f"{session_prefix}_{node.host}",
                "log": str(log_path),
                "exit_marker": str(log_path.with_suffix(".exit")),
                "command": command,
            }
        )
    report = {
        "config": config,
        "exp_name": exp_name,
        "checkpoint_dir": str(OUTPUT_ROOT / config / exp_name),
        "global_batch_size": batch_size,
        "global_device_count": global_device_count,
        "hosts": list(hosts),
        "num_train_steps": num_train_steps,
        "coordinator_address": coordinator_address,
        "jobs": jobs,
    }
    if dry_run:
        return report

    started = []
    tmux_script = r"""
session="$1"
command="$2"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 2
fi
tmux new-session -d -s "$session" "$command"
"""
    try:
        for job in jobs:
            _ssh(
                job["host"],
                ["bash", "-s", "--", job["session"], job["command"]],
                input_text=tmux_script,
            )
            started.append(job)
    except Exception:
        for job in started:
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", job["host"], "tmux", "kill-session", "-t", job["session"]],
                check=False,
                capture_output=True,
            )
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--num-train-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=126)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--mode", choices=("overwrite", "resume"), default="overwrite")
    parser.add_argument("--session-prefix", required=True)
    parser.add_argument("--coordinator-address", default="172.31.11.112:12365")
    parser.add_argument("--xla-memory-fraction", type=float, default=0.90)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hosts", nargs="+", default=list(DEFAULT_HOSTS))
    args = parser.parse_args()
    report = launch(
        config=args.config,
        exp_name=args.exp_name,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
        mode=args.mode,
        session_prefix=args.session_prefix,
        coordinator_address=args.coordinator_address,
        xla_memory_fraction=args.xla_memory_fraction,
        dry_run=args.dry_run,
        hosts=tuple(args.hosts),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
