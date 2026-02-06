#!/usr/bin/env bash
set -euo pipefail

# Inspect latest openpi_docker training artifacts on remote host.
# Outputs JSON with:
# - training_ok
# - config_name / exp_name / run_tag
# - latest_step / policy_dir
# - repo_id / prompt_from_task / injected_default_prompt / prompt_mode

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_PROFILE="/home/lyj/.codex/skills/openpi_docker/references/server_profile.env"
PROFILE_FILE="${PROFILE_FILE:-$DEFAULT_PROFILE}"

load_profile() {
  local profile="$1"
  if [ -f "$profile" ]; then
    while IFS= read -r line; do
      line="${line%%#*}"
      line="${line%%$'\r'}"
      if [ -z "${line//[[:space:]]/}" ]; then
        continue
      fi
      if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
        local key="${line%%=*}"
        local value="${line#*=}"
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"
        if [ -z "${!key:-}" ]; then
          export "$key=$value"
        fi
      fi
    done < "$profile"
  fi
}

load_profile "$PROFILE_FILE"

: "${REMOTE_USER:?REMOTE_USER is required}"
: "${REMOTE_HOST:?REMOTE_HOST is required}"
: "${REMOTE_PORT:?REMOTE_PORT is required}"
: "${MONITOR_DIR:?MONITOR_DIR is required}"
: "${OPENPI_OUTPUT_DIR:?OPENPI_OUTPUT_DIR is required}"

ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" /bin/bash -s -- "$MONITOR_DIR" "$OPENPI_OUTPUT_DIR" <<'REMOTE_SCRIPT'
set -euo pipefail

MONITOR_DIR="$1"
OPENPI_OUTPUT_DIR="$2"

python3 - "$MONITOR_DIR" "$OPENPI_OUTPUT_DIR" <<'PY'
import json
import pathlib
import re
import sys

monitor_dir = pathlib.Path(sys.argv[1])
output_dir = pathlib.Path(sys.argv[2])

result: dict[str, object] = {
    "training_ok": False,
    "reason": "",
    "monitor_dir": str(monitor_dir),
    "openpi_output_dir": str(output_dir),
    "metrics_latest_path": str(monitor_dir / "metrics_latest.json"),
    "config_name": None,
    "exp_name": None,
    "run_tag": None,
    "log_path": None,
    "repo_id": None,
    "prompt_from_task": None,
    "injected_default_prompt": None,
    "prompt_mode": None,
    "latest_step": None,
    "policy_dir": None,
}

metrics_latest_path = monitor_dir / "metrics_latest.json"
if not metrics_latest_path.exists():
    result["reason"] = f"missing metrics file: {metrics_latest_path}"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1)

metrics_latest = json.loads(metrics_latest_path.read_text())
result["metrics_latest"] = metrics_latest

log_path_raw = str(metrics_latest.get("log_path") or "")
log_path: pathlib.Path | None = None
if log_path_raw.startswith("/monitor/"):
    log_path = monitor_dir / pathlib.Path(log_path_raw).relative_to("/monitor")
elif log_path_raw:
    cand = pathlib.Path(log_path_raw)
    if cand.exists():
        log_path = cand

if log_path is None or not log_path.exists():
    train_logs = sorted(monitor_dir.glob("metrics/*/*/*/train.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if train_logs:
        log_path = train_logs[0]

if log_path is None or not log_path.exists():
    result["reason"] = "cannot locate latest train.log"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1)

result["log_path"] = str(log_path)

path_match = re.search(r"/metrics/([^/]+)/([^/]+)/([^/]+)/train\.log$", str(log_path))
if path_match:
    config_name, exp_name, run_tag = path_match.groups()
    result["config_name"] = config_name
    result["exp_name"] = exp_name
    result["run_tag"] = run_tag

log_text = log_path.read_text(errors="ignore")
log_lines = log_text.splitlines()
tail_text = "\n".join(log_lines[-250:])

search_space = log_text

repo_id_match = re.search(r"data_config:\s*DataConfig\((?:.|\n)*?repo_id='([^']+)'", search_space)
if repo_id_match is None:
    repo_id_match = re.search(r"repo_id='([^']+)'", search_space)
if repo_id_match:
    result["repo_id"] = repo_id_match.group(1)

prompt_from_task_match = re.search(r"prompt_from_task=(True|False)", search_space)
if prompt_from_task_match:
    result["prompt_from_task"] = prompt_from_task_match.group(1) == "True"

inject_match = re.search(r"InjectDefaultPrompt\(prompt=([^)]+)\)", search_space)
if inject_match:
    injected = inject_match.group(1).strip()
    result["injected_default_prompt"] = injected

injected_default_prompt = result.get("injected_default_prompt")
if injected_default_prompt in (None, "None"):
    result["prompt_mode"] = "client_prompt_required"
else:
    result["prompt_mode"] = "default_prompt_available"

config_name = result.get("config_name")
exp_name = result.get("exp_name")
latest_step: int | None = None
policy_dir: pathlib.Path | None = None

if isinstance(config_name, str) and isinstance(exp_name, str):
    ckpt_root = output_dir / config_name / exp_name
    if ckpt_root.exists():
        steps = sorted(int(p.name) for p in ckpt_root.iterdir() if p.is_dir() and p.name.isdigit())
        if steps:
            latest_step = steps[-1]
            policy_dir = ckpt_root / str(latest_step)

if latest_step is not None and policy_dir is not None:
    result["latest_step"] = latest_step
    result["policy_dir"] = str(policy_dir)

has_traceback = bool(re.search(r"Traceback \(most recent call last\):", tail_text))
has_err_tag = "[err]" in tail_text

if latest_step is None:
    result["reason"] = "no numeric checkpoint step found"
elif has_traceback or has_err_tag:
    result["reason"] = "errors found near log tail"
else:
    result["training_ok"] = True
    result["reason"] = "checkpoint and metrics look valid"

print(json.dumps(result, ensure_ascii=False, indent=2))
PY
REMOTE_SCRIPT
