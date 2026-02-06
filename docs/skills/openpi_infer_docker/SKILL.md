---
name: openpi_infer_docker
description: Reuse openpi_docker training outputs to locate the latest valid checkpoint, verify prompt behavior from code, start Docker-based policy serving, and confirm inference service health.
---

# OpenPI Inference Docker (Piper)

## Trigger phrases
- "openpi_infer_docker"
- "openpi inference docker"
- "piper inference docker"

## Depends on openpi_docker artifacts
This skill assumes training used `openpi_docker` and wrote monitor artifacts:
- `MONITOR_DIR/metrics_latest.json`
- `MONITOR_DIR/metrics/<config>/<exp>/<run_tag>/train.log`
- `OPENPI_OUTPUT_DIR/<config>/<exp>/<step>`

## Server profile
By default, scripts load:
`/home/lyj/.codex/skills/openpi_docker/references/server_profile.env`

You can override with `PROFILE_FILE=/path/to/server_profile.env`.

## Prompt behavior (verified from code)
- `serve_policy.py` passes `--default-prompt` into policy creation (`scripts/serve_policy.py`).
- If request payload contains `prompt`, Piper inference uses it (`src/openpi/policies/piper_policy.py`).
- If request payload has no `prompt` and no default prompt is injected, tokenization fails with `Prompt is required` (`src/openpi/transforms.py`).

For your setup ("client always sends prompt"), run without `--default-prompt` unless you need fallback behavior.

## Workflow
1) Inspect latest training status and extract inference parameters
```bash
bash docs/skills/openpi_infer_docker/scripts/inspect_training_remote.sh
```

2) Start inference server in remote `openpi_dev` container
```bash
PORT=6666 RTC_MODE=off \
bash docs/skills/openpi_infer_docker/scripts/start_infer_remote.sh
```

Optional fallback prompt:
```bash
PORT=6666 DEFAULT_PROMPT="put the towel on the table" \
bash docs/skills/openpi_infer_docker/scripts/start_infer_remote.sh
```

3) Verify inference is really running
```bash
PORT=6666 \
bash docs/skills/openpi_infer_docker/scripts/check_infer_remote.sh
```

Success criteria:
- target container exists and is running
- `serve_policy.py --port <PORT>` process exists
- container-local `http://127.0.0.1:<PORT>/healthz` returns `OK`

## Environment variables
- `PORT` default `6666`
- `RTC_MODE` default `off`
- `DEFAULT_PROMPT` optional
- `POLICY_REPO_ID` optional; if unset, auto-detected from latest training log
- `CONFIG_NAME`/`EXP_NAME`/`STEP` optional manual override
- `CONTAINER_NAME` default `openpi_dev`
- `FORCE_RESTART` default `1` (kill existing same-port server before start)
