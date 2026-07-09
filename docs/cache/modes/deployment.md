# Mode: Deployment

Use for SSH, conda env, remote train/serve, GPU status, logs, and artifact promotion.

## Defaults
- Jump: `ssh -p 12222 linyongjia@172.31.11.100`
- Nodes: `gpu12`, `gpu14` for training; `gpu25` for serve; `gpu28` for eval/aux.
- Remote repo: `/share/home/linyongjia/conda-pi/openpi`
- Remote env: `/share/home/linyongjia/miniconda3/envs/pi-conda`
- Dataset root: `/share/home/linyongjia/datasets`
- Output root: `/share/home/linyongjia/output/openpi`
- Cache: `OPENPI_DATA_HOME=/share/home/linyongjia/.cache/openpi`

## Remote Train
Prefer the project skill helper for norm-stats-then-train runs:

```bash
bash ~/.codex/skills/openpi-conda-remote-train/scripts/start_remote_train.sh \
  --node gpu12 \
  --dataset-name <dataset_dir_under_datasets> \
  --config <openpi_config> \
  --exp-name <run_name> \
  --num-train-steps <steps>
```

Rules:

- Use tmux for long runs.
- Use `pi-conda`; do not use uv.
- Verify dataset `meta/info.json` and `norm_stats.json` before policy training.
- For OpenArm LeRobot v2.1, prefer `scripts/compute_openarm_parquet_norm_stats.py`.
- Use gpu12/gpu14 for training; avoid stealing gpu25 if it is serving.

## Serve

```bash
cd /share/home/linyongjia/conda-pi/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
/share/home/linyongjia/miniconda3/envs/pi-conda/bin/python scripts/serve_policy.py \
  --port 6666 policy:checkpoint \
  --policy.config=<config> \
  --policy.dir=<checkpoint_dir>
```

`--port` must appear before `policy:checkpoint`.

## Local Env

Local validation env is `pi-conda`. If missing, create with conda from `environment.pi-conda.yml`; install project editable; use CPU torch wheels locally if CUDA runtime libraries are absent. Keep `pip check` clean.

## Writeback
- Default node/path/env changes -> `docs/cache/kernel.md`.
- New current training/serve workflow -> this file and, if OpenArm-specific, `docs/openarm_kai0_reproduction_plan.md`.
- Incidents or meaningful failures -> `docs/CHANGELOG.md`.
