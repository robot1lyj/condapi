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
- New current training/serve workflow -> this file and, if OpenArm-specific, `docs/openarm_recap_reproduction_plan.md`.
- Incidents or meaningful failures -> `docs/CHANGELOG.md`.

## Stage Scoring

- Long HQ/Site score-only runs must use episode-atomic outputs and `--resume`; never restart with `--overwrite` after progress exists.
- HQ watchdog session is `kai0_hq_score_monitor`; it may recover a stopped/stalled worker up to three times and writes status under `output/openpi/logs/openarm_kai0_stage_scores_hq_v1`.
- Site supervisor session is `kai0_site_score_monitor`; it starts a balanced Site shard when the matching HQ GPU slot is free, then writes the transfer audit under `datasets/openarm_site_score_review_v1`.
- End-to-end KAI0 supervisor is `kai0_pipeline_v1` on the jump host; status is `output/openpi/logs/openarm_kai0_pipeline_v1/status.json`. It may start Site-Stage only after a failed direct-transfer gate, then K-Data, norm, 4-GPU smoke, 80k, sweep, and gpu25 deployment in order.
- Formal K-Policy multi-node sessions are paired: `kai0_k_smoke_gpu12/gpu14` and `kai0_k_full_gpu12/gpu14`. Never restart only one JAX process.
- The reusable report server supports byte ranges; current HQ report is `openarm_hq_score_review_v1/hq_score_report/index.html` on gpu28 port 8767.
