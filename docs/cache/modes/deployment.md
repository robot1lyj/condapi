# Mode: Deployment

Use for SSH, conda env, remote train/serve, GPU status, logs, and artifact promotion.

## Defaults
- Jump: `ssh -p 12222 linyongjia@172.31.11.100`
- Current K-Policy nodes: `gpu12,gpu28` (4 cards, global batch128, workers8); node set is runtime-configurable. `gpu25` is single-card serve.
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
- Audit candidate nodes before launch; current formal run is the indivisible gpu12+gpu28 pair. Avoid stealing gpu25 if it is serving.
- Python SSH launchers must pass one `shlex.join(remote_argv)` command string to OpenSSH; raw trailing argv does not preserve boundaries through the remote login shell when arguments contain `&&`, redirects, spaces, or newlines.

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
- HQ finalization must pass `scripts/audit_openarm_hq_stage_scores.py`; K-Data must not rely only on parquet counts or report visuals.
- HQ watchdog session is `kai0_hq_score_monitor`; it may recover a stopped/stalled worker up to three times and writes status under `output/openpi/logs/openarm_kai0_stage_scores_hq_v1`.
- Site supervisor session is `kai0_site_score_monitor`; it starts a balanced Site shard when the matching HQ GPU slot is free, then writes the transfer audit under `datasets/openarm_site_score_review_v1`.
- End-to-end KAI0 supervisor is `kai0_pipeline_v1` on the jump host; status is `output/openpi/logs/openarm_kai0_pipeline_v1/status.json`. It may start Site-Stage only after a failed direct-transfer gate, then K-Data, norm, multi-node smoke, 80k, sweep, and gpu25 deployment in order.
- Before the multi-node smoke, formal K-Data must pass `scripts/audit_openarm_kai0_training_data.py`; this checks all binary labels/source counts, real positive/negative OpenPI loader samples, and every TDA episode tail through the TorchCodec-to-PyAV fallback.
- OpenArm parquet norm stats are written through an atomic temporary file; never treat a partially written JSON as recoverable training input.
- Formal K-Policy sessions are one distributed unit over the configured hosts; current run is gpu12+gpu28. Never restart only part of a JAX job.
- Formal JAX loaders reshuffle deterministically per dataset epoch and derive the resume epoch/batch offset from restored `train_state.step`; do not replace this with a fixed `DistributedSampler` epoch or restart data at batch zero.
- Formal K-Policy video loading uses TorchCodec normally and retries only its explicit end-of-stream tail error with PyAV/0.05s; full-dataset PyAV is correct but too slow for 80k.
- Treat a JAX checkpoint as ready only when Orbax `_CHECKPOINT_METADATA` and `params/_METADATA` exist; a numeric directory or `params/` alone may still be an asynchronous partial save.
- Formal K-Policy deployment uses `serve_policy.py --force-prompt 'Fold the T-shirt properly, Advantage: positive'`; this intentionally overrides client prompts only for this conditioned policy.
- Deployment success additionally requires `smoke_test_openarm_policy_server.py` to return finite `(50,16)` actions, validate 50-step/16D/degrees/HQ-gripper `0/-66` metadata, and bind evidence to the selected checkpoint; listening on `6666` is insufficient.
- HQ/Site policy sweep runs with `--resume`; reuse is allowed only for atomic v2 reports with identical checkpoint, dataset, sampled episodes/settings, config, and forced positive prompt.
- After deployment, the controller builds the self-contained K-Policy report at `policy_report/index.html` under the pipeline log root and serves it from gpu28 port `8769`; completion requires that server to listen.
- When remote code is behind local commits, sync the full `git ls-files` set; selectively copying `training/config.py` can omit new policy/transform dependencies. Validate Torch/OpenPI imports inside gpu12/gpu14/gpu28, not on the older-glibc jump host.
- The reusable report server supports byte ranges; current HQ report is `openarm_hq_score_review_v1/hq_score_report/index.html` on gpu28 port 8767.
