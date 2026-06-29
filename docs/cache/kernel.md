# Context Kernel

New sessions: read `AGENTS.md` + this file first, then `context_index.md` to select <=1 mode pack.

## Default Facts
- Default product: OpenPI VLA fine-tuning and inference for Piper/OpenArm dual-arm workflows.
- Local workspace: `/home/lyj/lyj/openpi`
- Default training access: mu01 jump `linyongjia@172.31.11.100:12222`, then `ssh -p 12222 gpu12/gpu14`; legacy docs may mention `172.31.11.122`.
- Training nodes: gpu08 (`172.31.11.108`), gpu12 (`172.31.11.112`), and gpu14 (`172.31.11.114` via mu01 `172.31.11.100`), 2x A800 each, 6 GPUs total.
- Remote code path: `/share/home/linyongjia/conda-pi/openpi`
- Remote datasets: `/share/home/linyongjia/data/`; legacy docs may mention `/share/home/linyongjia/datasets/`.
- Remote checkpoints/output: `/share/home/linyongjia/output/openpi`
- Remote cache base: `/share/home/linyongjia/.cache/openpi`
- Default training target: `pi05_piper_dual`; current OpenArm configs include `pi05_openarms_dual` and `pi05_openarms_dual_hq`.
- Default env: `pi-conda` (conda, Python 3.11, CUDA 12), server conda at `/share/home/linyongjia/miniconda3/bin/conda`
- Default entry: `scripts/train.py` (JAX), `scripts/train_pytorch.py` (PyTorch)
- Multi-node JAX training requires early env init: `JAX_COORDINATOR_ADDRESS`, coordinator `JAX_COORDINATOR_BIND_ADDRESS`, `JAX_NUM_PROCESSES`, `JAX_PROCESS_ID`; only process 0 writes wandb/metrics.
- Model checkpoints (GCS): `gs://openpi-assets/checkpoints/` (pi0_base, pi05_base, pi0_fast_base, etc.)
- Default pipeline: choose gpu08/gpu12/gpu14 -> dataset under remote data root -> `local/<alias>` symlink -> norm stats -> training -> offline evaluation.
- `norm_stats.json` must exist under `assets/<config>/local/<alias>/` before training.
- OpenPI does not auto-read LeRobot `info.json` splits; set `DataConfig.train_episodes` explicitly when a split matters.

## Security Kernel
- Do not commit credentials, tokens, private host keys, or server passwords.
- Training runs use offline W&B (`WANDB_MODE=offline`) and Hugging Face offline mode.
- Do not delete remote checkpoint, output, cache, or dataset directories unless the user explicitly asks.

## Context Loading
- Default: `AGENTS.md` + this file
- Route: open `context_index.md` only when selecting task context
- Mode pack: open <=1 `docs/cache/modes/*.md`; only add a second if the task truly spans modes

## Budget & Anti-Proliferation
- `kernel.md`: <=80 lines
- `context_index.md`: <=100 lines
- Each mode pack: <=80 lines
- No new cache/memory file unless no existing owner can hold the fact
- Before adding hot context: compress or demote to cold docs first
- One stable fact -> one primary owner

## Writeback Classes
- `critical`: default chain, entry points, package boundaries, public interfaces, security boundaries, deployment baseline, context loading strategy. Update owner docs same turn.
- `incident`: training failures, deploy errors, hardware warnings. Write to CHANGELOG.
- `batch`: repeated training runs, verification results. Condense into one history entry after sequence ends.
- `ephemeral`: status reads, one-off manual actions, temporary exploration. Default: don't update memory.
- `artifact`: checkpoints, norm stats, metrics plots, wandb logs. Store in output dir; reference from cold docs only if useful.

## Memory Audit (before saving)
1. Has a stable default fact changed?
2. Which file uniquely owns this fact?
3. Which writeback class applies?
4. Is any cache file over budget?
5. Should this be compressed, demoted, or not written?

## Response Protocol
1. Give the default conclusion first.
2. Note context ownership when relevant.
3. Distinguish: implemented / optional / historical / not-implemented behavior.
4. Provide precise file entry points, commands, verification, and memory writeback conclusions.
