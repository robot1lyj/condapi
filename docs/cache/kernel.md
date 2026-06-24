# Context Kernel

New sessions: read `AGENTS.md` + this file first, then `context_index.md` to select ≤1 mode pack.

## Default Facts
- Default product: openpi VLA 模型微调与推理 (Piper 双臂)
- Local workspace: `/home/lyj/lyj/openpi`
- Training server: `linyongjia@172.31.11.108` (SSH port 12222)
- Remote code path: `/share/home/linyongjia/conda-pi/openpi`
- Remote datasets: `/share/home/linyongjia/datasets/`
- Remote checkpoints/output: `/share/home/linyongjia/output/openpi`
- Remote cache base: `/share/home/linyongjia/.cache/openpi`
- Default training target: `pi05_piper_dual`
- Default env: `pi-conda` (conda, Python 3.11, CUDA 12)
- Default entry: `scripts/train.py` (JAX), `scripts/train_pytorch.py` (PyTorch)
- Model checkpoints (GCS): `gs://openpi-assets/checkpoints/` (pi0_base, pi05_base, pi0_fast_base, etc.)
- Default pipeline: 数据 → 归一化统计 → 训练 (conda run -n pi-conda python scripts/train.py <config> --exp-name=<name>)

## Security Kernel
- 训练服务器 172.31.11.108 是核心资产，不要在 commit 中暴露敏感凭证
- 离线 WandB 模式不得绕过 (WANDB_MODE=offline)
- HF/HuggingFace 强制离线 (HF_HUB_OFFLINE=1)
- 不要删除远端服务器的 checkpoint 目录或数据集目录

## Context Loading
- Default: `AGENTS.md` + this file
- Route: open `context_index.md` only when selecting task context
- Mode pack: open ≤1 `docs/cache/modes/*.md`; only add a second if the task truly spans modes

## Budget & Anti-Proliferation
- `kernel.md`: ≤80 lines
- `context_index.md`: ≤100 lines
- Each mode pack: ≤80 lines
- No new cache/memory file unless no existing owner can hold the fact
- Before adding hot context: compress or demote to cold docs first
- One stable fact → one primary owner

## Writeback Classes
- `critical`: default chain, entry points, package boundaries, public interfaces, security boundaries, deployment baseline, context loading strategy. ← Update owner docs same turn.
- `incident`: training failures, deploy errors, hardware warnings. ← Write to CHANGELOG.
- `batch`: repeated training runs, verification results. ← Condensed single line in history after sequence ends.
- `ephemeral`: status reads, one-off manual actions, temporary exploration. ← Default: don't update memory.
- `artifact`: checkpoints, norm stats, metrics plots, wandb logs. ← Store in output dir; reference from cold docs only if useful.

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
