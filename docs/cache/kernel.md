# Context Kernel

New sessions load `AGENTS.md` + this file first, then `context_index.md` and at most one mode pack.

## Defaults
- Product: OpenPI VLA fine-tuning/inference for OpenArm cloth folding; Piper configs remain separate legacy/support paths.
- Local workspace: `/home/lyj/lyj/openpi`
- Remote access: `ssh -p 12222 linyongjia@172.31.11.100`, then `ssh gpu12/gpu14/gpu25/gpu28`.
- Main training nodes: gpu12/gpu14, each 2x A800 80GB. gpu25 usually serves policy on port `6666`; gpu28 is eval/aux.
- Remote repo/env/output: `/share/home/linyongjia/conda-pi/openpi`, env `pi-conda`, output `/share/home/linyongjia/output/openpi`.
- Remote datasets: policy datasets under `/share/home/linyongjia/datasets`; some Stage/reference data may live under `/share/home/linyongjia/data`.
- Main OpenArm configs: `pi05_openarms_dual_site_align_v1_probe`, `pi05_openarms_dual_evo_acp_hil_v1_probe`, and formal KAI0 `pi05_openarm_kai0_awbc_v1`; formal K-Policy must not reuse legacy `pi05_openarms_dual_awbc_v1`.

## OpenArm Contract
- Task prompt: `Fold the T-shirt properly`.
- State/action: 16D `[右臂7关节, 右夹爪, 左臂7关节, 左夹爪]`.
- Units: arm joints degrees; gripper HQ motor degrees, `0=open`, `-66=closed`. ROS/runtime conversion stays at client boundary.
- OpenArm configs use `LeRobotOpenArmDataConfig` with `OpenArmInputs/OpenArmOutputs`; never route OpenArm through Piper transforms or Piper 14D `swap_left_right`.
- Clean site/HIL data only through `scripts/convert_openarm_hq_dataset.py`; HIL clean export drops hold frames and marks only real human VR as intervention.
- Evo-RL ACP uses `ACPPromptTransform` on `complementary_info.acp_indicator`; clean HIL dataset name is `openarm_hil_evo_v1`.
- KAI0 Stage boundary: HQ-Stage scores HQ, then directly scores Site-A150 for a transfer audit; only if that fails do 140 train + 10 val annotations adapt Site-Stage. Site-F1 is excluded, final Site-Score must be model-predicted, and linear Site-GT was deleted.
- KAI0 AWBC stage groups do not replace model advantage: Site uses manual `flatten_done`; HQ `0:536` uses model crossing and folding-only `536:999` is stage 1; TDA inherits its HQ source stage.
- K-Policy serving must force `Fold the T-shirt properly, Advantage: positive`; a default prompt is insufficient when clients send their own prompt.
- KAI0 pipeline controller is jump-host tmux `kai0_pipeline_v1`; canonical status is `output/openpi/logs/openarm_kai0_pipeline_v1/status.json`.

## Safety Kernel
- Do not commit credentials, tokens, private host keys, or server passwords.
- Do not delete remote datasets/checkpoints/caches unless the user explicitly asks.
- Training/serve runs use tmux and offline-friendly W&B/Hugging Face settings.
- OpenPI does not auto-use LeRobot split intent; set `DataConfig.train_episodes` when a split matters.

## Context Loading
- `docs/cache/context_index.md` routes only; it must not store facts.
- `docs/cache/modes/code_change.md`: code/config/docs changes, tests, commits.
- `docs/cache/modes/deployment.md`: SSH, conda, remote train/serve, artifacts.
- Current OpenArm KAI0 / Evo-RL / hybrid plan: `docs/openarm_recap_reproduction_plan.md`.

## Budget And Writeback
- Budgets: `kernel.md` <=80 lines, `context_index.md` <=100, each mode <=80.
- One stable fact has one owner; compress or demote before adding memory.
- `critical`: default machine/env/model/data contract/boundary changed -> update owning hot/canonical doc same turn.
- `incident`: training/deploy/hardware failure -> `docs/CHANGELOG.md` or a short canonical note.
- `batch`: repeated runs/validations -> one concise history entry after sequence ends.
- `ephemeral`: status reads and one-off checks -> no memory update.

## Resume Audit
- After compact/resume/interruption: confirm latest user request, `pwd`, git root, `AGENTS.md`, this file, `git status --short`, `git log --oneline -5`, and the relevant mode pack.
- Trust committed files, artifacts, explicit checkpoints, and current repo docs over conversation memory.

## Answer Protocol
- Lead with the current conclusion.
- Distinguish implemented, planned, historical, and forbidden behavior.
- Give precise files/commands/verification when relevant.
- End with memory writeback/verification status for substantial work.
