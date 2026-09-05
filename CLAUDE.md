# CLAUDE.md

本仓库的项目规则由 `AGENTS.md` 负责，Context OS 热记忆由 `docs/cache/` 负责；不要把本文件当第二套记忆系统。

## 启动

按 `AGENTS.md` → `docs/cache/kernel.md` → `docs/cache/context_index.md` → 一个 mode pack 的顺序加载。接手项目从 `README.md` 和 `docs/00_handoff_index.md` 开始。

## 当前主线

默认是 YAM 双臂（与 YAM-ABC 同硬件配置）的 Pi0.5 LoRA 后训练，任务顺序为乐高分拣、再进入 DAgger。YAM 合同、环境、训练、数据、推理分别由 `docs/01`–`docs/05` 持有；`docs/06_openarm_research_plan.md` 仅是历史占位；`docs/07_change_log.md` 只记录历史原因和结果。OpenArm、Piper 和独立 `yam-abc-reproduce` 代码不是本项目默认路径。

## 快速开发检查

```bash
module load miniconda3/26.1.1
conda activate /home/wuyan/.conda/envs/condapi-yam
ruff check .
ruff format .
python -m pytest --strict-markers -m "not manual"
git diff --check
```

代码改动前先检查 `git status --short`；YAM 训练合同是 14D `[左臂6, 左夹爪, 右臂6, 右夹爪]`，只能使用 YAM transforms。RTC 改动必须保留旧 `rtc_mode=off` 路径。完成后按 `AGENTS.md` 写回唯一 owner、`git add -A`、中文 commit，并把同一提交推送到 Gitea `origin` 和 GitHub `github`。
