# CLAUDE.md

本仓库的项目规则由 `AGENTS.md` 负责，Context OS 热记忆由 `docs/cache/` 负责；不要把本文件当第二套记忆系统。

## 启动

按 `AGENTS.md` → `docs/cache/kernel.md` → `docs/cache/context_index.md` → 一个 mode pack 的顺序加载。接手项目从 `README.md` 和 `docs/00_handoff_index.md` 开始。

## 当前主线

默认是 OpenArm 双臂 T-shirt folding 的 OpenPI VLA 后训练和 rollout。OpenArm 合同、环境、训练、数据、推理、研究计划分别由 `docs/01`–`docs/06` 持有；`docs/07_change_log.md` 只记录历史原因和结果。Piper 仅保留 `docs/reference/legacy/piper.md` 一个入口，不是默认路径。

## 快速开发检查

```bash
ruff check .
ruff format .
conda run -n pi-conda python -m pytest --strict-markers -m "not manual"
git diff --check
```

代码改动前先检查 `git status --short`；OpenArm 只能使用 16D degree/HQ 夹爪合同和 OpenArm transforms。RTC 改动必须保留旧 `rtc_mode=off` 路径。完成后按 `AGENTS.md` 要求写回唯一 owner、`git add -A`、中文 commit，不 `git push`。
