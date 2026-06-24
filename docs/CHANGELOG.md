# openpi Change Log

Record **what changed + why**. Not config values, verification steps, or impact scope —
those live in config files, git log, and CI respectively.

Design decisions → `docs/decisions/`. Architecture → `docs/ARCHITECTURE.md`.

---

## 2026-06-24

- **Context OS 记忆系统初始化**: 建立四层 Context OS 记忆架构 (CLAUDE.md → cache/ → memory/ → CHANGELOG)，将服务器地址、远端路径、GCS 模型位置等核心事实固化到跨会话记忆中。
- **训练服务器 IP 更新**: `172.31.11.122` → `172.31.11.108` (gpu08)，所有引用已同步更新。

## 2026-06 (近期)

- **离线 WandB 支持**: 训练强制使用离线 W&B 模式，同时生成本地 metrics.jsonl/metrics.csv/训练曲线图。
- **Piper 双臂 Conda 训练**: `conda-pi` 分支建立非容器 conda 环境的离线训练路径，支持 `pi05_piper_dual` 配置。
