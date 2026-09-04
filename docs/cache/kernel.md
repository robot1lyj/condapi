# 00 · Context Kernel

启动顺序：`AGENTS.md` → 本文件 → `docs/cache/context_index.md` → 与任务匹配的一个 mode。这里只放高频、会改变操作安全的事实；详细说明由编号化 `docs/` 持有。

## 当前默认

- 产品：OpenPI VLA 在 OpenArm 双臂 T-shirt folding 上的后训练和 rollout。
- 任务 prompt：`Fold the T-shirt properly`；K-Policy 服务需强制 `Fold the T-shirt properly, Advantage: positive`。
- OpenArm state/action 为 16D `[右臂7关节, 右夹爪, 左臂7关节, 左夹爪]`；训练单位为 degree，HQ 夹爪 `0=open,-66=closed`。
- OpenArm 只走 `LeRobotOpenArmDataConfig`、`OpenArmInputs/Outputs`；Piper 是 legacy，不是默认路径。
- policy server 不定义 OpenArm 初始/复位位姿；旧记录中的 `/home/lyj/openarm_ros2_docker` 在新平台尚未核实，机器人 ROS/client `reset()` 主机和路径待单独确认，细节只看 `docs/05_inference_and_rollout.md`。

## 运行资源

- 本地仓库：`/home/wuyan-lyj/condapi`。
- 新平台 SSH：`wuyan@10.18.31.234:22`；2026-09-04 实测登录主机为 `rocky-login.hlink.local`；工作台入口为 `http://10.18.31.233:3080/`（按手册需内网 HTTP 代理）。
- 新服务器工作区：`/home/wuyan/lyj/YAM`；数据：`/home/wuyan/lyj/YAM/YAM_data`；代码入口：`/home/wuyan/lyj/YAM/YAM_code`（接管时为空，尚未同步 OpenPI）。
- 环境入口：`module load miniconda3/26.1.1`，再 `conda activate /home/wuyan/.conda/envs/yam`；接管时环境 Python 为 3.13.12，OpenPI 所需 Python 3.11/完整依赖尚未在计算节点验证。
- 接管时唯一核实的运行任务是 Slurm `1962/abc-download`（`gpu001`，仍在运行）；旧训练服务器、gpu25/6666、旧 checkpoint 和旧路径均只属于历史快照，不能作为新平台默认值。

## 当前路线

- KAI0/K-Policy 的完整边界和 HIL-T30、Evo-RL、Hybrid 下一步只看 `docs/06_openarm_research_plan.md`。
- HIL-T30 计划为三类各 10 条：错误对角线恢复、重复甩平恢复、已展开但未折叠接管；计划不等于已采集。
- 现有 Evo ACP probe 的 dropout `0.0` 不是正式目标；正式路线目标为 `0.3`，E-Value/train/infer 完成前不得宣称 Evo-RL 完成。

## 安全闸门

- 长任务必须用 tmux；不删除远端数据、权重或缓存，不提交凭据。
- 训练前核对 `meta/info.json`、视频可解码性、norm stats、16D/单位/夹爪范围和 prompt。
- checkpoint 必须有完整 Orbax 元数据；服务必须通过真实 WebSocket smoke，确认有限 `(50,16)` 动作和元数据，端口监听本身不算成功。
- RTC 改动必须保留旧路径，并支持 `rtc_mode=off` 或自动回退。

## 记忆写回

- 默认、路径、合同改变 → 更新本文件和对应 canonical doc。
- 新训练/部署故障 → `docs/07_change_log.md`；连续实验结束后只写一条摘要。
- 一次性状态不写记忆。一个事实只能有一个 owner；index 只路由，不存事实。

## 恢复审计

压缩/中断后重新确认：`pwd`、`git status --short`、最近提交、`AGENTS.md`、本文件、相关 mode，以及当前远端 checkpoint/服务状态。

## 回答约定

先给当前结论，区分已实现、已验证、计划中和禁止项；涉及实验证据时给出文件/命令，结尾说明文档写回与验证结果。
