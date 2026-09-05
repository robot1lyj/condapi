# 00 · Context Kernel

启动顺序：`AGENTS.md` → 本文件 → `docs/cache/context_index.md` → 与任务匹配的一个 mode。这里只放高频且影响安全的事实；详细说明由编号化 `docs/` 持有。

## 当前默认

- 项目：OpenPI 对 YAM 双臂的训练适配；YAM 与 YAM-ABC 同硬件配置。
- 当前首选：Pi0.5 LoRA；默认配置 `pi05_yam_lora`，训练重点是 LeRobot 数据、norm stats、loader smoke 和 checkpoint gate。
- 运行边界：训练仍在服务器 GPU；训练后推理默认部署到 NVIDIA Jetson AGX Thor 端侧，远程 WebSocket 只保留兼容/调试回退。
- Thor 系统基线：JetPack 7.2.1 / L4T r39.2.1 / Ubuntu 24.04 / Kernel 6.8 / CUDA 13.2.1 / TensorRT 10.16.2；实际设备到手后必须现场核验 SKU 和版本。
- Pi0.5 Thor 路线：JAX/Flax checkpoint 是权威源和 golden reference，首选 JAX→PyTorch BF16→TensorRT FP8，验证通过后才试 NVFP4；不把直接 JAX runtime 作为首个默认路径。
- YAM 双臂 state/action 为 14D：`[左臂6关节, 左夹爪, 右臂6关节, 右夹爪]`。
- 数据键：`observation.state`、`action`、`observation.images.top_rgb`、`observation.images.left_rgb`、`observation.images.right_rgb`。
- 训练默认 mask 为 `(6,-1,6,-1)`：关节相对当前 state，夹爪绝对；模型内部 32D、horizon 50，输出裁回 14D。
- YAM 数据的物理单位以 metadata/audit 为准；禁止把 OpenArm 16D、OpenArm degree/HQ 夹爪或 Piper transform 套到 YAM。

## 运行资源

- 本地仓库：`/home/wuyan-lyj/condapi`。
- 外部只读参考：`/home/wuyan-lyj/YAM`；不把其中独立的 YAM-ABC-Reproduce 代码作为本项目实现。
- Thor 系统盘 ISO 已下载到 `/home/wuyan-lyj/thor-system/jetpack-7.2.1/`；系统盘制作、实机刷写和端侧 smoke 仍属计划中，详见 `docs/08_thor_edge_deployment.md`。
- 新平台 SSH：`wuyan@10.18.31.234:22`；实测登录主机 `rocky-login.hlink.local`；工作台 `http://10.18.31.233:3080/`。
- 服务器工作区：`/home/wuyan/lyj/YAM`；数据 `YAM_data`；当前项目代码目标 `YAM_code`。
- 环境入口：`module load miniconda3/26.1.1`，再 `conda activate /home/wuyan/.conda/envs/condapi-yam`。
- 服务器数据目录：`/home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks`；先作为原始/待审计数据，不可直接训练或覆盖。
- 接管时唯一核实的任务是 Slurm `1962/abc-download`（`gpu001`，下载任务仍在运行）；不可停止、删除或抢占。

## Git 备份

- `origin`：Gitea `http://192.168.110.142:3000/wuyan_lyj/conda_pi.git`。
- `github`：GitHub `https://github.com/robot1lyj/condapi.git`。
- Git 身份：用户名 `wuyan_lyj`，邮箱 `linyongjia@wuyanai.cn`；本地仓库提交和服务器 Gitea 同步均使用该身份。
- 同步边界：本地工作站将提交推送到 Gitea `origin/main` 和 GitHub `github/main` 互为备份；服务器无法连接 GitHub，服务器只从 Gitea `origin` 拉取/同步代码，不执行 GitHub push/pull。
- 服务器登记 Gitea 的公钥正文在 `docs/02_installation_and_environment.md` 第 5 节；不要与本机登录服务器的 ED25519 公钥混淆。
- 不 force push，不把凭据写入 URL/历史；服务器公钥登记到 Gitea 后优先使用 SSH/非交互认证。

## 安全闸门

- 不提交密码、token、私钥；只可向用户提供服务器已有 SSH 公钥。
- 长任务用 Slurm 或 tmux；不在登录节点训练；不删除远端数据、权重、缓存。
- 训练前核对 YAM metadata、三路视频、14D state/action、action 单位、task/prompt、norm stats 和样本首中尾。
- Thor 本地或远程兼容 smoke 都必须确认有限 `(50,14)` YAM 动作、模型 32D metadata、checkpoint/norm 绑定；社区 LIBERO 的 horizon 10 性能不能直接外推到 YAM horizon 50。

## 记忆写回

- 默认/路径/合同改变 → 更新本文件和对应 canonical doc。
- 新故障或已完成结果 → `docs/07_change_log.md`；不把一次性状态伪装成稳定事实。
- 恢复审计先确认 `pwd`、`git status --short`、最近提交、相关远端和 Slurm 状态。
