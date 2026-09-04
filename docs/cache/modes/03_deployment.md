# 03 · Mode — Deployment

用于 SSH、离线 conda、远端训练/服务、checkpoint 交接和真机 rollout。详细命令与合同分别由 `docs/02_installation_and_environment.md`、`docs/03_training_and_evaluation.md`、`docs/05_inference_and_rollout.md` 持有。

## 固定边界

- 新平台 SSH 为 `wuyan@10.18.31.234:22`；长任务使用 Slurm 作业或 tmux，环境先 `module load miniconda3/26.1.1` 再激活 `/home/wuyan/.conda/envs/yam`，不要使用 uv，也不要把登录节点当 GPU 训练节点。
- 新平台仓库、数据和输出路径见 `docs/02_installation_and_environment.md`，启动前以远端实时检查为准；旧 `/share/home/...`、`pi-conda` 和 gpu25 profile 不再是默认值。
- GPU 节点需要先审计 Slurm 分配和占用；接管时的 `1962/abc-download` 下载任务不可抢占、停止或删除。
- 正式多节点 JAX 作业是不可拆分的整体；新平台多节点拓扑尚未核实，不得直接复用旧 coordinator。

## 训练与服务闸门

- 训练前检查 `meta/info.json`、视频（含尾帧）、norm stats、episode split 和 OpenArm 16D 合同。
- checkpoint 只有在 Orbax 元数据完整时才可部署；数字目录或单独 `params/` 可能仍是异步保存中间态。
- `serve_policy.py` 的 `--port` 放在 `policy:checkpoint` 前；新平台服务节点/端口尚未核实，OpenArm K-Policy 仍使用强制 positive prompt。
- 端口监听不是成功条件；真实 WebSocket smoke 必须返回有限 `(50,16)`，并验证 horizon、单位、夹爪范围和 checkpoint 绑定。

## 安全与写回

- 真机先低风险 rollout，保留可回退 checkpoint；不删除远端数据、权重、缓存或日志。
- 新的默认节点/路径/服务行为 → 对应 canonical doc 和 kernel；事故/结果 → `docs/07_change_log.md`。
- RTC 必须保留 `rtc_mode=off` 旧路径或自动回退。
