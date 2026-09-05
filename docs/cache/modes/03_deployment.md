# 03 · Mode — Deployment

用于 SSH、conda、远端训练/服务、GPU、checkpoint 交接和最小 policy smoke。详细命令由 `docs/02_installation_and_environment.md`、`docs/03_training_and_evaluation.md` 和 `docs/05_inference_and_rollout.md` 持有。

## 固定边界

- SSH 为 `wuyan@10.18.31.234:22`；先加载 `miniconda3/26.1.1`，再激活项目环境 `/home/wuyan/.conda/envs/condapi-yam`。
- 当前代码目标是 `/home/wuyan/lyj/YAM/YAM_code`，数据目标是 `/home/wuyan/lyj/YAM/YAM_data`；外部 YAM-ABC 代码不作为本项目依赖。
- 长任务使用 Slurm 作业或 tmux，不在登录节点训练；GPU、分区和输出目录每次实时核验。
- 接管时的 `1962/abc-download` 下载任务不可停止、删除或抢占；不删除任何远端数据/权重/缓存。

## YAM 训练闸门

- 先审计 LeRobot metadata、三路图像、14D state/action、task/prompt、视频首中尾和 norm stats。
- `pi05_yam_lora` 是低显存首选；模型内部 action 是 32D/50 步，YAM 输出合同是 14D。
- checkpoint 必须有完整参数元数据和 `assets/yam/norm_stats.json`，之后才做服务 smoke。
- 端口监听不等于推理可用；真实 WebSocket smoke 必须验证 `(50,14)`、有限值、YAM metadata 和 checkpoint 绑定。

## Git 写回

- 每次提交后同时推送 `origin`（Gitea）和 `github`（GitHub），核对两个 `main` 指向同一 commit。
- 不 force push，不在 remote URL、脚本或日志中写密码/token。
