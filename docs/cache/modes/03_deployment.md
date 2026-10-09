# 03 · Mode — Deployment

仅在远端环境、训练作业、checkpoint 交接或 Thor 服务操作需要额外边界时读取。服务器、地址、Conda prefix、容器/作业和下载状态以 [02](../../02_installation_and_environment.md)、[03](../../03_training_and_evaluation.md)、[08](../../08_thor_edge_deployment.md) 的当前证据为准，使用前实时复核；此 mode 不固定动态值。

- 长训练用获准计算节点上的 Slurm 或 tmux，不在登录节点或本地工作站运行训练循环；不停止既有下载，不删除远端数据、权重或缓存。
- Thor 只运行模型推理；不得读取或修改 3588 的机械臂控制、相机采集和系统部署。按模型系列隔离环境/容器，模型包完整性、精度、真实 Thor 本机推理及 Thor↔3588 直连 smoke 分别验收；端口开放不等于可用。
- 新 checkpoint 操作先确认原始权重、数据、norm、transform 与转换精度身份。量化和加速的验收条件取 [08](../../08_thor_edge_deployment.md)，Pi checkpoint 交接按 [手册](../../reference/thor/12_checkpoint_handoff.md)。
- 2026-09-29 用户已确认旧夹爪、动作相位和 136000 左臂异常解决；旧报告只作历史。操作前重新核对现场 checkpoint、输入和服务合同，不能从该确认推定当前就绪；功耗模式与退出恢复策略依 [08](../../08_thor_edge_deployment.md)。
- 客户端任务的新指令或追加内容须先展示并经用户审核；服务端实施可继续，详见 `AGENTS.md`。
- Git 推送、合并和远端同步遵守 `AGENTS.md` 当前授权，先核对实际分支和工作树；不 force push 或在 URL 中写凭据。
