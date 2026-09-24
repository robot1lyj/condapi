# 03 · Mode — Deployment

仅在远端环境、训练作业、checkpoint 交接或 Thor 服务操作需要额外边界时读取。服务器、地址、Conda prefix、容器/作业和下载状态以 [02](../../02_installation_and_environment.md)、[03](../../03_training_and_evaluation.md)、[08](../../08_thor_edge_deployment.md) 的当前证据为准，使用前实时复核；此 mode 不固定动态值。

- 长训练用获准计算节点上的 Slurm 或 tmux，不在登录节点或本地工作站运行训练循环；不停止既有下载，不删除远端数据、权重或缓存。
- Thor 只运行模型推理；不得读取或修改 3588 的机械臂控制、相机采集和系统部署。按模型系列隔离环境/容器，模型包完整性、精度、真实 Thor 本机推理及 Thor↔3588 直连 smoke 分别验收；端口开放不等于可用。
- 新 checkpoint 操作先确认原始权重、数据、norm、transform 与转换精度身份。量化和加速的验收条件取 [08](../../08_thor_edge_deployment.md)，Pi checkpoint 交接按 [手册](../../reference/thor/12_checkpoint_handoff.md)。
- 用户曾报告 136000 Pi policy 左臂异常；真机动作前复核 [故障报告](../../reports/thor/rtc-20h-136000-20260923/README.md)和现场状态。推理/测试临时 MAXN，结束恢复日常 120W，具体操作依 [08](../../08_thor_edge_deployment.md)。
- Git 推送、合并和远端同步遵守 `AGENTS.md` 当前授权，先核对实际分支和工作树；不 force push 或在 URL 中写凭据。
