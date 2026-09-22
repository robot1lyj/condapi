# OpenWAM 服务器独立训练环境 · 2026-09-22

## 当前安装交接

用户在微调接口任务中追加要求准备服务器训练环境；已授权独立 prefix 安装与环境审计，未授权启动实际模型训练。本地接口已提交 `9013b33`，环境审计/源码恢复合同补充提交 `8e29b1c`，位于 `codex/openwam-adapter`。仍保留其他任务未提交的看板改动。

- 原始本地 OpenWAM revision：`7c5861e45cfe1339a0323f0e0b03a3316c37971c`；原样 vendor 与逐文件哈希在 `third_party/openwam/UPSTREAM.json`。
- 服务器：`yam-server`（`rocky-login.hlink.local`）。安装目录 `/home/wuyan/lyj/openwam-install-7c5861e`，独立 detached 工作树 `code/`，来源仅为 Gitea 分支。
- 新 Conda prefix：`/home/wuyan/.conda/envs/vla-openwam`。未修改 Pi/Evo/Molmo 环境。
- 安装脚本/日志/退出回执：安装目录下 `install.sh`、`install.log`、`install.exit`；当前续装使用 `resume_openwam_install.sh`、`install-recover.log`、`install-recover.exit` 和 tmux 会话 `openwam-install-recover`。恢复时先查退出回执和日志，不重复创建 prefix/启动第二份安装。
- 复用已存在 Python3.12/FFmpeg bootstrap archive，解包后运行 conda-unpack。共享 NFS 解包较慢，不能误记为数据损坏；目前尚无环境可训练结论。
- 目标依赖以固定上游 CUDA12.8 constraints 为候选；实际版本与验证结论以随后安装/审计产物为准。安装禁用 GPU、DeepSpeed 可选预编译，使用低优先级及单线程。原生 trainer 使用 PyTorch AdamW，ZeRO-2，不依赖 CPU optimizer offload；额外 offload/FlashAttention 编译不属于已通过项。
- 2026-09-22 15:00 CST 资源观察：Pi 作业2165在gpu001使用4GPU/64CPU/480G，gpu002也已分配，无空闲GPU。不会附着现有训练作业执行测试。GPU audit 要等独立 Slurm allocation。
- 服务器直连 Gitea 超时；通过工作站临时 `-R 127.0.0.1:12225:192.168.110.142:2222` 隧道获取源码，以本机已知 Gitea ed25519公钥固定校验（安装目录 `gitea_known_hosts`），不关闭主机公钥校验、不访问GitHub。未更新服务器原训练 checkout。

## 检查点下载

- 微调基础模型按官方 OpenWAM-α 文档选择 `OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model`，不是 Thor 部署用的 RoboTwin-Full 策略。官方元数据在 2026-09-22 16:27 CST 解析到 revision `52df4e66c82c5c8b480adcc8d01f4db7415dfb56`，8 个文件合计 `24835230480` bytes；其中 `checkpoint_step_154000.safetensors` 为 `24813767464` bytes，LFS SHA256 为 `180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d`。
- 目标目录为 `/home/wuyan/lyj/openwam-models/OpenWAM-Alpha-Pretrain-Foundation-Model`，通过 `download_openwam_foundation.py` 在 tmux `openwam-foundation-download` 中断点下载，固定 revision 后逐文件大小/SHA256 写入 `foundation-download-receipt.json`。下载期间设 `HF_HUB_DISABLE_XET=1`，模型完成前不得把目录交给训练。
- 官方配置确认 `action_dim=state_dim=80`，YAM 原始数据为 14D 关节合同；当前适配器只接受明确的 14D 原生 checkpoint，或调用者提供经过物理语义审核的 14 槽 80D 投影。不会把关节角静默放入 EEF xyz/rot6d 槽位。

## 已有验证

本地临时测试依赖位于 `/tmp/condapi-openwam-test-deps`，调用既有 condapi-yam Python但不修改其安装包。控制层全套轻量测试73项通过；其中OpenWAM17项覆盖真实微型MP4/Parquet、训练集统计隔离、原生normalizer回读、mask/相机布局、native config微调/恢复与本地训练拒绝。没有模型前后向或训练循环。

相关改动 Ruff 与格式检查通过，`git diff --check` 通过。全仓 Ruff 检出141项既有问题，格式检查另有24个既有文件待格式化；主要在Pi/Thor旧脚本与历史报告，本次不扩散修改。GPU模型加载、实际更新/恢复、完整依赖安装验收仍待完成。

并行Thor实验使用另一上游revision（见 [Thor owner](../../../reference/thor/14_openwam_inference.md)），未来传递checkpoint必须核对来源与保存配置，不能把两个版本的导入成功合并成训练部署验收。
