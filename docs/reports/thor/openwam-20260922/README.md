# OpenWAM Thor准备完成

观察时间：2026-09-22 17:14 CST。用户限定暂不暂停Pi，**未运行GPU模型加载或推理性能/精度测试**。

- 官方代码commit：`90e94ae31efddd64b59e00365cfc501d9a972eb1`。
- 镜像：`openwam:thor-20260922-r1`，ID `sha256:e8b039d0e55ff193439df46fd74956a2407c5e72e0676ad6908612b48bb943bc`。
- Thor检查点：`/home/wuyan-lyj/thor/openwam/checkpoints/OpenWAM-Alpha-Sim-RoboTwin-Full`。
- HF revision：`04b96af53eeeb111c64631f822efcbb82f4b186e`。
- 主文件：`checkpoint_step_118655.safetensors`，24,813,767,464字节；SHA256 `d07ff6f8cfb627ebf47313e65af38773fabcb3861d52fdc7c3bc5e7bee33f6fe`，与官方LFS元数据一致。
- 9个文件下载完成；详细文件哈希见[下载回执](thor-download-receipt.json)。2089个BF16张量CPU扫描无NaN/Inf，见[有限性回执](finite-audit.json)。这不证明动作合理性、YAM适配或任务成功率。
- 无GPU/无网络的CLI、引擎与工厂导入通过；分词器离线实际编码通过（T5Tokenizer，测试提示词12tokens）。
- Docker下载与审计进程均退出0；保留日志与下载断点恢复历史。未清理其他容器/模型。Pi健康检查OK。

下一步需用户允许暂停Pi后，独占MAXN测试官方10步基线，再逐项测试compile、提示词缓存、DiT近似缓存与量化。测试结束恢复Pi；不自动替换生产模型。路线与LoRA边界归[14手册](../../../reference/thor/14_openwam_inference.md)。
