# 2026-09-09 17:04 · 停训后软件与数据复核

结论：暂不放行正式训练，尚未定位非法访问根因；没有证据认定是数据、算法、NCCL、驱动或硬件中的任何一项单独导致。用户本轮要求先排除问题再训练，本轮未启动训练或训练 smoke，未改服务器环境/运行快照。

## 已完成

- 17:04:35 在 Slurm2064 的 gpu001 上只读检查全部4458个训练 Parquet、10,374,181帧：逐文件SHA256与既有norm provenance完全一致，state/action全部Nx14且有限；文件清单、manifest/info哈希一致，norm各统计值有限。见[data-audit.json](data-audit.json)。没有全量视频重新解码，也未重新验证基础权重完整哈希。
- 本地仅CPU轻量配置/合同测试19项通过、6项排除，没有模型前后向或训练循环。见[contract-tests.txt](contract-tests.txt)。覆盖入口参数、checkpoint拒绝逻辑、H50尾帧重复、关节delta/夹爪absolute、非有限值拒绝、相机映射及采样器相关边界。
- 服务器pip check通过；实际训练器和加载器哈希匹配a53bb00提交。见[environment-and-crash.txt](environment-and-crash.txt)。pip一致性不证明动态库ABI或GPU执行正确。
- f93a792→a53bb00的训练循环、模型、加载器、分片和YAM变换无差异；重启主要修改run隔离、保存周期与故障记录，不能称为非法访问修复。
- 当前环境包版本复核：JAX/jaxlib/CUDA12插件0.5.3、Flax0.10.2、Torch2.7.1、NCCL2.26.2、cuDNN9.5.1.17。驱动595.45.04。版本观察不代表已排除环境故障。

## 节点观察与解释

17:03:13四GPU各1MiB、0%利用率，无检出的训练/compute-sanitizer进程；2064仍有效。
EDAC mc0 CE137754、UE0，相比11:42历史CE131367继续增加。它是主机内存纠错计数，不是显存ECC；没有建立与GPU崩溃的因果关系，也不能因UE0忽略持续报警。

[NVIDIA Xid说明](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html)指出Xid13常见于应用非法访问，但系统软件和硬件也可能导致；Xid43常见于应用故障后的终止。libcuda内PC和NCCL报错只能给出故障发生/被发现的位置，尚无首个非法读写算子的完整证据。

上午完整memcheck和NCCL定向memcheck均超时，不能记为通过，不重复同样耗时诊断。现有小时巡检automation为PAUSED，本轮未启用。

## 放行所缺的证据

1. 管理员给出DIMM_B1与GPU1（PCI52:00.0）的实际检查结果，或提供健康替代节点；目前反馈是软件归因意见，不能代替节点健康验收。
2. 获准恢复计算验证后，在隔离目录用同一数据/权重/配置做受控定位：固定batch与真实加载器分开、首次错误同步定位、计算与通信分开，逐次只变一个因素。避免以取消GPU1并随意改变FSDP/batch作为已修复证明。
3. 确认首份完整checkpoint及独立进程恢复；覆盖第二次故障窗口的短跑仍不排除第一次15小时偶发故障，正式放行须明确长期观察边界。

完整数值复核降低了“数据文件被改坏、维度不对或NaN/Inf”的可能性，不能排除视频解码、样本相关动态行为、XLA生成算子或系统层缺陷；不能宣称算法和数据已经绝对无问题。
