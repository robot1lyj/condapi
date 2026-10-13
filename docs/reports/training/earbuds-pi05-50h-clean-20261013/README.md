# 耳机50h清洗选集（2026-10-13）

用户明确要求异常数据整集排除，不修复、不裁帧、保留原件。CPU审核作业2326（32CPU，代码37e8e2f）已完成：原候选仅source35033的左腕视频3464帧而合同3465帧；整集排除，沿同一seed42顺序补入source60778。最终1384集、5,403,161帧@30fps=50.029269h；原val17集全部通过，独立保留。全部选中低维行和三路视频逐帧审核通过，不代表逐集人工成功标注或实机成功验收。

`audit.jsonl`含每集完整审核结果及源文件签名；`excluded.json`记录被排除文件的原始SHA256和异常；`source-episodes.json`为原train ID，`episodes.json`为新转换后ID；原source manifest字节及哈希不变。转换复制全部健康视频，逐文件验证SHA256；仅在全部源签名与审核一致时复用全解码证据。转换和norm使用独立v2目录，旧失败现场保留。

训练配方仍为官方Pi0.5 base、全量RTC0..10/H50、batch32/FSDP4，累计200000截止，固定LR decay338000。实际作业与首步状态见[运行记录](../earbuds-pi05-50h-20261012/README.md)。

## 数据发布与实际开训

固定代码 `b1446eefdf8c3cd347689bb41cc77fa6af251aaa`；服务器checkout `/home/wuyan/lyj/YAM/code-snapshots/earbuds50h-clean-v2-20261013`。数据准备2327完成并输出DATA_READY，norm覆盖5,403,161帧，12个真实loader首/中/尾等价对照通过；norm SHA256 `109af6930ddfdf9e0c04995809cd5e056037b42bbeeb54399ec1b4871ddb3cce`。

训练2328已在gpu001进入真实优化。2026-10-13 13:38+08核验step51，loss0.0478722453、grad_norm0.57456696、LR5.09490746e-7，所有已记录指标有限；早期步时约2.27秒，不作为完整工期承诺。四张RTX4090利用率100%、温度50–55°C，nvidia-smi占用约22.3GiB/卡（含JAX预留）；训练日志活跃分配约9.4GiB。首个计划保存1000步，此时尚未产生完整checkpoint。

实际启动配置确认累计200000、batch32/FSDP4、RTC10、H50、LR decay338000，独立clean-v2数据/norm路径已生效。面板已切到耳机run，阶段和总体目标均为200000，SSH镜像及Slurm状态正常，页面已显示真实step/loss曲线。可复核[作业提交](launch_jobs.json)和[首步验收](startup-verified.json)；动态状态使用前重新查询。
