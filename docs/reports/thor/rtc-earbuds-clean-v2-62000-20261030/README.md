# 耳机clean-v2 retry4：62000步下载交接

## 授权与当前状态

2026-10-30用户要求拉取最新耳机训练检查点。本轮只下载与核对完整性，不转换、不构建、不预热/测试/启动推理，不改训练或操作3588。查询服务器各耳机run后，锁定新分支最新完整62000，不能按跨run的数值大小误选旧114000；后续训练产生新保存点也不静默追更本轮目标。

**2026-10-30 16:20:15+08下载与校验完成。** [下载回执](download-receipt.json)为`downloaded_sha_verified_not_converted`，19文件/12,440,562,368 bytes（约11.59GiB），[源SHA清单](source.sha256)逐项通过，冻结脚本退出0。16:20后复核Pi仍running=false、目标无train_state，Thor可用约303GiB。没有转换、参数有限性审计、精度/时延测试、启动或切换推理；SHA通过不当作模型质量或部署验收。

已核对[训练合同](training_contract.json)、[初始配置](initial_config.json)与正数Orbax commit时间。checkpoint实际norm SHA `b7c63db5b78959d5da3ce720fe8f45c075eac57ed97bf949dacc9c86108f1878`，相对训练norm仅缺末尾一个换行，身份检查通过；这是新分支自己的norm，不是旧114000的norm。后续转换必须绑定本目录资产与独立新报告/引擎，不复用旧模型产物。

## 身份与路线

- 源run：`/home/wuyan/lyj/YAM/training-runs/pi05_yam/earbuds_pi05_rtc_base_50h_clean_v2_excl_pantheon_retry4_20261022/62000`。已确认正式`_CHECKPOINT_METADATA`、正数commit时间、params/_METADATA和checkpoint自身norm存在，不取临时保存目录。
- 源control：`/home/wuyan/lyj/YAM/training-runs/control/earbuds_pi05_rtc_base_50h_clean_v2_excl_pantheon_retry4_20261022`，配套`training_contract.json`和`initial_config.json`独立取回。
- 本run从旧耳机114000/params初始化、步数重新计数，使用clean-v2排除pantheon后的数据与新norm；62000不代表原run训练倒退。合同Pi0.5/H50/32D、训练时RTC dmax10，训练norm SHA `05f65ada286965f1c5c5b32e62bf3fcdafb7d41d2e22d6683e0c05efc443c5b2`，不能复用此前旧114000 norm或引擎。
- Thor目标：`/home/wuyan-lyj/thor/pi/checkpoints/earbuds-pi05-rtc-clean-v2-excl-pantheon-retry4-20261022/62000`；只取params/assets/完成元数据和合同配置，不取train_state/优化器。源params/assets共12,440,526,965 bytes，约11.59GiB。
- Thor stage：`/home/wuyan-lyj/thor/pi/probes/rtc-earbuds-clean-v2-62000-20261030-r1`。冻结[下载脚本](download.sh)通过工作站转发既有SSH agent，Thor直接从服务器读取六路rsync，不复制私钥或先落工作站大权重；flock防同stage重复下载，保留partial以供明确恢复。
- 下载后逐文件SHA核对，校验本run配置/commit/RTC合同及checkpoint自身norm身份；只容许既有Orbax少末尾一个换行的norm差异。SHA一致不等于参数有限性、精度或部署验收；这些留待新的转换/测试授权。

此前清理的旧114000模型不恢复；乐高20万/23万/24万与官方基础权重、运行中的其他容器均不动。
