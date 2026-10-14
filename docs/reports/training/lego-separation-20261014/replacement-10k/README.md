# 首轮10000步替换配方 v2

用户明确确认从240000重新初始化独立10000步run，并取消2344。旧run最后记录481步，无完整新检查点；目录、日志和取消回执保留，旧步数不计入新run。

| 项目 | 已确认v2 |
|---|---|
| run | lego_pi05_rtc_separation_r1_10k_20261014 |
| 初始化 | 原50h run的240000/params，新优化器 |
| 数据 | 原冻结200集，专项100＋回放100，各278516帧，期望50/50 |
| 目标/日程 | 新run 0→10000，warmup200，cosine10000，LR3e-6→3e-7 |
| 模型 | Pi0.5全量、global32/FSDP4、H50、RTC0..10 |
| norm/optimizer/seed | 继承240000 norm；AdamW .9/.95/eps1e-6/weight_decay1e-10/clip1；EMA关闭；seed42 |
| 保存 | 每1000保存，keep_period1000，保留1000至10000全部千步点 |
| 资源 | 4张RTX4090、64CPU、480G；预留10小时，粗估7–8小时 |
| 预算/停止 | 每池期望160000个帧起点，约0.574轮；10000结束不自动延长，NaN/Inf/数据或保存错误停止排查 |

正式配置为[pi05-lego-separation-r1-10k.json](../../../../../configs/native/pi05-lego-separation-r1-10k.json)，审批归[approval.json](approval.json)。原5000步配置与提交代码保留供追溯。
