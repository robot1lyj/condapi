# PLD 推理延迟优化与十步前缀验收（2026-10-13）

**已上线：2026-10-13 13:01 CST**，`pld-rl:thor-latency-20261013-r1`在原`ws://192.168.250.1:8000`实查running/healthy/MAXN；生产48例包含actor的P50/P95为213.44/214.14ms，另9例RTC与原客户端16回复校验通过。旧容器`pld-rl-retained-pld-latency-20261013-r1`停止保留。客户端重连，继续collect＋前缀10采20集；优化后新跨IPC实录已通过：409有效回复/0迟到，现场worker P95=257.91ms、服务端P95=214.23ms；47窗口待判奖。详见[跨IPC审核](../../rl/lego-pld-collect-20261013/e05-fast-d10/README.md)，纯服务耗时与端到端指标继续分别报告。

用户已明确“现场已暂停，现在验收升级”。本轮复用原20w Pi、相同ResNet18和零更新actor、正负可行域、门控阈值、半径、RTC七步去噪以及FP32/TF32精度；没有训练、替换权重、修改3588或改变奖励。执行技能为`thor-checkpoint-deploy`，此次是用户明确授权的服务性能升级，不是新checkpoint转换。

## 实际瓶颈

MAXN现场实查已开启，Thor直连网口enP2p1s0为2500Mbps/full。最新九步录制中，活动请求服务端P50约221.85ms，PLD后处理19.03ms；差值约202.79ms为Pi及其处理。worker减服务端P50约34.52ms，worker减已单独计时的pack/send/wait/unpack约12.82ms。这12.82ms仍包括进程间传递、请求构造、返回处理和校验，不能统称校验耗时。

最初“观测tick到发请求约33ms”的解释已细化：token.created_at与相机配对anchor差P50为25.34ms；anchor与映射控制tick差P50为8.01ms。created_at又来自控制循环now，不是实际socket发送时刻，不能把这33ms当作已证实存在且可直接删除的一次线程sleep。图像来自三路相机历史，state按anchor插值；禁止移动tick来伪造低延迟。

当前客户端PldClient已将握手身份保存，build_request使用已缓存identity。每包是少量字符串比较和14D/50×14数组核对，不重算checkpoint/hash文件。本地工作站对最新293个真实回复共1172次validate_reply回放，P50约0.019ms、P95约0.024ms；只说明这个函数在工作站很轻，不能冒称RK实测或免除逐包动态检查。[计时](local-validator.json)。

## 代码改动与等价性

源码`22fcebf040509a56e1d41a7474eb6c4e5caa6806`，候选镜像`pld-rl:thor-latency-20261013-r1`、ID`sha256:963d5a2863db85fb8a4c2230ded97de6390f5738a386b994e29c3889b8dd763f`。

1. FK将当前左右状态与未来各50个姿态一起批量计算，保留标量实现作参考。前三集740个真实请求，所有高度差为0、门控决定一致，489个活动阶段float32上下文完全相同。Thor隔离CPU测量102姿态P50从8.9467ms降至0.3117ms，P95从9.0073降至0.3186ms。[Thor结果](thor-cpu.json)。
2. NumPy数组打包对C连续非空数组使用memoryview，省去tobytes中间分配；跨步/Fortran/空数组保留原转换。固定三路640×480RGB打包字节完全相同；Thor CPU P50 0.2929→0.2192ms，收益小，图像大小与内容完全不变，不称零拷贝网络或图像压缩。尚未更新RK上的SDK副本。
3. server_timing新增request_decode_ms、base_infer_ms、pld_postprocess_ms，旧infer_ms、回执/动作/模式均保持。所有动态校验继续有效，不通过删保护来降低耗时。

72项轻量回归与Ruff通过，[记录](local-checks.json)。CPU首个隔离容器因报告目录UID不匹配写入失败，重试匹配数据owner后通过，失败容器和日志保留；不是数据损坏，不更改模型逻辑。

## 完整 Pi＋PLD 相同输入对照

在Thor顺序运行原版和优化版，输入真实录像图像和前缀，逐请求固定Pi噪声与actor随机数，四模式/双臂/三次重复，共48组；前一轮作为预热，统计后两轮32组。

| 部分 | 原版P50/P95 ms | 优化版P50/P95 ms |
|---|---:|---:|
| Pi及处理 | 203.389 / 204.383 | 202.433 / 203.134 |
| PLD后处理 | 18.929 / 19.341 | 10.325 / 10.804 |
| 完整计算 | 222.338 / 223.742 | 212.855 / 213.769 |

原Pi动作、最终动作、2287维actor输入、探索样本分别逐值相同，最大差0。[等价回执](same-input-equivalence.json)、[原版](ab-old.json)、[优化版](ab-new.json)。不同分项的分位数不能直接相加，整机总量约省9.5–10ms，其中可稳定归因的FK优化约8.6ms；这些是Thor内完整前向，不是跨IPC实测，也不是成功率提升。

候选真实WebSocket 48例P50/P95约212.965/214.375ms，另外9例RTC d=0/1/10通过；原客户端校验器验证16组返回包通过。[客户端校验](client-preflight.json)。生产切换结果以同目录deployment.json及production报告为准，不以候选成绩代替。

## 十步录制与批量采集

用户最新session_20261013_122649_fbc88d/episode_000002采用旧线上版本、前缀10，293回复中292接受、0迟到、1包结束取消；248个专家区间全部来源核验通过，38窗口中37个待判奖，1个关联歧义待复核。左右跟随恢复，整轮人工记录9/10。前一集前缀9有13/236迟到、10/22窗口损失。详情见[实录审核](../../rl/lego-pld-collect-20261013/README.md)。

前缀10是给接管留333ms预算，控制依旧30Hz、Pi仍七步去噪；它已在一次真实录制中通过，不承诺所有未来网络抖动。升级重连后继续collect＋前缀10采20集，保留每集C/N、成功/失败及异常标记；当前actor仍0次RL更新，局部奖励尚未发布，不能称已提升成功率。

## 客户端剩余工作与存储

[客户端延迟改造建议](client-latency-handoff.md)提供精确计时位置、单次编码/bytes进程通道的候选方向与验收边界。按项目约定先供用户审核，没有自动发送或修改客户端。当前数据不支持“把13ms校验关掉就省13ms”；应先测Pipe、准备和校验各自耗时，保持30Hz时间对齐和每包动态检查。

工作站完整原始回执、固定噪声输出和分析脚本：`/home/wuyan-lyj/condapi-data/rl/pld-latency-20261013-r1/`。Thor CPU/完整A/B及源代码：`/home/wuyan-lyj/thor/pld-grasp/checks/latency-20261013-r1/`；服务release：`/home/wuyan-lyj/thor/pi/services/pld-latency-20261013-r1/`。原推理权重和所有旧容器/失败证据保留。

服务器论文归档：`/home/wuyan/lyj/pld-grasp/runs/latency-20261013-r1/evidence/`，固定噪声全输出、完整返回包及报告共21个成员逐项SHA核验通过，[回执](server-receipt.json)。只归档实验结果，服务器没有从GitHub取代码，没有训练；原始机器人视频仍保留在IPC及上述工作站只读副本，未声称本次归档包含视频。
