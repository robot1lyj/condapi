# PLD-Grasp gate v3 修复上线与首两集试采

2026-10-13 10:40 CST，用户明确确认“已停止执行，仅保持连接”后，修复版接管原8000服务。当前`pld-rl:thor-collect-20261013-r1`已实查running/healthy/MAXN。它可用于受控collect试采，尚无此版跨IPC真机rollout、成功率改善或已训练专家的证据。

## 实现与身份

- 源码`e21431b26f46aa9320ef9f32f26b07a6d08dfbac`；镜像`sha256:5b072a86671680654d4b9d03d4bb5d0102b191b8a20719debc8aa1d12719f8cf`。
- actor快照`583ae73a422912ea6e5e0e495127d40d5b6d5dbc6ad6ec8af0dac7f322d31144`；合同`87b1affa6d81e1d0cf6299bf6e827bd260361104338708e67579c874ee65d7ec`。包、配置、优化器与RNG留在服务器；[manifest](bundle-manifest.json)绑定具体actor文件SHA。
- Pi继续原50h/200000、七步trained RTC、FP32权重/TF32 engine与quantile前缀；原权重、norm、engine SHA见[部署回执](deployment.json)。夹爪SDK边界仍为0.02–0.98，候选残差半径仍为关节0.02rad/夹爪0.05。
- 新特征2287维，增加负方向可行幅度及窗口阶段信息。新包独立导出、0次训练更新，不能加载旧2275维包假装兼容；旧gate与bundle继续按旧合同解释。

## 本次四项修复

1. 抬升达到门槛就停止新增修正；随后进入强制零动作的draining阶段，Pi继续运行。无需确认帧仍在上升，也无需额外停住1秒。
2. 排队终点覆盖此前所有可能执行的修正包络，再给至少2个控制tick观测余量。奖励必须取排队后的新鲜图像；成功使用两个不同帧且包含最后可用帧。RTC已承诺前缀与来源不改写，排队步骤进入replay，但不更新actor/温度或添加探索熵。窗口完成的同次观测允许开始下一窗口，避免额外空一轮请求。
3. 正负方向分别计算可行域；夹爪在0.98处仍可向闭合方向修正，在0.02处仍可向张开方向修正。零输出保持物理Pi基线。采样、解码、特征、critic动作与replay重算一致。
4. 增加not_lifted/no_closure/dropped结果：五类状态仍映射held=1、其余=0，中间步骤0。未抬起/未闭爪不再强制5cm，但需要完整实测轨迹、终态图像和抓取意图审核；不把它们冒称视觉上已证明空手。预测后2s未闭爪、闭爪后3s未退出会开始排队；这些是首批任务预算，需由新录像验证是否过早。HIL、故障、丢帧、无效测量和误触发不能强填0。

图像新鲜性依据现有`camera_host_received_at`和视频索引，不是新增的相机曝光标定。首批仍需结合录像抽检结果与动作时序，不能宣称已实现无人工复核的自动奖励器。新`outcomes-prepared.json`保留未抬升候选及未批准的规则建议；旧Qwen二分类准备文件不强塞无5cm案例。

## 验证及证据边界

- [CPU回归](local-validation.json)：本地及服务器52项通过，包含旧合同、快速抬升停住、排队尾部、五类目标结果、缺失/HIL拒绝、端点单向修正、连续性和无5cm图像导出。
- Slurm2325：五类合成轨迹共54条transition；CLI做40次critic/18次actor更新并重载，排队状态actor更新为0。[服务器报告](server-validation.json)。随后**另建**零更新生产初始化包，[bootstrap运行记录](bootstrap-run.json)与合成实验分开；合成权重没有部署。
- 候选与生产各48例真实录像输入（左右臂、四模式、三次重复），另各9例普通RTC。生产包含actor的P50=222.92ms、P95=223.91ms；这是Thor本地服务计时，不是机械臂端到端时延。[候选](preflight-bundle-smoke.json)、[生产](production-bundle-smoke.json)、[普通RTC](production-rtc-smoke.json)。
- 原客户端`5eff0c46e1a9a6550ec2cb5a81be8e50d14e9c6e`的校验器对候选/生产各16个真实返回包通过，未修改客户端。[回执](client-wire-validation.json)。
- 独立Thor actor前向用真实录像图像与合成排队状态核验2287维输入、强制零修正、前缀保持及排队完成。[回执](draining-forward.json)。它不替代真实物理时序验收。
- 旧probe反事实门控重放37窗口（左6/右31），覆盖旧20候选，0跨两个新版闭合，0边界拒绝；34次lift_guard、3次reopened。[校准](calibration.json)、[窗口](windows.json)。**这些不是抓取成功次数。** 初版多等一次观测阻塞下一抓，仅覆盖18/20，失败输出保留为[初次校准](initial-calibration-not-ready.json)。

中间工程失败也保留：Slurm2324因新检出尚未填充文件，在测试导入前失败；完成受限检出后2325通过。首个独立排队前向容器已完成断言，但写报告被目录权限拒绝；改为匹配数据所有者UID的第二个容器成功，未修改模型代码或扩大权限。

## 两集采集与处理

重新连接`ws://192.168.250.1:8000`，确认新握手`gate_version=3`及上述合同/快照，选择collect，按原连续HIL录两集；无需每抓按键、客户端切段或编号。第一集留验证，第二集作为训练候选。每集保留完整session侧车、manifest/H5/三视频，整轮C/N继续只作评估。需要人工介入时保留实际标记，不将该窗口冒充无人专家经验。

录完先检查实际非零修正来源、排队结束、退出是否及时、误介入/限位、奖励与录像是否一致及各类排除占比。确认真实链路后再看有效尝试数、左右臂与成功/失败分布，决定是否更新SAC或补采；两集不是训练收敛保证。

Thor CPU入口用`run_pld_curator.py --purpose candidates --export-evidence`，并将第一集列入`--validation-episodes`。审核发布后用`--purpose rewards --rewards <审核文件>`写新目录，再用`prepare_blocks.py`编译。原默认purpose=probe保留为诊断用途，不用它发布collect奖励。完整字段归[现行方案](../../rl/pld-sac-block-v2/README.md)。

## 归档与回退

Thor release：`/home/wuyan-lyj/thor/pi/services/pld-collect-20261013-r1/`，保留源码包、镜像构建/预检/切换日志、actor包与验证结果。原collect-r2容器改名`pld-rl-retained-pld-collect-20261013-r1`并停止保留；旧Pi、probe、r1中间版本及全部原始数据/权重均保留，未操作3588。

服务器：`/home/wuyan/lyj/pld-grasp/runs/collect-bootstrap-20261013-r1/`。`synthetic-validation/`只作数值检查，`export/`为真正部署的checkpoint0/bundle，`deployment/`归档本次回执和源码包；训练环境未改装。源码从Gitea经临时反向通道获取，未从服务器访问GitHub。工作站完整返回包与校准输入在`/home/wuyan-lyj/condapi-data/rl/pld-collect-20261013-r1/`；Git只保存必要回执，不提交权重或整段响应。
