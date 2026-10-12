# pld-rl 推理容器上线（2026-10-12）

## collect-r2 当前上线（19:55 CST）

用户批准修正门控并确认客户端已停推理后，`pld-rl:thor-collect-20261012-r2`已在原8000地址healthy/MAXN，五模式开放。可录两集collect，初始actor为**零均值有界探索**，尚无真机改善证据；`replay_contract_ready=true`只表示正式记录合同已装好，不代表已有训练数据。Pi仍为原50h/200000七步trained RTC，权重/engine/norm均未更换。

| 项目 | 当前证据 |
|---|---|
| 镜像 / 运行源码 | `sha256:37ab017566f6984cee94c8b546745c7cf9af4316863c4094bb35c1c00c611c64` / `2612e08f8c2668e665ca07ba00dcf5b5bcc318af` |
| actor快照 / 合同 | `ac5d1e0df8eb8ba7c13d4f8446c31dc2674526d9d21478c3fca353d7e98d8e2d` / `97bf9c4b4b5173477a9c3bf01e6f1ab4656b47d6757d919237daeec7278d6601` |
| actor初始化 | Slurm2321，固定e4e1a7b导出checkpoint0/完整优化器与RNG；0梯度更新，不用旧漂移actor |
| 半径 / 连续性 | 每关节±0.02rad、夹爪±0.05；残差每tick guard 0.008rad/0.02；SDK原0.02–0.98端点不改 |
| 本机输入验收 | 候选与生产各48例probe/shadow/eval/collect，左右臂均触发；普通RTC各9例d=0/1/10通过 |
| 延迟 | 生产含actor P50/P95=222.86/224.21ms；普通Pi路径198.88/199.71ms；不是跨IPC物理控制端到端计时 |
| 原客户端兼容 | 对真实返回包调用只读YAM原`validate_reply`，16例通过；字段保留、前缀精确，未改客户端 |
| 尚待验证 | 新actor跨IPC/真机rollout、实际非零修正效果、首两集奖励；本机输入测试没有连接执行器 |

新版gate按Pi未承诺计划提前选择、相对开合确认/重新允许，闭合后跟踪下降低点，连续上升且≥5cm交还Pi，无额外1s保持。现场FK闭合高度范围过滤框边收拢；它不是语义检测器，换场景须重新校准/替换适配。网络`base/actions`满足原客户端校验；训练用新增`physical_base/physical_actions`，SDK端点映射不冒充残差；RTC延迟执行沿原token核验。[方案与采集步骤](../../rl/pld-sac-block-v2/README.md)定义奖励及训练边界。

[部署](collect-r2/deployment.json)、[模式与延迟](collect-r2/production-bundle-smoke.json)、[普通RTC](collect-r2/production-rtc-smoke.json)、[客户端校验](collect-r2/client-wire-validation-r2.json)、[校准](collect-r2/calibration.json)、[合同](collect-r2/contract.json)、[初始化](collect-r2/bootstrap-run.json)保留独立身份。41项本地轻量回归通过；Thor离线CPU/runc独立导出v3为38候选/116旧诊断区间、奖励0，未训练，[离线回执](collect-r2/curator-launcher.json)保留原始只读边界。

收尾复验：生产返回包另16例通过原客户端校验器，[生产回执](collect-r2/client-wire-validation-production-r2.json)与候选16例独立保留。最终离线管家固定7cb65ce发布r4，明确禁止参考低点之前的高处画面获得抬升资格；Thor CPU导出38候选/35合格/0假前置抬升，[启动回执](collect-r2/curator-launcher-v4.json)。服务器完整核验1083张JPEG和全部顶层输出SHA，[归档回执](collect-r2/curator-server-v4.json)标记`derived-v4`为当前候选版本、training_ready=false；v3及原件不改写。GPU服务器从Gitea取7cb65ce源码，41项同组CPU协议/数组检查也通过，没有训练循环。资料归`datasets/rollouts/lego-probe-20261012-e01-v1/derived-v4/`、`datasets/catalog/lego-probe-20261012-e01-v4.json`和`runs/collect-bootstrap-20261012-r1/deployment-r2/`。当前在线镜像仍是已验收2612e08，离线管家升级没有重启生产。

当前release为`/home/wuyan-lyj/thor/pi/services/pld-collect-20261012-r2/`，actor与encoder只读复用r1文件。原probe容器`pld-rl-probe-retained-20261012-r1`、r1探索容器`pld-rl-retained-pld-collect-20261012-r2`及旧Pi都已停止保留；镜像、权重、旧日志不删除。r1为本轮中间工程版本，补齐客户端wire字段兼容后以r2为正式试采版本，未在r1执行真机探索。未操作3588，无客户端任务消息。以下15:31条目是首版历史，不代表当前能力。

用户明确要求新增名为`pld-rl`的容器，并允许停止原Pi推理。2026-10-12 15:31 CST完成切换：**`pld-rl` running/healthy，`ws://192.168.250.1:8000`不变，旧`pi05-rtc-infer`已停止并保留**。当前只是`probe_diagnostics_only`，实际动作仍全部来自Pi，没有加载或执行RL actor。

## 当前合同

| 项目 | 当前值 |
|---|---|
| 专家名称 | PLD-Grasp 抓取修正专家 |
| 开放模式 | `off`、`probe`；`collect/eval/shadow`明确拒绝，不静默退回Pi |
| 基础策略 | 原50h/200000 Pi0.5 trained RTC，7步、FP32权重/TF32计算、quantile前缀 |
| 动作 | 原50×14物理目标，30Hz policy tick，原0.2rad/tick粗大关节步进拒绝保持 |
| probe内容 | 当前Pi基础计划、原观测/原请求身份、因果窗口/活动臂、FK、上个下发命令；整段actions逐位保持 |
| 训练资格 | `actor_loaded=false`、`replay_contract_ready=false`、`replay_eligible=false`；诊断记录不进入SAC |
| 数据与奖励 | 当前未运行Qwen/DINO判奖、未启用RL训练；C/N仍仅评估 |
| 电源 | MAXN，`thor-pld-rl-maxn-20261012-r1.service` active，跟随新容器；停止后恢复120W |
| 启动策略 | Docker `restart=no`，MAXN为transient unit；不是重启机器后自动上线 |
| 客户端 | 未读取、修改或部署3588；跨IPC与真机采集仍待客户端联调 |

握手当前`pld.contract_sha256=f405be16d4e845bc6bd112384d5a9ad684d42488b367de82ae5ff7a54eee9184`、`snapshot_sha256=8ec6645a3ec984ae3b68c05cac33244030b8d0d1640ce459706682d5dcb511e0`。这是**诊断合同**，不是伪造的正式动作边界；客户端每轮从握手读取，不硬编码。正式限位、每tick最大目标变化和actor bundle补齐后，切换另一份有版本的合同，再采可训练的probe/collect。

## 产物与身份

镜像`pld-rl:thor-probe-20261012-r1`，ID `sha256:080d09d5426928029a8e78e6a213e8cd6d27e20b8482518701f088a1d6323499`。继承实际原镜像`sha256:a6227665c32d1c7b2b7ca01a2d1a29e18d64bf764e5914db6eb06db5419a8e56`，没有pip安装、权重转换或TensorRT重建。仅增加PLD模型侧源码、固定FK模型、probe入口和WebSocket扩展，保持原`/app`工作目录及数据挂载。

- 推理源码固定`7804a55e287703efe15ff83d448c747e9e59cca7`；Dockerfile归`scripts/thor/Dockerfile.pld_rl`。
- 宿主部署控制器固定`bb127b4`，SHA `2ba697c2565f3bf4bb13480b7d5325d79f0e3143c3272f43f063c6a25b0a06d9`；它只修正MAXN交接，不改变镜像中的推理数值逻辑。
- 权重SHA `298714be810f5e7e822bcac55c5efe55789f1c96c0ac4ed0a1caa86f2670e42a`。
- norm SHA `fd0d9cb2db3ce475c1175c7f21a3a0dd3aa7f59d36edc680e7044cd11b9f1edf`。
- engine SHA `15ae4a5ceb57b6b6d8743c721cbe43178ab29c714be95cfbf1b9b893574ba71c`。

Thor完整发布目录：`/home/wuyan-lyj/thor/pi/services/pld-rl-20261012-r1/`，保留源码tar、构建日志、控制器、原容器必要信息、新容器创建参数、预检和正式端口完整回执。工作站备份：`/home/wuyan-lyj/condapi-data/rl/pld-rl-deploy-20261012-r1/`。旧容器/镜像/20w与23w权重/所有engine与原始数据未删除或覆盖；预检容器`pld-rl-preflight-20261012-r1`停止保留，8001无监听。

## 验证

本地38项轻量检查通过（8数组、5数据管家、4probe、21协议），没有训练或模型前向。probe诊断不能进入SAC；未支持模式在Pi推理前拒绝；连接/模式重置保留。

预检8001通过普通RTC9例＋probe9例；随后停旧容器、等待旧MAXN会话恢复120W并释放锁，再启动新MAXN会话及正式容器。正式8000再通过普通RTC9例＋probe9例，覆盖`d=0/1/10`，全部输出有限、RTC前缀保持；probe整段等于原Pi结果，三种不支持模式明确拒绝且连接可继续使用。

| 正式端口测试 | 本机往返P50 | 本机往返P95 |
|---|---:|---:|
| 普通RTC | 199.66ms | 210.41ms |
| probe | 200.83ms | 201.78ms |

每组仅9个真实记录输入，包含首次请求，不是现场控制端到端时延或抓取成功率。没有连接执行器、训练或实际施加修正。[切换回执](deployment.json)、[普通RTC](rtc-smoke.json)、[probe](probe-smoke.json)、[预检](preflight.json)保留实际身份与逐例数据。

MAXN交接使用短命Docker管理客户端连接宿主systemd（宿主PID命名空间，仅systemd客户端/共享库/通信目录只读挂载、drop全部capabilities）；模型容器没有这些宿主管理挂载或host PID能力，没有新增sudoers、设备控制或持久开机配置。复用原已验证`maxn_session.py`与`watch_docker_container.py`，通过systemd transient unit跟随`pld-rl`。失败回退逻辑在控制器中，**本次没有故意触发失败验证**，不冒称已演练回退。

## 零修正probe数据链路补齐（16:24 CST）

源码固定`bfa58595afcde615066750c831431ce0c8b60205`，客户端只读参照固定YAM `187f7dc5493696b16ddbb2596c8c14db4b3cd7a8`。新增原生manifest UUID、session侧车合并/去重、取消与错误隔离、独立probe协议审计和Thor CPU启动器。27项轻量回归通过（13动作/probe合同、10管家、4原生读取/审计）；另用客户端原录制器＋真实NumPy ProbeRuntime导出左右两条模拟记录，在工作站和Thor都通过完整读取/证据导出。未运行本地VLA前向或训练。

Thor检查共104行、16个正常probe回复、86个来源核验tick；两条记录都包含三视频索引。源文件、处理代码、输出及选中图片SHA复核一致；records/rewards为空，训练资格false。这是**模拟原生录制链路验收**，不计机器人成功率，也不代替跨IPC/真机测试。[检查回执](probe-data-integration.json)、[原生probe审计](probe-native-audit.json)、[CPU容器启动参数](probe-curator-launcher.json)保留。

离线源码发布：`/home/wuyan-lyj/thor/pld-grasp/releases/probe-curator-20261012-r1/code/`；源码tar SHA `6679f34d1ed9c6ddabad05ec5190c673e4dbd1eef7bbe50f3d4eb8607898cdf8`。检查与完整派生输出：`/home/wuyan-lyj/thor/pld-grasp/checks/probe-native-20261012-r1/`。工作站原始模拟夹具、全部检查版本：`/home/wuyan-lyj/condapi-data/rl/pld-probe-integration-20261012-r1/`；模拟包SHA `9a0c63fc4f41f283d8ef0049314ca2258e2e240ce06adcf5b775fadb83e92070`。不覆盖真实14集或已有派生版本。

复用已有数据处理镜像`sha256:2711020dd277e9114ef6296ddc0ed721de376a4b1156183ed39c6c1450a2439c`，无安装；以runc/2CPU/4GiB、无网络/无GPU/只读根、源码和原始文件只读挂载运行，输出独立目录。在线Pi镜像、权重、engine和握手均未修改。16:24重新读取Thor本机health/握手：healthy、原20w、仍仅off/probe、actor未加载；诊断SHA与上文相同。

## 下一步

服务端已经具备开始零修正probe的条件。客户端新版部署后按握手选择probe，先采2轮诊断，复制完整session（包括policy_events.jsonl），核查完整trace、原RTC来源/前缀、三视频与逐tick实际提交动作，并抽查容器误介入。用户最后确认先probe，再由Thor按Pi变化/FK位移选择正式探索范围，尚未执行十倍放大；客户端只如实报告已有限位/夹爪映射，没有步进上限就保留null，无需自行新增限速。正式合同/actor验收后再小批collect、判奖、SAC更新与Pi/eval对比。[当前训练方案](../../rl/pld-sac-block-v2/README.md)仍只用抓取1/0，整轮C/N不参与SAC更新。
