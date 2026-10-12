# pld-rl 推理容器上线（2026-10-12）

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

## 下一步

客户端按握手只启用off/probe，先约2集诊断，核查完整trace、原RTC来源/前缀、三视频与逐tick实际提交动作，并抽查容器误介入。提供真实限位、最大步幅与夹爪映射后，服务器导出正式零均值探索包；再安装到Thor，完成完整合同probe验收后小批collect、判奖、SAC更新和Pi/eval对比。[当前训练方案](../../rl/pld-sac-block-v2/README.md)仍只用抓取1/0，整轮C/N不参与SAC更新。
