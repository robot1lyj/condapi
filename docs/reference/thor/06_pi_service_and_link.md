# Thor 06 · Pi 系列服务与网线直连（G5，待工程实施）

前置：[G4](05_container_runtime.md) 通过。这里不是已验收的一键教程：本项目 Pi 系列生产镜像、锁定依赖、Compose 与完整精度工具尚待实现，不能让安装 agent 猜出版本后直接上线。

推进顺序：先解决容器依赖并做一次 JAX GPU 测试，再用少量代表性 YAM 样本跑通原 checkpoint 和 LoRA、检查输出，随后联调只回传不执行动作的 WebSocket。正式发布记录、长时间稳定性/性能测试后置；逐层精度定位在转换、量化或发现输出差异时展开，不作为安装系统的前置任务。原 JAX 参考结果保留，不能跳过格式转换后的精度比较。

## 1. 必须先解决的兼容性阻断项

2026-09-07 静态核对：[pyproject.toml](../../../pyproject.toml) 当前要求 Python 3.12、`jax[cuda12]==0.5.3`、`flax==0.10.2`、`orbax-checkpoint==0.11.13`。它是现有项目依赖，不是已经验收的 Thor CUDA 13 锁文件。不能直接 `pip install -e .` 让解析器在候选镜像里改写 GPU 栈；也不能无依赖安装之后不检查缺包/不兼容。

工程 agent 先在独立候选镜像中审计 ARM64 wheel、Thor GPU kernels、驱动/CUDA 用户态兼容性、JAX/Flax/Orbax 恢复及模型算子；按 [JAX 官方安装说明](https://docs.jax.dev/en/latest/installation.html) 选择并验证，不把“Linux aarch64 支持”理解成“我们的 Thor LoRA 全链路已测试”。使用 conda/pip，不使用 uv，不在宿主安装模型 Python 环境，不更改服务器训练依赖来迁就 Thor。

交付物至少包含：系列 Dockerfile、固定版本依赖、代码 commit、基础/最终镜像 digest、构建日志和测试报告。未交付则停在“G4 完成、G5 待实施”。[官方 Pi0.5 教程](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/) 的 Torch/TRT 路径不替代这项工作。

## 2. 先验证 JAX GPU，再恢复原模型

在已完成依赖审计的候选 Pi 容器内执行下列小测试；失败即停，不接受 CPU 回退。执行位置是容器 shell，编译缓存仅写允许的运行目录：

```bash
python - <<'PY'
import jax
import jax.numpy as jnp
print('jax:', jax.__version__, 'devices:', jax.devices())
assert jax.default_backend() == 'gpu', 'CPU fallback is not accepted'
x = jnp.ones((256, 256), dtype=jnp.float32)
y = jax.jit(lambda a: a @ a)(x)
y.block_until_ready()
assert all(d.platform == 'gpu' for d in y.devices())
assert bool(jnp.all(jnp.isfinite(y)))
assert float(jnp.max(jnp.abs(y - 256))) == 0.0
print('PASS: JAX GPU JIT matrix multiplication')
PY
```

这只证明小算子，不证明 Pi 推理。之后按 [08 精度闸门](../../08_thor_edge_deployment.md#6-端侧验收闸门) 恢复完整 JAX/LoRA checkpoint，使用训练配置与匹配的 norm。原权重、LoRA、tokenizer 和 golden 只读挂载，不与转换产物混放。

特别注意：[policy_config.py](../../../src/openpi/policies/policy_config.py) 会按目录内是否存在 `model.safetensors` 自动选择后端。原 JAX 目录中混入该文件可能走错后端。该加载器的 JAX 分支还显式以 BF16 恢复；因此“原生 JAX”也不等于全 FP32，golden 必须记录实际恢复/计算精度，不要为验收临时偷换 dtype。

精度比较使用同一份输入及真实噪声数组，不仅是相同随机种子。固定 prompt、图像预处理、norm、去噪步数和 action horizon；逐步/完整 action 的阈值在测试前确定，不用跑完后的误差倒推通过标准。YAM 合同和操作 smoke 的 owner 为 [04](../../04_data_contracts.md) 与 [05](../../05_inference_and_rollout.md)。

LoRA-aware 转换与 FP32 中间产物若尚未实现和验证，禁止直接拿现有转换脚本替代原 JAX。全量微调也不能免除精度对照；FP8/NVFP4 不是安装必选项。

## 3. Pi 系列服务的工程交付检查

容器粒度严格按 [08 第 3.1 节](../../08_thor_edge_deployment.md#31-按模型系列隔离容器的部署约定)：Pi 系列一个 Compose 服务，配置/checkpoint 选模型，默认一次加载一个；其他系列独立。缓存按模型和运行环境隔离，日志按模型/批次保留。切换同系列模型时重启、预热、重新验收，不声称支持热切换。

上线前必须审查：

- 显式 GPU runtime/资源声明、只读资产挂载、可写缓存与日志目录、非特权运行和端口绑定；不透传相机/机械臂设备或 Docker socket。[Compose GPU 支持](https://docs.docker.com/compose/how-tos/gpu-support/)
- 所有变量展开后的 Compose 配置可通过 `docker compose config`；它只检查配置，不证明 GPU 或模型正确。
- 首次加载、真实推理、编译预热完成之后才报告 ready；现有服务 `/healthz` 的 HTTP 成功不能替代模型就绪与精度验收。
- 正式模型启动入口沿用 [05 的服务命令](../../05_inference_and_rollout.md#5-thor3588-网络推理通道)，显式传 YAM 配置和 checkpoint，不能落入默认 ALOHA_SIM。首轮 `rtc_mode=off`，不改旧推理/回退边界。
- 镜像与模型摘要、norm、配置、测试输入/输出、时延/内存报告绑定；明确重启策略、日志轮转、失败不接收生产请求、前一版本回退方法。

## 4. Thor ↔ 3588 直连网线

执行范围仅 Thor；3588 的 IP、协议与控制安全状态由负责人提供并操作。本手册不生成 3588 配置命令。

1. 从 G3 网卡清单和现场插拔链路状态确认 Thor 直连网卡；保存现有连接配置、地址和路由。不要套用教程的接口名。确认本地显示器/备用管理通道可用后才改网络，避免把 SSH 管理链路改断。
2. 与控制侧约定互不冲突的静态地址、相同子网掩码、policy 端口；检查该网段不与管理网络、VPN、Docker 子网重叠。直连网卡不配置默认网关/DNS，保留独立上网路径；初版不启用巨帧、不做网桥或 IP 转发。
3. 指导 agent 根据实际 NetworkManager/其他网络管理器生成 Thor 单侧变更，先展示差异、备份与回退，再经用户确认应用。网卡/IP 未确定前不提供可误用的写配置命令。
4. 初轮模型测试只在 Thor 本地暴露；通过后把容器 policy 端口明确发布到 Thor 直连 IP，而非所有宿主接口。容器内监听 `0.0.0.0`；3588 连接 Thor IP，不连接容器私网地址。[Docker 端口发布](https://docs.docker.com/engine/network/port-publishing/)
5. 检查 Docker 实际防火墙/NAT 和所有接口的暴露情况，不只看 UFW 配置。直连不等于认证；本服务不应暴露到公网/办公网。
6. 先由双方确认链路/地址；ping 只算网络检查。再由控制侧负责人在禁止机械臂执行动作的安全模式下发送真实 observation，Thor 返回 action，仅记录不驱动。检查三路图像、14D state、prompt、`(50,14)` 有限输出、norm/单位合同、超时和断线行为，记录往返时延。

G5 放行必须同时有本地原 JAX 精度证据与真实 WebSocket observation/action 往返证据。控制侧执行、限位、急停和最终机械臂任务验收不在本组安装权限内，不能因推理输出有限就允许机械臂运动。
