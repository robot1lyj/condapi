# Thor 05 · 容器运行时与 GPU 验收（G4）

前置：[G3](04_host_acceptance.md) 的系统/GPU 检查通过，缺项有记录。执行位置：Thor 宿主。拉镜像会占用磁盘与网络，测试容器会使用 GPU；需确认空闲资源和维护窗口。

## 1. 先审计，ISO 安装不重复覆盖运行时

正常 ISO 安装先看 `sudo docker info` 和 `docker compose version`，运行时正常就直接做第 3 节一次 GPU 测试。下面的完整清单和第 2 节只在缺项或错误时使用；不把全量包审计、配置备份作为已有正常运行时的必做步骤。

```bash
sudo docker version
sudo docker info
sudo docker ps -a
docker compose version
dpkg-query -W nvidia-container-toolkit
nvidia-ctk --version
systemctl is-active docker
systemctl is-enabled docker
```

用完整输出核对服务端版本、`nvidia` runtime、已有容器、存储目录/剩余空间及开机启动状态。不要把客户端版本当服务端健康证明。默认用 `sudo docker`；加入 docker 组相当于授予 root 级能力，不作为必须操作。[Docker 权限说明](https://docs.docker.com/engine/install/linux-postinstall/)

官方 ISO 路线已包含 Docker/Container Toolkit；实际包和功能仍须检查。只有缺失或来自非 ISO 安装时才处理以下分支。[Thor Docker Setup](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_docker.html)

## 2. 缺项的条件处理，不整段照抄

指导 agent 先检查现有 APT 源和已安装包（不分享其中的认证信息），再确认与目标 BSP 一致的候选版本。记录要变更的软件包和 Docker 配置；有正在运行的容器时不重启服务。

- **Docker 缺失**：按 [Docker 官方 Ubuntu APT 安装流程](https://docs.docker.com/engine/install/ubuntu/) 核对 Ubuntu 版本与 ARM64 支持，审查现有冲突包后逐项安装 Engine、CLI、containerd、Buildx 和 Compose 插件；不得无审计执行文档中移除冲突包的整段命令。镜像/容器数据不得清空。
- **Toolkit 缺失**：优先使用与本机 JetPack 匹配的 NVIDIA APT 仓库；Thor 指南使用 `nvidia-container` 包。先检查候选版本和模拟安装依赖，不能混用任意 CUDA/Ubuntu 源“装上就算”。
- **只有 Compose 缺失**：按已确认 Docker 官方源安装匹配的 `docker-compose-plugin`，不重新安装整个运行时。
- **已安装但无 `nvidia` runtime**：先保存 `/etc/docker/daemon.json` 原文件（若不存在记录不存在）与 Docker 服务配置到新建备份目录。用户同意修改和重启后，才逐条执行下列配置命令；前一条失败即停。

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo docker info
```

该配置方式来自 [NVIDIA Container Toolkit 文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker)。执行后审查配置差异，保留原有网络、日志、数据目录等设置；不要用教程 JSON 整体覆盖现有文件。本手册通过显式 `--runtime=nvidia` 请求 GPU，不要求改全局 default-runtime，也不需要 `--privileged`。

如果运行时服务未启用，先排除配置错误，再经用户同意启动/设置开机启动；无法解析配置、APT 依赖冲突或候选包不匹配时停下记录。不得使用 `curl | sh`、强制忽略依赖或删除 `/var/lib/docker` 作为修复办法。这些缺项分支需现场审核，不是本次已经测试过的安装脚本。

## 3. 基础 GPU 容器 smoke

官方 Pi 教程基础镜像与本地构建标签的区别见 [08 第 3.2 节](../../08_thor_edge_deployment.md#32-官方容器的具体口径)。本测试使用该 NVIDIA PyTorch 基镜像仅验证容器 GPU；不是决定把 JAX 模型转成 PyTorch。

确认磁盘容量和下载许可后，逐条执行并保存输出：

```bash
sudo docker pull --platform linux/arm64 nvcr.io/nvidia/pytorch:26.05-py3
sudo docker image inspect nvcr.io/nvidia/pytorch:26.05-py3 --format '{{.Os}}/{{.Architecture}} {{json .RepoDigests}}'
```

预期平台为 `linux/arm64`；记录实际 digest。若没有 ARM64 manifest、拉取失败、空间不足，立即停止；不要改成 AMD64 仿真、关闭 TLS 校验或换不明第三方镜像。标签可变，运行前以记录的 digest 固定本次测试对象。

下面是待现场执行的模板：指导 agent 必须把 `填入已核对的digest` 替换为上一步实际 `sha256` 值；不能执行占位符，也不要伪造 digest。测试不挂载源码、模型、磁盘设备或 Docker socket，不开放端口，不需要 3588。

```bash
sudo docker run --rm -i --pull=never --network=none --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  nvcr.io/nvidia/pytorch@sha256:填入已核对的digest python3 - <<'PY'
import torch
assert torch.cuda.is_available(), 'GPU is unavailable'
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
x = torch.ones((256, 256), device='cuda', dtype=torch.float32)
y = x @ x
torch.cuda.synchronize()
assert bool(torch.isfinite(y).all().item())
assert float((y - 256).abs().max().item()) == 0.0
print('PASS: container GPU matrix multiplication')
PY
```

测试容器正常退出后由 `--rm` 自动移除，仅移除这次无挂载的临时容器可写层；镜像保留。终端输出需保存，失败时不要清理镜像/缓存。若用户需要保留故障容器现场，执行前去掉 `--rm` 并记录容器 ID，由用户决定何时清理。

## G4 放行

- [ ] GPU 运算断言通过，不仅是 `nvidia-smi` 或 `cuda.is_available()`。
- [ ] 宿主/镜像版本、架构、digest、测试退出码和日志已记录。
- [ ] Docker/Compose/Toolkit 可用，修改的配置有备份，未影响无关容器。
- [ ] 结论只能写“容器 GPU 基础路径通过”，不能写“JAX/Pi/YAM 已通过”。下一步见 [06](06_pi_service_and_link.md)。
