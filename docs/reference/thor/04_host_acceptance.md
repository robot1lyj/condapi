# Thor 04 · 首次启动与宿主验收（G3）

前置：[G2](03_bsp_install.md) 通过。执行位置：用户已登录的 Thor 宿主终端，不是制盘电脑、服务器或容器。本章默认只读。

## 正常情况只做这组

```bash
cat /etc/nv_tegra_release
findmnt /
nvidia-smi
docker --version
docker compose version
```

版本正确、从 NVMe 启动、GPU 正常显示，便进入 [05 容器 GPU 测试](05_container_runtime.md)。Docker/Compose 缺项按下一章补齐，不影响系统已经装好的结论。以下是异常时的补充诊断，不要求正常安装逐条跑完、留图或为了验证再重启。

## 1. 系统与启动盘

需要详细诊断时执行；只保留相关输出，命令失败不要忽略：

```bash
hostname
date -Is
uname -m
uname -r
cat /etc/os-release
cat /etc/nv_tegra_release
findmnt -no SOURCE,FSTYPE,OPTIONS /
lsblk -o NAME,PATH,TYPE,SIZE,MODEL,SERIAL,TRAN,MOUNTPOINTS
df -h /
```

核对 `aarch64`、[08 的系统版本基线](../../08_thor_edge_deployment.md#2-官方系统基线)，并把 `/` 的设备追溯到 G0 确认的 NVMe。映射设备/LVM/加密层出现时需要追溯底层盘，不能只按设备字符串包含 `nvme` 判定。根文件系统若来自安装 USB、只读异常或空间不足，G3 不通过。

## 2. GPU、软件包与运行状态

```bash
nvidia-smi
dpkg-query -W nvidia-l4t-core nvidia-container-toolkit
docker --version
docker compose version
systemctl is-active docker
sudo nvpmodel -q
```

`sudo nvpmodel -q` 仅查询；本章不执行 MAXN 切换、锁频或 railgating 修改。保存 GPU 名称、驱动、当前功耗模式。`nvidia-smi` 的 CUDA 数字表示驱动支持信息，不能证明宿主装了相同版本的 CUDA Toolkit，更不能证明 JAX 已可用。[NVIDIA nvidia-smi 说明](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

包/工具缺失时记录缺项；`jetson_release` 来自可选辅助工具，找不到它不能单独判定 BSP 失败，也不需要为了取得版本读数临时装包。宿主 GPU 不可用则先定位系统问题；Docker/Compose 缺失交给 [05](05_container_runtime.md) 的条件分支处理。

## 3. 日志与联网（只读）

```bash
ip -br link
ip -br addr
ip route
timedatectl status
systemctl --failed
sudo journalctl -b -p err --no-pager
```

保存当前启动的错误，区分无关告警与存储/GPU/固件故障；不能“零告警”机械放行，也不能忽略 I/O 错误。分享日志前遮盖无关 IP、用户名、设备序列号等信息，绝不贴凭据。

管理网络先证明可用并记录，不改现有默认路由，不把教程中的接口名套过来。直连 3588 的数据网口留到 G5，那里不应充当 Internet 默认网关。

## 4. 这时不要做什么

- 不安装 Ubuntu 的 `nvidia-cuda-toolkit`，不运行桌面显卡 `.run` 驱动覆盖 Jetson 驱动。
- 不执行一揽子 `apt upgrade`、`do-release-upgrade` 或换源来“升级到目标 BSP”。官方明确进入 r39.2.x 不能照旧 r38.x 的 APT 升级步骤完成。[BSP Upgrade](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_bsp.html#bsp-upgrade)
- 不把“ISO 已安装”当成宿主已安装全部 JetPack 开发组件。模型依赖属于容器；确需宿主组件再核对 [JetPack SDK Setup](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_jetpack.html)。
- 不通过重装系统、修改安全启动或清理磁盘掩盖诊断结果。

## G3 完成标准

开头的基础检查正常即可进入 [05](05_container_runtime.md)。扩展日志和重启只用于排查实际问题；不额外做一轮形式化验收。出现存储/GPU 等关键错误再修复，不带着已知故障运行模型。
