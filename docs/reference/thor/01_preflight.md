# Thor 01 · 设备与安全预检（G0）

前置：[00 执行规则](00_start_here.md)。本章只读，不制盘、不重启正在运行的设备。目标是证明本机适用官方开发套件 USB ISO 流程。

## 1. 收集现场信息

正常情况先确认三件事：是否为官方开发套件、制盘电脑系统、U 盘和目标 NVMe 是否允许清空。新机无保留数据时无需备份空盘；已经确认过的信息不要重复询问。以下详细信息仅在旧机、多盘、定制载板或身份不明确时补查，不要求每项照片/序列号：

- Thor 整机型号、模块与载板型号：是 NVIDIA Jetson AGX Thor Developer Kit，还是第三方 IPC/定制载板？只有“Thor”名称不能确认适用性。
- 当前能否启动、系统是否已有数据/模型/配置、是否曾刷机、是否启用磁盘加密或定制安全启动。
- NVMe 数量、目标盘型号/容量/序列号、哪些盘绝对不能动；制盘 USB 的型号/容量/序列号及其原有数据。
- 制盘电脑的操作系统；显示器、键盘、原装电源是否就绪；是否有稳定互联网下载环境。

官方系统目标与安装介质由 [08 第 2 节](../../08_thor_edge_deployment.md#2-官方系统基线) 持有。官方发行版支持某个 Thor 模块，不等于开发套件 ISO 已适配所有第三方载板。第三方整机、改装存储/载板、设备身份不明时，G0 不通过：取得厂商 BSP 与刷写流程后重新审查，不能直接套用开发套件命令。

依据：[NVIDIA BSP 安装方式](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_bsp.html)、[Thor 板级刷写配置说明](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/FlashingSupportJetsonThor.html)（2026-09-07 核对）。

## 2. 两块磁盘、三个写入点

| 写入点 | 实际目标 | 风险 | 当场确认内容 |
|---|---|---|---|
| Etcher Flash | 插在制盘电脑上的 USB 整盘 | 原分区和文件被覆盖 | USB 型号/容量/身份，已备份或允许丢弃 |
| QSPI capsule update | Thor 板载启动固件 | 中断可能无法正常启动，恢复可能需重新刷机 | 型号与版本兼容、稳定供电、允许固件更新 |
| Install on NVMe | Thor 上的目标 NVMe | 按整盘数据将丢失处理 | 目标盘唯一、身份核对、备份可读、明确同意重装 |

USB 与 NVMe 不能混为同一个写入目标。对有重要数据的旧盘，先备份并确认备份可读；新盘由用户确认没有要保留的数据即可。固件更新直接按已匹配版本的安装器提示操作，不另设繁琐确认单。

有多个 NVMe 或无法确认安装器目标时必须停下；不要假设会弹出第二次确认或完整的磁盘选择器。需要拆盘时按整机厂商的断电/防静电规范由有能力的人操作，不指导带电插拔内部器件。

## 3. 材料与供电

主流程选本地显示器安装：显示器直接接 Thor HDMI/DP，避开 KVM；使用原装供电和正常散热。准备至少 16 GB USB，制盘电脑至少 25 GB 可用空间，更多空间用于备份/下载。不要把开发套件 ISO 当 Ubuntu Live 系统，也不要直接复制 ISO 到 U 盘当作制作完成。[官方 Quick Start](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/quick_start.html)

优先保留开发套件原有、已确认身份的 NVMe。本章不为任意容量 SSD 承诺默认分区方案兼容；更换小容量盘或特殊分区需求应返回板级刷写文档另审。

如果已有 Thor Linux，可由用户在 Thor 上执行只读检查；若尚无可用系统，不要求执行，记录实物标签/采购信息，不能凭空填入结果：

```bash
hostname
uname -m
cat /etc/nv_tegra_release
lsblk -o NAME,PATH,TYPE,SIZE,MODEL,SERIAL,TRAN,MOUNTPOINTS
findmnt /
```

不要为了读版本重启重要服务。UEFI 版本在之后的安全开机阶段拍照记录；已有记录不能当作当前读数。

## G0 简短完成检查

- [ ] 明确是适用本流程的官方开发套件；否则停止。
- [ ] USB 与 Thor 目标 NVMe 已区分；有要保留的数据才检查备份。
- [ ] 稳定供电、散热、显示器和输入设备就绪。
- [ ] 旧系统停止/重装窗口由用户确认，无需保护的运行任务已确认。
- [ ] 制盘电脑系统已知，下一步仅前往 [02](02_iso_and_usb.md)。
