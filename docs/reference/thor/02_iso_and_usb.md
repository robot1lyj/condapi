# Thor 02 · 官方 ISO 与启动 U 盘（G1）

前置：[G0](01_preflight.md) 通过。执行位置：制盘电脑，不是 3588，也不是 Thor 上的模型容器。风险：Flash 将覆盖所选 USB 整盘。

## 本机直接开始

当前 Ubuntu x86_64 制盘电脑所需的 Etcher `.deb` 已下载，文件身份见 [08 的软件记录](../../08_thor_edge_deployment.md#制盘电脑上的-etcher-安装包)。不必再去网页下载。准备在这台电脑安装时执行：

```bash
sudo apt install /home/wuyan-lyj/thor-system/tools/etcher-2.1.6/balena-etcher_2.1.6_amd64.deb
```

这是安装软件，会安装依赖但不会自动烧录。看一眼 APT 提示，正常安装即可；异常大范围移除/依赖冲突再停下来处理。不需要在 Thor 上执行这条命令。安装后从应用菜单打开 balenaEtcher，选现有 ISO 和准备好的 U 盘。软件已安装则直接打开，不重复安装。

## 1. 固定镜像来源与身份

正常路径：现有 ISO 已于 2026-09-07 检查完成，直接使用。未复制/修改、也无读取错误时不重复下载、重算哈希或追查签名；直接跳到第 3 节确认 U 盘。第 2 节留给换电脑复制、重新下载或报错的情况。

使用 [NVIDIA JetPack 下载页](https://developer.nvidia.com/embedded/jetpack/downloads) 的 JetPack ISO；与 [08 第 2 节](../../08_thor_edge_deployment.md#2-官方系统基线) 选定版本逐项匹配。使用网页中的实际下载链接，不拼接日期文件名、不使用来历不明网盘。网页换版时不要自动换用“最新”：先核对发行说明、硬件、UEFI 兼容性，再更新 owner。

08 中记录了现有 ISO 的工作站绝对路径、字节数及本地 SHA-256。文件不在 Git 中；另一个 agent/电脑未必能访问该路径。先确认文件存在，需要复制到另一台制盘电脑时，复制后重新计算 SHA-256。

本地 SHA-256 用于识别同一文件，不冒充 NVIDIA 签名认证。当前正常安装接受官方 HTTPS 下载产物及已完成的文件检查，无需额外寻找未提供的签名而暂停安装。若官方同时提供对应 ISO 的校验和，可顺手核对；不要拿 Debian 软件包哈希代替 ISO 哈希。

## 2. 文件检查（只读）

由指导 agent 把下列路径替换成制盘电脑上已确认的真实 ISO；保留引号。不要把网页 HTML、下载中的临时文件当 ISO。比较的是精确字节数，不是文件管理器四舍五入后的 GB。

Linux：

```bash
stat -c '%n | %s bytes' '/实际目录/实际镜像.iso'
file '/实际目录/实际镜像.iso'
sha256sum '/实际目录/实际镜像.iso'
```

macOS：

```bash
stat -f '%N | %z bytes' '/实际目录/实际镜像.iso'
file '/实际目录/实际镜像.iso'
shasum -a 256 '/实际目录/实际镜像.iso'
```

Windows PowerShell：

```powershell
Get-Item -LiteralPath 'C:\实际目录\实际镜像.iso' | Select-Object FullName, Length
Get-FileHash -LiteralPath 'C:\实际目录\实际镜像.iso' -Algorithm SHA256
```

预期：文件名/版本正确、下载完成、长度匹配；工作站同一下载产物或它的副本应匹配 08 的本地哈希。不同哈希不能自行更新记录“让它通过”；先查清下载是否换版、未完成或损坏。Linux/macOS `file` 应识别为 ISO/启动镜像而非 HTML；Windows 此处用长度/哈希及 Etcher 对文件的识别共同检查，不承诺哈希本身能鉴定镜像类型。

下载失败重试写入新文件/目录，不覆盖或停止已有下载，不删除旧 ISO。未验证文件不能用于烧录。

## 3. 锁定 USB 整盘

确认 USB 没有需保留的数据，在 Etcher 中按型号/容量选择它即可。只有存在多个相似磁盘或身份不清时，再安全移除无关外接存储、插入前后比对下面的只读列表：

- Linux：`lsblk -o NAME,PATH,TYPE,SIZE,MODEL,SERIAL,TRAN,MOUNTPOINTS`。
- Windows PowerShell：`Get-Disk | Select-Object Number,FriendlyName,SerialNumber,BusType,Size,IsBoot,IsSystem`。
- macOS：`diskutil list external physical`，对目标再用磁盘工具/`diskutil info` 查看身份。

盘符、`/dev/sdX`、`/dev/diskN` 都可能改变；不能沿用上次编号。把 Etcher 界面的型号/容量和现场盘身份交叉核对。USB 转接器不提供序列号时，用插拔差异、实物标识和容量共同确认；仍不唯一就停下。禁止选择电脑的启动盘/系统盘、备份盘或 Thor 的 NVMe。

## 4. Etcher 写入与校验

从 [balenaEtcher 官网](https://etcher.balena.io/) 取得对应电脑系统的版本；记录版本，不从广告或第三方下载站取安装包。

1. 打开 Etcher，选 **Flash from file**，选择已校验的 ISO。
2. 选 **Select target**，核对 USB 型号/容量，确保不是电脑系统盘。
3. 用户确认这只 USB 可以清空后，点 **Flash** 并处理系统提权提示。不要求额外抄写确认句或拍照。
4. 写入期间不拔盘、不关闭电脑、不并行格式化。等待写入和校验结束，保存成功界面；不能跳过校验把写入结束当成功。
5. 若主机提示“需要格式化才能使用”，取消；不要格式化刚写好的启动盘。安全弹出 USB。

选择 ISO → 选择目标 → Flash 是 [NVIDIA 官方主流程](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/quick_start.html#steps)；目标双重确认和备份是本项目额外安全要求。不额外做 `mkfs`、分区、`dd` 或“修复”启动分区。

## G1 放行与停止

- [ ] ISO 来源、版本、长度、SHA-256 和发布者校验状态已记录。
- [ ] 用户对具体 USB 的覆盖已确认，Etcher 校验成功并安全弹出。
- [ ] USB 写入错误、校验失败、身份不清：停止，见 [07](07_recovery_and_handoff.md)，不带病进入安装。
- [ ] G1 通过后进入 [03](03_bsp_install.md)，安装 NVMe 仍需另一次确认。
