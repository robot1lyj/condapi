# Thor 03 · 固件更新与 NVMe 系统安装（G2）

前置：[G0](01_preflight.md) 和 [G1](02_iso_and_usb.md) 均通过，用户已同意当前维护窗口。执行位置：Thor 本机。本文主流程使用显示器，不远程操控 3588。

## 1. 开机前检查

正常单盘开发套件只需确认该 NVMe 可重装，有重要数据则先备份。不反复索要序列号和照片。多个 NVMe、目标无法确定或改装存储时再展开核对，避免覆盖保护盘。

原装电源、散热与直连显示器接好，接键盘，插入验证过的安装 USB，再开机。不要按 Recovery 按钮开始正常 ISO 安装。接口位置按 [官方硬件布局](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/hardware_layout.html) 识别，不能把串口、Recovery 或其他接口混用。

一般会直接从 USB 启动，不需要先改 UEFI。需要手动选启动设备时，在 NVIDIA 启动画面按 `Esc`，进入 `Boot Manager` 选择对应 USB 启动项；不改无关设置。只有版本/启动异常时才记录 UEFI 版本并查矩阵。主流程依据 [官方 Quick Start](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/quick_start.html#boot-from-the-usb-stick-and-install-the-bsp-on-nvme)（2026-09-07 核对）。

## 2. 先核对固件兼容性，再处理 capsule 提示

正常开发套件使用选定 ISO 直接启动，按安装器提示处理即可。旧机版本特殊或启动异常时，再用实际 UEFI 版本对照 [官方 UEFI/ISO 矩阵](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/twa_uefi_iso_compatibility.html)。允许启动的组合也可能需要安装器的 capsule 更新。

不要降级尝试旧 ISO。r39.2.x UEFI 不支持直接用 r38.2/r38.4 ISO 降级；旧版重装的 Display Hand-Off 绕行也不是当前流程的常规前置步骤。出现未知版本/不支持组合时停止，不自行选一个“接近的版本”。

若出现明确的 **QSPI capsule update** 提示：

1. 确认是当前官方安装器的 QSPI 更新提示，不要求正常提示截图。
2. 使用稳定供电，按提示 `Y` 更新。
3. 等待进度及自动重启，期间不拔电、强制重启或拔掉安装介质。官方说明该更新会运行两遍；不能看到再次出现进度条就当作卡死。
4. 更新完成并返回安装流程后才继续。没有提示时不人工制造更新步骤；也不能据此填写“更新成功”，只记录“未出现提示”。

遗漏提示或不确定结果：先保存现场，确认没有正在写入固件；按 [官方 capsule 说明](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/quick_start.html#qspi-capsule-update-prompt-appears) 重新进入安装流程检查。该建议不授权在更新进行中断电。看不到进度/无法确认是否仍在写入时按 [07](07_recovery_and_handoff.md) 停止并求证。

## 3. 安装 NVMe：第二个磁盘覆盖确认点

出现 Jetson BSP 安装菜单后，目标为 **Install on NVMe**。不要选择 UFS/USB 或其他存储选项。此处按回车可能立即开始安装；不要依赖后续还有确认对话框。

按回车前只确认一件事：Thor 的这块目标 NVMe 允许覆盖安装。已有重要数据先备份，新机无保留数据直接继续，不用填写确认表。

目标不明确或现场菜单与手册不同：停在菜单，不按回车试探。用户确认后才开始安装。保持供电、散热，记录关键界面和错误；耗时只是参考，不能以“超过十分钟”为强制重启依据。

## 4. 完成安装后移除 USB

只有在安装明确完成、出现完成/重启提示或确定已进入安装后的重启阶段后，才移除安装 USB，使下一次从 NVMe 启动。不能在固件更新或 NVMe 写入进行中拔盘。若再次进入安装菜单，不再选 Install，先确认完成记录与启动设备，避免重复覆盖。

随后完成 Ubuntu `oem-config`：键盘、时区、用户名与密码由用户设置。密码不贴进聊天/仓库；创建独立账号，不抄教程用户名/密码。网络可先使用经确认的独立管理网络，直连 3588 的网口配置留到后续。

## 无显示器、异常与恢复边界

本组不把无显示器安装当默认。若只能 headless，暂停主流程，指导 agent 先核对官方 [Quick Start 的 headless 分支](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/quick_start.html) 和 [UEFI 38.0.0 串口显示问题](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/twa_headless_on_uefi-38-0-0.html) 的实际入口；链接/内容变动时从官方导航重新找到对应页。

安装前的 Debug-USB 控制台与首次启动配置的普通 USB-C 并非同一个接口阶段，不能把某个固定 `/dev/ttyACM0` 当通用答案。没有读清对应版本接口/串口设置、无法辨认安装菜单时停止，不盲按方向键或确认键。

G2 放行：固件结果可解释、NVMe 安装明确完成、安装盘已在安全阶段移除、首次配置完成或正常进入该界面。接下来按 [04](04_host_acceptance.md) 验证系统，不能只凭桌面显示就说安装全链路成功。
