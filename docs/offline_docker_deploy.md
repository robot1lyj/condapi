# 离线服务器 Docker 部署指南（openpi_dev）

本指南面向“离线服务器 + Docker 部署”的场景：在有网机器构建镜像与准备项目文件，迁移到离线服务器后直接启动即可使用。

## 1. 有网机器：拉取代码与构建镜像
```bash
git clone --recurse-submodules <repo_url> openpi
cd openpi

# 构建自包含开发镜像（含依赖 + 代码）
docker build -f scripts/docker/dev.Dockerfile -t openpi_dev .

# 导出镜像文件，准备迁移
docker save openpi_dev -o openpi_dev.tar
```

如需离线使用模型权重，建议提前下载到统一目录（例如 `/data/openpi_cache`），之后整体拷到离线服务器并挂载为 `/openpi_assets`。

## 2. 有网机器：打包项目文件
```bash
cd ..
tar -czf openpi_repo.tar.gz openpi
```

## 3. 迁移到离线服务器
将 `openpi_dev.tar` 与 `openpi_repo.tar.gz` 拷贝到离线服务器（U盘/内网均可）。

## 4. 离线服务器：导入与启动
```bash
# 导入镜像
docker load -i openpi_dev.tar

# 解压项目
tar -xzf openpi_repo.tar.gz
cd openpi

# 启动容器（默认挂载当前代码目录 /app，默认端口 6666）
scripts/docker/run_dev.sh
```

默认会把仓库挂到 `/app`，因此你可以在宿主机修改代码并立即生效。  
脚本默认以当前宿主机用户 UID:GID 运行容器，避免宿主机文件被 root 覆盖权限。  
若需要后台运行：`scripts/docker/run_dev.sh -d`。

## 5. 常用启动命令（端口 / GPU / 挂载）
端口默认是 `6666`，如果你要显式指定或改成其他端口：
```bash
# 宿主机 6666 -> 容器 6666
scripts/docker/run_dev.sh -p 6666:6666

# 改成 7000
scripts/docker/run_dev.sh -p 7000
```

指定 GPU 数量或设备：
```bash
# 使用 2 张 GPU
scripts/docker/run_dev.sh --gpus 2

# 使用指定 GPU 设备
scripts/docker/run_dev.sh --gpus 0,1

# 不使用 GPU
scripts/docker/run_dev.sh --no-gpu
```

映射额外目录或文件（可重复）：
```bash
scripts/docker/run_dev.sh \
  --bind /data/models:/openpi_assets/models \
  --bind /data/config.yaml:/app/config.yaml
```

替换默认代码挂载目录：
```bash
scripts/docker/run_dev.sh --mount /path/to/openpi
```

指定用户/权限（避免宿主机文件被 root 改权限）：
```bash
# 显式指定 UID:GID
scripts/docker/run_dev.sh --user 1000:1000

# 如果确实需要 root
scripts/docker/run_dev.sh --as-root
```

如果之前已经被 root 改了权限，可在宿主机修复：
```bash
sudo chown -R "$USER":"$USER" /share/home/linyongjia/lyj/openpi
```

## 6. 在容器中运行服务/实验
进入容器后可按需运行（注意端口与映射一致）：
```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=/openpi_assets/checkpoints/pi05_libero \
  --port 6666
```

也可以在启动时直接执行命令：
```bash
scripts/docker/run_dev.sh -p 6666 --gpus 2 -- \
  uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=/openpi_assets/checkpoints/pi05_libero \
  --port 6666
```

## 7. 数据缓存目录（权重/下载）
默认映射 `~/.cache/openpi` 到容器 `/openpi_assets`。  
如需自定义目录：
```bash
OPENPI_DATA_HOME=/data/openpi_cache scripts/docker/run_dev.sh --gpus 1
```

## 8. 需要新增挂载时的处理（方案 A：保存环境并重建）
Docker 不能给“已存在的容器”新增挂载，所以需要保存当前环境并重建容器。流程如下：
```bash
# 停止当前容器
docker stop openpi_dev

# 把当前容器保存成新镜像（保留已安装的依赖/修改）
docker commit openpi_dev openpi_dev_custom:2025-01-15

# 删除旧容器（释放名称）
docker rm openpi_dev

# 用新镜像重建，并加入新的挂载
IMAGE=openpi_dev_custom:2025-01-15 \
scripts/docker/run_dev.sh \
  --mount /share/home/linyongjia/lyj/openpi \
  --bind /new/data:/data \
  --gpus 2
```

说明：
- `docker commit` 只会保存容器内文件系统，不会包含挂载的目录。
- 镜像标签建议加日期，方便追溯与回滚。
