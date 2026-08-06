---
type: reference
description: Docker 部署内网大模型，镜像离线导入方式
importance: 3.0
---
内网环境没有外网，Docker 镜像需要离线搬运：

- 联网机器上 docker save 导出镜像 tar 包
- 内网机器 docker load 导入
- 构建镜像时把依赖装好，运行时容器内不联网

大模型权重同样走离线拷贝，用 volume 挂载进容器。
