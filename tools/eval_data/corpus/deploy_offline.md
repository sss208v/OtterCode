---
type: reference
description: 完全离线部署清单：依赖、模型权重、镜像的搬运方式
importance: 3.5
---
完全离线部署检查清单：

- Python 依赖：联网机器 pip download，内网离线安装
- 大模型权重：外网下载后拷贝进内网服务器
- Docker 镜像：docker save / load 搬运
- 分词词典等资源随包分发，运行时不再联网

所有软件更新都走离线包，禁止运行时访问外网。
