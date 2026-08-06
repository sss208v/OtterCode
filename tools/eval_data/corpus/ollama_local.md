---
type: reference
description: ollama 本地模型运行：模型拉取与服务接口
importance: 3.0
---
ollama 本地跑模型的流程：

- 联网机器 ollama pull 模型，再拷贝模型目录进内网
- ollama serve 启动服务，默认监听 11434 端口
- 通过 OpenAI 兼容接口调用：/v1/chat/completions

模型文件较大，拷贝时注意磁盘空间；量化版能显著减小体积。
