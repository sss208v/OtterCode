---
type: reference
description: 部署 embedding 模型：模型选择、服务化与批处理
importance: 2.0
---
embedding 模型部署要点：

- 中文任务选中文优化的模型，如 bge 系列
- 服务化后用 HTTP 接口批量编码，减少调用次数
- 向量维度影响存储与检索速度，量化可压缩体积

部署后先用一组中文句子验证相似度排序是否符合直觉。
