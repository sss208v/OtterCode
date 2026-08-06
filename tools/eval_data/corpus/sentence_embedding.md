---
type: reference
description: 句子向量方案：embedding 模型选择与相似度计算
importance: 2.5
---
句子向量检索方案：

- 用 embedding 模型把句子编码成向量
- 相似度用余弦距离计算，最近邻检索取 top-k
- 中文场景优先选在中文语料上微调过的模型

向量方案能理解语义，但需要额外部署模型服务与向量索引。
