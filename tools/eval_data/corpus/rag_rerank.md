---
type: project
description: rerank 阶段设计：对 BM25 候选做二次排序
importance: 3.0
---
rerank 阶段的设计：

- 第一路 BM25 或向量召回宽候选（top-15 到 top-50）
- 第二路用重排模型或 LLM 对候选精细排序
- 重排只看相关性，不看关键词覆盖

宽召回加精排序的组合，比单路深召回性价比更高。
