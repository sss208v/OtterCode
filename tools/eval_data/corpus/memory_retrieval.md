---
type: project
description: 记忆检索流程：BM25 预筛 top-15 再加 LLM rerank 选最多 5 条
importance: 4.0
---
记忆召回流水线：

1. 扫描 memory 目录的文件头，缓存 description 与正文 token
2. BM25 预筛得到 top-15 候选
3. 候选不超过 5 条时跳过模型，直接返回
4. 超过 5 条时让模型从清单里 rerank，最多选 5 条

预筛漏掉的记忆后续永远不会被看到，所以预筛的召回率是关键指标。
