---
type: project
description: RAG 检索增强生成流水线的整体架构
importance: 4.0
---
RAG 流水线分三步：索引构建、检索召回、生成回答。

- 索引：文档切块后做 tokenize，建立倒排索引
- 检索：query 分词后召回 top-k 候选块
- 生成：把候选块拼进上下文交给大模型作答

当前项目在记忆检索上采用了 BM25 预筛加 LLM rerank 的轻量混合方案。
