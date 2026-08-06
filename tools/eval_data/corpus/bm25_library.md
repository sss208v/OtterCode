---
type: reference
description: bm25s 与 rank_bm25 库对比，参数与性能差异
importance: 2.0
---
BM25 库选型对比：

- rank_bm25：经典实现，接口简单，适合小语料
- bm25s：带持久化索引，支持自定义分词器，适合大语料
- 两者都是标准 BM25 公式，k1 默认 1.5、b 默认 0.75

本项目手写了 BM25 打分，未引入外部库，参数与公式一致。
