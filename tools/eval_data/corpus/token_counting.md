---
type: reference
description: token 估算方法：CJK 每字约 1.5 token，其余 4 字符一 token
importance: 2.0
---
不调接口的 token 估算规则：

- 中文每字约 1.5 token
- 英文与数字约 4 字符一 token
- 代码与符号按字符数近似折算

估算误差在 20% 以内即可用于预算决策，精确值以分词器为准。
