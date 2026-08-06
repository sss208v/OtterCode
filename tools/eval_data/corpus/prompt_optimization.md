---
type: feedback
description: prompt 优化技巧：few-shot 示例与输出格式约束
importance: 2.0
---
用户反馈的 prompt 优化技巧：

- 给两三个 few-shot 示例比纯规则描述稳定
- 输出格式用明确标记约束，如 JSON 或列表
- 关键约束放开头，模型对前面的指令更敏感

内网模型指令遵循能力弱，提示词要写得更直白。
