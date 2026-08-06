---
type: project
description: 技能自进化机制，usage.jsonl 审计与版本快照
importance: 3.0
---
技能自进化循环在每次回复后运行：

- 判断反馈是否值得沉淀为可复用技能
- 写入 usage.jsonl 审计记录，保留 provenance 与版本快照
- 新技能必须经过多次复用验证，一次性任务不建技能

进化发生在 .otter/skill-evolution 目录下，属于运行时资产。
