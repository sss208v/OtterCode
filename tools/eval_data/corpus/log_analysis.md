---
type: project
description: 权限日志分析：permissions.log 的字段与审计用途
importance: 2.0
---
权限日志分析：

- 每次权限决策都写入 .otter/logs/permissions.log
- 字段含时间、工具、动作与决策结果（allow/deny/confirm）
- 审计时按用户与工具分组统计拒绝频率

日志只追加不覆盖，排查问题时先看最近一次 deny 记录。
