---
type: project
description: 项目数据库表结构记录，users 与 orders 表字段定义
importance: 3.0
---
项目使用 sqlite 存储业务数据，核心表：

- users 表：id 主键、username 唯一、email、created_at
- orders 表：id、user_id 外键、amount、status、paid_at

查询高频字段建了索引：users.username、orders.user_id。
改表结构前先在 memory 里记一笔，避免遗忘字段含义。
