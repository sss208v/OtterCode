---
type: reference
description: sqlite 数据库备份与恢复的具体步骤
importance: 2.0
---
sqlite 备份恢复步骤：

1. 备份：sqlite3 app.db ".backup backup.db"
2. 恢复：把 backup.db 复制回原位，或用 ".restore" 命令
3. 热备优先用 backup API，直接拷贝文件可能损坏 WAL

备份前先执行一次 VACUUM 可以减小文件体积。
