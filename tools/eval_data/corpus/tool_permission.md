---
type: project
description: 工具权限模式说明：default、plan、acceptEdits、bypassPermissions、dontAsk
importance: 3.5
---
工具权限模式共五种：

- default：危险命令与新建文件需要用户确认
- plan：只允许写计划文件，其余编辑拒绝
- acceptEdits：编辑自动放行，越界仍拒绝
- bypassPermissions：跳过确认，硬黑名单仍拦截
- dontAsk：所有本应确认的操作自动拒绝

子代理统一使用 acceptEdits，不再继承越权模式。
