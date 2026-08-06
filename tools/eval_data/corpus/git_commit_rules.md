---
type: project
description: git commit 消息规范，type(scope) 前缀约定
importance: 3.0
---
提交信息统一使用 conventional commits 风格：

- feat：新功能，如 feat(retrieval): 支持中文分词
- fix：缺陷修复，如 fix(bm25): 修正 idf 计算
- refactor：重构，如 refactor(tools): 合并重复函数
- docs、test、chore 用于文档、测试与杂项

一条提交只做一个改动，避免混入无关修改。
