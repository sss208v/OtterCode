---
type: project
description: memory 文件格式约定：frontmatter 字段与正文规范
importance: 3.5
---
memory 文件用 Markdown 加 YAML frontmatter：

- type：user / feedback / project / reference
- description：一句话摘要，检索与索引展示用
- importance：数值，淘汰排序参考

正文前若干行会参与检索，重要信息放前面。文件全部 UTF-8 编码。
