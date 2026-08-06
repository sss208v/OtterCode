---
type: project
description: 检索效果评测方法：Recall、MRR 等指标的定义与口径
importance: 3.0
---
检索效果评测口径：

- Recall@k：top-k 里命中相关文档的比例，预筛场景最关键
- MRR：第一个命中结果的排名的倒数，衡量排得够不够靠前
- Precision@k：top-k 里相关文档的占比

评测集要人工标注，query 覆盖同义词、术语与停用词等硬场景。
