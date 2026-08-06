"""共享文本分词器（BM25 等关键词检索用）。

中英混排检索的 token 化规则（零第三方依赖，纯标准库）：
- 英文/数字按字母数字连续段切分并小写化；
- 连续汉字块展开为相邻两字（2-gram），1-2 字的短块直接保留；
- 过滤中文功能词停用词表（STOP_TOKENS），如 "帮我"、"一下"；
- 长度 >3 且以 s 结尾的英文词做单数还原（skills -> skill）。

memory.py 的记忆检索与 skills.py 的技能检索共用本模块，避免两处
分词逻辑漂移；BM25 打分公式与参数常量仍各自引用 agents.memory。
"""

from __future__ import annotations

import re

# 英文按字母数字连续段；汉字按 1-2 字捕获，长块再由 tokenize 展开为 2-gram。
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]{1,2}")

# 中文功能词停用词表：2-gram 方案下高频出现的无意义组合，检索时过滤。
STOP_TOKENS = frozenset({
    "请帮",
    "帮我",
    "我做",
    "做一",
    "一次",
    "一下",
    "这个",
    "那个",
    "一个",
    "用户",
    "问题",
    "回答",
    "生成",
    "使用",
    "需要",
})


def _normalize(text: str) -> str:
    """小写化并把下划线/连字符转成空格，让代码标识符按词切开。"""
    return str(text or "").lower().replace("_", " ").replace("-", " ")


def tokenize(text: str) -> list[str]:
    """返回有序 token 列表（文档与查询两侧共用）。

    顺序对 BM25 无影响（打分按词频统计），但保持确定性与单测可断言。
    """
    raw = _normalize(text)
    tokens = [m.group(0) for m in _TOKEN_RE.finditer(raw)]
    for chunk in re.findall(r"[\u4e00-\u9fff]+", raw):
        if len(chunk) >= 2:
            tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
    expanded: list[str] = []
    for token in tokens:
        if not token.strip() or token in STOP_TOKENS:
            continue
        expanded.append(token)
        if len(token) > 3 and token.endswith("s"):
            expanded.append(token[:-1])
    return expanded
