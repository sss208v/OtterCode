#!/usr/bin/env python3
"""BM25 中文 tokenizer A/B 评测工具。

对比两种分词方式对中文 BM25 检索效果的影响。核心原则是单一变量：
只换分词器，BM25 打分公式、k1/b 常量、语料、query、gold 标注全部固定。

- ascii : 改造前基线（agents/memory 的纯 ASCII 正则），中文被完全丢弃
- bigram: 现网方案（agents/tokenizer 的字符 2-gram + 停用词）

公平性说明：
- ascii 精确复刻改造前 memory 流水线的行为，仅作基线展示；
- BM25 打分复用 agents.memory 的公式与常量（BM25_K1/BM25_B），不重新实现。

用法：
    python tools/eval_tokenizer_ab.py                 # 跑全部可用模式
    python tools/eval_tokenizer_ab.py --mode ascii,bigram
    python tools/eval_tokenizer_ab.py --topk 15
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Callable

# 直接运行脚本时 sys.path[0] 是 tools/，需补上仓库根才能 import agents。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.frontmatter import parse_frontmatter          # noqa: E402
from agents.memory import BM25_B, BM25_K1                 # noqa: E402
from agents.tokenizer import tokenize                     # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent / "eval_data"
CORPUS_DIR = EVAL_DIR / "corpus"
GOLD_PATH = EVAL_DIR / "gold.json"
DEFAULT_REPORT = Path(__file__).resolve().parent / "eval_report.md"

VALID_CATEGORIES = {"normal", "jargon", "mixed", "stopword", "synonym", "short"}

# ─── 分词器 ────────────────────────────────────────────────


def tokenizer_ascii(text: str) -> list[str]:
    """现状基线：与 agents.memory._tokenize 完全一致的纯 ASCII 切分，中文被丢弃。"""
    return re.findall(r"[a-z0-9]+", text.lower())


def tokenizer_bigram(text: str) -> list[str]:
    """现网方案：直接复用共享 tokenizer（2-gram + 停用词），避免实现漂移。"""
    return list(tokenize(text))


def available_modes(requested: str | None) -> list[str]:
    """解析 --mode。"""
    all_modes = ["ascii", "bigram"]
    if requested:
        wanted = [m.strip().lower() for m in requested.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in all_modes]
        if unknown:
            raise SystemExit(f"未知模式: {', '.join(unknown)}（可选: {', '.join(all_modes)}）")
        return wanted
    return all_modes


# ─── 语料与评测集加载 ─────────────────────────────────────


def load_corpus() -> list[dict]:
    """读取种子语料。

    doc_id = 文件名（去 .md），文档文本 = 文件名 + description + 正文，
    与 memory 侧 bm25_topk 的文档构成方式对齐。
    """
    docs: list[dict] = []
    for f in sorted(CORPUS_DIR.glob("*.md")):
        raw = f.read_text(encoding="utf-8")
        try:
            parsed = parse_frontmatter(raw)
            text = f"{f.stem} {parsed.meta.get('description') or ''}\n{parsed.body}"
        except Exception:
            text = raw  # 无 frontmatter 的文件直接用全文
        docs.append({"id": f.stem, "text": text})
    return docs


def load_gold() -> list[dict]:
    data = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    queries = data["queries"]
    for item in queries:
        if item.get("category") not in VALID_CATEGORIES:
            raise ValueError(f"gold 中未知类别: {item.get('category')!r}（query={item.get('id')!r}）")
    return queries


# ─── BM25 打分与指标 ──────────────────────────────────────


def bm25_topk(
    docs: list[dict],
    query: str,
    tokenizer: Callable[[str], list[str]],
    k: int,
) -> list[tuple[str, float]]:
    """与 agents.memory.bm25_topk 相同的打分循环；仅文档文本与 tokenizer 由调用方给定。"""
    query_tokens = tokenizer(query)
    if not query_tokens:
        return []

    doc_texts = [(d, tokenizer(d["text"])) for d in docs]

    # 统计每个 token 出现在几个文档里，用于 IDF。
    df: dict[str, int] = {}
    for _, tokens in doc_texts:
        for tok in set(tokens):
            df[tok] = df.get(tok, 0) + 1

    n = len(doc_texts)
    avgdl = sum(len(tokens) for _, tokens in doc_texts) / n if n else 0.0

    k1, b = BM25_K1, BM25_B
    scored: list[tuple[float, str]] = []
    for d, tokens in doc_texts:
        dl = len(tokens)
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        score = 0.0
        for tok in query_tokens:
            freq = tf.get(tok, 0)
            if not freq:
                continue
            idf = math.log(1 + (n - df[tok] + 0.5) / (df[tok] + 0.5))
            if avgdl > 0:
                denom = freq + k1 * (1 - b + b * dl / avgdl)
            else:
                denom = freq + k1 * (1 - b)
            score += idf * (freq * (k1 + 1)) / denom
        if score > 0:
            scored.append((score, d["id"]))

    # 分数降序、id 升序，保证报告可复现。
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [(doc_id, score) for score, doc_id in scored[:k]]


def recall_at(k: int, ranked_ids: list[str], gold_ids: list[str]) -> float:
    if not gold_ids:
        return 0.0
    top = set(ranked_ids[:k])
    return sum(1 for g in gold_ids if g in top) / len(gold_ids)


def mrr_at(k: int, ranked_ids: list[str], gold_ids: list[str]) -> float:
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in gold_ids:
            return 1.0 / rank
    return 0.0


def precision_at(k: int, ranked_ids: list[str], gold_ids: list[str]) -> float:
    top = ranked_ids[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in gold_ids) / len(top)


# ─── 评测与报告 ───────────────────────────────────────────


def run_eval(modes: list[str], topk: int) -> tuple[list[dict], dict[str, list[dict]], int]:
    docs = load_corpus()
    gold = load_gold()
    tokenizers = {
        "ascii": tokenizer_ascii,
        "bigram": tokenizer_bigram,
    }
    results: dict[str, list[dict]] = {}
    for mode in modes:
        results[mode] = []
        for item in gold:
            ranked = bm25_topk(docs, item["query"], tokenizers[mode], topk)
            ranked_ids = [doc_id for doc_id, _ in ranked]
            results[mode].append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "gold": item["gold"],
                    "ranked": ranked,
                    "recall1": recall_at(1, ranked_ids, item["gold"]),
                    "recall3": recall_at(3, ranked_ids, item["gold"]),
                    "recall5": recall_at(5, ranked_ids, item["gold"]),
                    "recall15": recall_at(topk, ranked_ids, item["gold"]),
                    "mrr": mrr_at(topk, ranked_ids, item["gold"]),
                    "p3": precision_at(3, ranked_ids, item["gold"]),
                }
            )
    return gold, results, len(docs)


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _metric_rows(gold: list[dict], results: dict[str, list[dict]], topk: int, idx: list[int] | None = None):
    """返回 (label, mode -> 均值) 的指标行。idx 为 None 时统计全部 query。"""
    rows = []
    n = len(idx) if idx is not None else len(gold)
    for metric, label in [
        ("recall1", "Recall@1"),
        ("recall3", "Recall@3"),
        ("recall5", "Recall@5"),
        ("recall15", f"Recall@{topk}"),
        ("mrr", f"MRR@{topk}"),
        ("p3", "P@3"),
    ]:
        vals = {}
        for m, per_query in results.items():
            data = per_query if idx is None else [per_query[i] for i in idx]
            vals[m] = sum(r[metric] for r in data) / n if n else 0.0
        rows.append((label, vals))
    return rows


def _table(modes: list[str], rows: list[tuple[str, dict[str, float]]]) -> list[str]:
    lines = ["| 指标 | " + " | ".join(modes) + " |", "| --- | " + " | ".join("---" for _ in modes) + " |"]
    for label, vals in rows:
        lines.append(f"| {label} | " + " | ".join(_fmt(vals[m]) for m in modes) + " |")
    return lines


def build_report(gold: list[dict], results: dict[str, list[dict]], topk: int, n_docs: int) -> str:
    modes = list(results)
    lines: list[str] = []
    lines.append("# BM25 中文 tokenizer A/B 评测报告")
    lines.append("")
    lines.append(f"- 语料: {n_docs} 篇（tools/eval_data/corpus）")
    lines.append(f"- 评测集: {len(gold)} 条 query（tools/eval_data/gold.json）")
    lines.append(f"- top-k 截断: {topk}；BM25 公式与常量复用 agents.memory（k1={BM25_K1}, b={BM25_B}）")
    lines.append("- 单一变量：仅分词器不同；ascii 为改造前基线（中文全丢），bigram 为现网方案")
    lines.append("")

    lines.append("## 总体指标")
    lines.append("")
    lines.extend(_table(modes, _metric_rows(gold, results, topk)))
    lines.append("")

    categories = sorted({g["category"] for g in gold})
    lines.append("## 按类别")
    lines.append("")
    for cat in categories:
        idx = [i for i, g in enumerate(gold) if g["category"] == cat]
        n = len(idx)
        lines.append(f"### {cat}（{n} 条）")
        lines.append("")
        rows = []
        for metric, label in [("recall3", "Recall@3"), ("recall15", f"Recall@{topk}"), ("mrr", f"MRR@{topk}")]:
            vals = {}
            for m in modes:
                vals[m] = sum(results[m][i][metric] for i in idx) / n if n else 0.0
            rows.append((label, vals))
        lines.extend(_table(modes, rows))
        lines.append("")

    lines.append("## 逐 query 明细")
    lines.append("")
    for i, g in enumerate(gold):
        lines.append(f"### {g['id']} [{g['category']}] {g['query']}")
        lines.append("")
        lines.append(f"- gold: {', '.join(g['gold'])}")
        for m in modes:
            r = results[m][i]
            top3 = " ".join(f"{doc_id}({score:.3f})" for doc_id, score in r["ranked"][:3]) or "—"
            marks = "".join("✓" if doc_id in g["gold"] else " " for doc_id, _ in r["ranked"][:3])
            lines.append(
                f"- {m}: {top3} {marks}  (R@{topk}={_fmt(r['recall15'])}, MRR={_fmt(r['mrr'])})"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BM25 中文 tokenizer A/B 评测")
    parser.add_argument("--mode", default=None, help="逗号分隔的模式子集: ascii,bigram（默认全部）")
    parser.add_argument("--topk", type=int, default=15, help="BM25 截断深度（默认 15）")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help=f"报告输出路径（默认 {DEFAULT_REPORT}）")
    args = parser.parse_args(argv)

    modes = available_modes(args.mode)
    if not modes:
        raise SystemExit("没有可用模式（请指定 ascii 或 bigram）")

    gold, results, n_docs = run_eval(modes, args.topk)
    report = build_report(gold, results, args.topk, n_docs)
    print(report)
    out = Path(args.report)
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\n[report] 已写入 {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
