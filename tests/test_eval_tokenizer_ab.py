"""tools/eval_tokenizer_ab.py 的评测工具单测。

覆盖：gold 数据完整性、指标计算正确性、bigram tokenizer 与生产实现一致性、
真实语料上的端到端冒烟与确定性。
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"

# tools 无 __init__.py，按路径导入；仓库根入 sys.path 供 import agents。
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(REPO_ROOT))

try:
    import eval_tokenizer_ab as evalmod
except Exception as _exc:  # agents 依赖缺失（如 rich）时明确跳过而非误报
    evalmod = None
    _IMPORT_ERROR = _exc


def setUpModule():
    if evalmod is None:
        raise unittest.SkipTest(f"评测模块导入失败，跳过全部用例: {_IMPORT_ERROR}")


class TestDataIntegrity(unittest.TestCase):
    """种子语料与 gold 评测集的完整性校验。"""

    def test_corpus_has_enough_docs(self):
        docs = evalmod.load_corpus()
        self.assertGreaterEqual(len(docs), 30, "语料篇数不足，评测说服力不够")
        ids = [d["id"] for d in docs]
        self.assertEqual(len(ids), len(set(ids)), "语料存在重复 id")
        for d in docs:
            self.assertTrue(d["id"] and d["text"], f"语料条目为空: {d['id']!r}")

    def test_gold_schema_and_referenced_docs_exist(self):
        gold = evalmod.load_gold()
        corpus_ids = {d["id"] for d in evalmod.load_corpus()}
        self.assertGreaterEqual(len(gold), 20, "评测 query 太少")
        for item in gold:
            self.assertTrue(item["id"], "query 缺少 id")
            self.assertTrue(item["query"].strip(), f"{item['id']} 缺少 query 文本")
            self.assertIn(item["category"], evalmod.VALID_CATEGORIES, f"{item['id']} 类别非法")
            self.assertTrue(item["gold"], f"{item['id']} 的 gold 为空")
            for doc_id in item["gold"]:
                self.assertIn(doc_id, corpus_ids, f"{item['id']} 标注的 {doc_id} 不在语料中")

    def test_categories_cover_hard_cases(self):
        cats = {g["category"] for g in evalmod.load_gold()}
        for need in ("synonym", "short", "stopword", "jargon", "mixed"):
            self.assertIn(need, cats, f"评测集缺少 {need} 类 hard case")


class TestMetrics(unittest.TestCase):
    """指标计算的手工验证。"""

    def test_recall_at(self):
        self.assertEqual(evalmod.recall_at(3, ["a", "b", "c"], ["b"]), 1.0)
        self.assertEqual(evalmod.recall_at(1, ["a", "b", "c"], ["b"]), 0.0)
        self.assertEqual(evalmod.recall_at(3, ["a", "b", "c"], ["a", "b"]), 1.0)
        self.assertEqual(evalmod.recall_at(3, ["a", "b", "c"], ["a", "d"]), 0.5)
        self.assertEqual(evalmod.recall_at(5, ["a"], ["b"]), 0.0)

    def test_mrr_at(self):
        self.assertEqual(evalmod.mrr_at(15, ["a", "b", "c"], ["b"]), 0.5)
        self.assertEqual(evalmod.mrr_at(15, ["a", "b"], ["c"]), 0.0)
        self.assertEqual(evalmod.mrr_at(2, ["a", "b", "c"], ["c"]), 0.0, "超出 k 的命中不计")

    def test_precision_at(self):
        self.assertEqual(evalmod.precision_at(3, ["a", "b", "c"], ["a", "d"]), 1 / 3)
        self.assertEqual(evalmod.precision_at(3, ["a", "b", "c"], []), 0.0)
        self.assertEqual(evalmod.precision_at(3, [], ["a"]), 0.0)


class TestTokenizers(unittest.TestCase):
    """分词器语义与生产实现的一致性。"""

    def test_ascii_baseline_drops_chinese(self):
        self.assertEqual(evalmod.tokenizer_ascii("帮我检索数据库"), [])
        self.assertEqual(evalmod.tokenizer_ascii("BM25 参数"), ["bm25"])

    def test_bigram_matches_production_tokenizer(self):
        """评测用的 bigram 与共享 tokenizer 及 memory 包装器三方一致，防实现漂移。"""
        from agents.memory import _tokenize
        from agents.tokenizer import tokenize

        samples = [
            "帮我压缩一下上下文",
            "数据库检索方案",
            "git commit 消息规范",
            "bm25 参数调优",
            "把之前聊过的内容捞出来",
        ]
        for text in samples:
            self.assertEqual(evalmod.tokenizer_bigram(text), list(tokenize(text)), text)
            self.assertEqual(list(_tokenize(text)), list(tokenize(text)), text)

    def test_bigram_produces_cjk_tokens(self):
        tokens = evalmod.tokenizer_bigram("帮我检索数据库")
        self.assertIn("检索", tokens)
        self.assertIn("数据", tokens)
        self.assertNotIn("帮我", tokens, "停用词应被过滤")

    def test_unknown_mode_raises(self):
        with self.assertRaises(SystemExit):
            evalmod.available_modes("foo")


class TestEndToEndSmoke(unittest.TestCase):
    """真实语料上的冒烟：指标范围、基线失效、整体对比与确定性。"""

    @classmethod
    def setUpClass(cls):
        cls.gold, cls.results, cls.n_docs = evalmod.run_eval(["ascii", "bigram"], topk=15)

    def test_corpus_and_gold_loaded(self):
        self.assertGreaterEqual(self.n_docs, 30)
        self.assertEqual(len(self.results["ascii"]), len(self.gold))

    def test_metrics_within_range(self):
        for mode, per_query in self.results.items():
            for r in per_query:
                for metric in ("recall1", "recall3", "recall5", "recall15", "mrr", "p3"):
                    self.assertGreaterEqual(r[metric], 0.0, f"{mode} {r['id']} {metric}")
                    self.assertLessEqual(r[metric], 1.0, f"{mode} {r['id']} {metric}")

    def test_ascii_baseline_fails_pure_chinese_query(self):
        """纯中文 query 在 ascii 基线下 token 为空，预筛必然漏召回。"""
        idx = next(i for i, g in enumerate(self.gold) if g["id"] == "q08")
        self.assertEqual(self.results["ascii"][idx]["recall15"], 0.0)

    def test_bigram_beats_ascii_overall(self):
        ascii_avg = sum(r["recall15"] for r in self.results["ascii"]) / len(self.gold)
        bigram_avg = sum(r["recall15"] for r in self.results["bigram"]) / len(self.gold)
        self.assertGreater(bigram_avg, ascii_avg, "中文语料上 bigram 应显著优于纯 ASCII 基线")

    def test_run_is_deterministic(self):
        _, results2, _ = evalmod.run_eval(["ascii", "bigram"], topk=15)
        for mode in ("ascii", "bigram"):
            for r1, r2 in zip(self.results[mode], results2[mode]):
                self.assertEqual([d for d, _ in r1["ranked"]], [d for d, _ in r2["ranked"]], mode)


if __name__ == "__main__":
    unittest.main()
