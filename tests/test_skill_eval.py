# tests/test_skill_eval.py
# 针对 agents/online_skill_eval.py 分层评测体系的最小聚焦测试：
#   1. 时间切分：较早样本归 mutate_dev，较晚样本归 promotion_test；样本 <2 全归 mutate_dev
#   2. programmatic 规则：从 skill 文本编译规则并确定性评估（零 token 基线）
#   3. judge 容错：非 JSON / 未配置 side_query 时降级不崩溃
#   4. 预算控制：LLM judge 调用超限标记 skipped（VeRO budget-controlled evaluation）
#   5. 配对评测：有 skill / 无 skill 分组 pass rate 差（SkillsBench 方法论）
#   6. 晋升门槛：分数不足 / 硬规则失败不晋升；首次 healthy 晋升
# 全部为纯函数/异步函数测试，不依赖网络、不写盘。
# 运行方式：python -m unittest discover -s tests

import asyncio
import unittest

from agents import online_skill_eval as eval_mod
from agents import skill_evolution as se_mod
from agents.online_skill_eval import (
    _assign_replay_splits,
    _compile_eval_rules,
    _evaluate_rule,
    _evaluate_rule_async,
    _paired_metric,
    _promotion_decision,
    _skill_status,
)


def _sample(sample_id, time_str, has_skill_context=False, latest_assistant=""):
    return {
        "sample_id": sample_id,
        "time": time_str,
        "split": "mutate_dev",
        "has_skill_context": has_skill_context,
        "latest_assistant": latest_assistant,
    }


class TestTimeSplit(unittest.TestCase):
    def test_time_based_split_ratio(self):
        samples = [
            _sample("a", "2026-01-01T00:00:00Z"),
            _sample("b", "2026-02-01T00:00:00Z"),
            _sample("c", "2026-03-01T00:00:00Z"),
            _sample("d", "2026-04-01T00:00:00Z"),
        ]
        split = _assign_replay_splits(samples)
        dev = [s for s in split if s["split"] == "mutate_dev"]
        test = [s for s in split if s["split"] == "promotion_test"]
        self.assertEqual(len(dev), 3, "75% 较早样本应归 mutate_dev")
        self.assertEqual(len(test), 1, "25% 较晚样本应归 promotion_test")
        # 时间最晚的样本必须是 promotion_test（回答“演化能否泛化到未来”）
        self.assertEqual(test[0]["sample_id"], "d")

    def test_single_sample_all_dev(self):
        split = _assign_replay_splits([_sample("a", "2026-01-01T00:00:00Z")])
        self.assertEqual(len(split), 1)
        self.assertEqual(split[0]["split"], "mutate_dev")

    def test_empty_pool(self):
        self.assertEqual(_assign_replay_splits([]), [])


class TestProgrammaticRules(unittest.TestCase):
    def test_compile_rules_from_skill_text(self):
        skill = {
            "name": "research-report",
            "description": "write research report with sources",
            "when_to_use": "when user asks for research",
            "instructions": "cite sources. keep within 3 paragraphs. output JSON.",
            "tags": [],
        }
        rules = _compile_eval_rules(skill, include_llm_rules=False)
        rule_ids = {r["rule_id"] for r in rules}
        self.assertIn("must_cite_sources", rule_ids)
        self.assertIn("paragraph_limit", rule_ids)
        self.assertIn("json_parseable", rule_ids)
        self.assertTrue(all(r["kind"] == "programmatic" for r in rules), "无 LLM 规则时全部为 programmatic")

    def test_evaluate_nonempty_rule(self):
        rule = {"rule_id": "response_nonempty", "hard": True, "kind": "programmatic", "params": {"mode": "nonempty"}}
        self.assertTrue(_evaluate_rule(rule, "answer")["passed"])
        self.assertFalse(_evaluate_rule(rule, "  ")["passed"])

    def test_evaluate_cite_sources_rule(self):
        rule = {"rule_id": "must_cite_sources", "hard": True, "kind": "programmatic", "params": {"mode": "mentions_sources"}}
        self.assertTrue(_evaluate_rule(rule, "see https://example.com/doc")["passed"])
        self.assertFalse(_evaluate_rule(rule, "no links here")["passed"])


class TestJudgeFallback(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_judge_non_json_response_no_crash(self):
        async def _case():
            rule = {
                "rule_id": "skill_instruction_alignment",
                "kind": "llm_binary",
                "hard": False,
                "params": {"mode": "requirement", "requirement_text": "keep it short"},
            }

            async def bad_side_query(system, user):
                return "this is not json at all"

            outcome = await _evaluate_rule_async(
                rule, "some response", sample={"latest_user": "q"}, skill_name="s", side_query=bad_side_query
            )
            return outcome

        outcome = self._run(_case())
        self.assertFalse(outcome["passed"], "非 JSON judge 响应按未通过处理（保守方向）")
        self.assertIn("non-JSON", outcome["details"].get("reason", ""))

    def test_judge_not_configured_skips(self):
        async def _case():
            rule = {
                "rule_id": "skill_instruction_alignment",
                "kind": "llm_binary",
                "hard": False,
                "params": {"mode": "requirement", "requirement_text": "keep it short"},
            }
            return await _evaluate_rule_async(rule, "resp", sample={"latest_user": "q"}, skill_name="s", side_query=None)

        outcome = self._run(_case())
        self.assertTrue(outcome.get("skipped"), "side_query 未配置时规则应标记 skipped（降级纯 programmatic）")

    def test_judge_valid_json_passes(self):
        async def _case():
            rule = {
                "rule_id": "skill_instruction_alignment",
                "kind": "llm_binary",
                "hard": False,
                "params": {"mode": "requirement", "requirement_text": "keep it short"},
            }

            async def good_side_query(system, user):
                return '{"reason": "follows", "pass": true}'

            return await _evaluate_rule_async(rule, "resp", sample={"latest_user": "q"}, skill_name="s", side_query=good_side_query)

        outcome = self._run(_case())
        self.assertTrue(outcome["passed"])
        self.assertEqual(outcome["details"].get("reason"), "follows")


class TestBudgetControl(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_budget_exhausted_marks_skipped(self):
        async def _case():
            rule = {
                "rule_id": "skill_instruction_alignment",
                "kind": "llm_binary",
                "hard": False,
                "params": {"mode": "requirement", "requirement_text": "keep it short"},
            }

            async def exhausted_side_query(system, user):
                raise eval_mod._BudgetExhausted()

            return await _evaluate_rule_async(rule, "resp", sample={"latest_user": "q"}, skill_name="s", side_query=exhausted_side_query)

        outcome = self._run(_case())
        self.assertTrue(outcome.get("skipped"))
        self.assertEqual(outcome["details"].get("reason"), "judge_budget_exhausted")


class TestPairedMetric(unittest.TestCase):
    def test_paired_delta_computation(self):
        pool = [
            _sample("a", "2026-01-01T00:00:00Z", has_skill_context=True),
            _sample("b", "2026-01-02T00:00:00Z", has_skill_context=True),
            _sample("c", "2026-01-03T00:00:00Z", has_skill_context=False),
            _sample("d", "2026-01-04T00:00:00Z", has_skill_context=False),
        ]
        summary = {
            "outcomes": [
                {"sample_id": "a", "passed": True},
                {"sample_id": "b", "passed": True},
                {"sample_id": "c", "passed": False},
                {"sample_id": "d", "passed": False},
            ]
        }
        paired = _paired_metric(pool, summary)
        self.assertEqual(paired["with_skill_pass_rate"], 1.0)
        self.assertEqual(paired["without_skill_pass_rate"], 0.0)
        self.assertEqual(paired["delta"], 1.0)
        self.assertEqual(paired["with_skill_samples"], 2)
        self.assertEqual(paired["without_skill_samples"], 2)

    def test_paired_no_without_group(self):
        pool = [_sample("a", "2026-01-01T00:00:00Z", has_skill_context=True)]
        summary = {"outcomes": [{"sample_id": "a", "passed": True}]}
        paired = _paired_metric(pool, summary)
        self.assertEqual(paired["without_skill_pass_rate"], 0.0)
        self.assertEqual(paired["without_skill_samples"], 0)


class TestPromotionGate(unittest.TestCase):
    def _candidate(self, avg, hard):
        return {"average_score": avg, "hard_failures": hard}

    def test_first_healthy_promotes(self):
        result = _promotion_decision(
            status="healthy",
            candidate=self._candidate(1.5, 0),
            champion={},
        )
        self.assertTrue(result["promoted"], "首次 healthy 候选应直接成为 champion")

    def test_score_below_delta_rejected(self):
        champion = {"summary": {"average_score": 1.5, "hard_failures": 0}}
        result = _promotion_decision(
            status="healthy",
            candidate=self._candidate(1.51, 0),  # 1.51 < 1.5 + 0.05
            champion=champion,
        )
        self.assertFalse(result["promoted"], "分数差小于 min_score_delta 时不应晋升")

    def test_score_above_delta_promotes(self):
        champion = {"summary": {"average_score": 1.5, "hard_failures": 0}}
        result = _promotion_decision(
            status="healthy",
            candidate=self._candidate(1.6, 0),  # 1.6 >= 1.5 + 0.05
            champion=champion,
        )
        self.assertTrue(result["promoted"])

    def test_more_hard_failures_rejected(self):
        champion = {"summary": {"average_score": 1.5, "hard_failures": 0}}
        result = _promotion_decision(
            status="healthy",
            candidate=self._candidate(1.6, 2),  # 分数高但硬规则失败增加
            champion=champion,
        )
        self.assertFalse(result["promoted"], "硬规则失败数增加时不应晋升")

    def test_watch_status_rejected(self):
        result = _promotion_decision(status="watch", candidate=self._candidate(2.0, 0), champion={})
        self.assertFalse(result["promoted"])


class TestSkillStatusPairedGate(unittest.TestCase):
    def _status(self, paired):
        return _skill_status(
            replay_count=5,
            promotion_test_count=2,
            retrieved=10,
            relevant=6,
            used=5,
            pruned=False,
            rule_summary={"pass_rate": 0.9, "promotion_test_hard_failures": 0, "hard_failures": 0},
            paired=paired,
            min_replay_samples=2,
            min_promotion_tests=1,
            min_retrieved=5,
            min_used_rate=0.2,
            min_relevance_rate=0.35,
            min_rule_pass_rate=0.8,
        )

    def test_negative_paired_delta_watch(self):
        status, reasons = self._status(
            {"with_skill_pass_rate": 0.5, "without_skill_pass_rate": 0.9, "delta": -0.4, "with_skill_samples": 3, "without_skill_samples": 2}
        )
        self.assertEqual(status, "watch")
        self.assertTrue(any("no skill gain" in r for r in reasons), "配对无增益应阻止晋升")

    def test_positive_paired_delta_healthy(self):
        status, reasons = self._status(
            {"with_skill_pass_rate": 0.9, "without_skill_pass_rate": 0.4, "delta": 0.5, "with_skill_samples": 3, "without_skill_samples": 2}
        )
        self.assertEqual(status, "healthy")
        self.assertFalse(any("no skill gain" in r for r in reasons))


class TestSkillTrace(unittest.TestCase):
    def test_record_trace_merge_by_trace_id(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            orig_dir = se_mod.get_evolution_dir
            se_mod.get_evolution_dir = lambda: tmp_root
            try:
                se_mod.record_skill_trace(
                    skill_name="code-review",
                    trace_id="trace-abc",
                    latest_user="review this",
                    latest_assistant="first answer",
                )
                se_mod.record_skill_trace(
                    skill_name="code-review",
                    trace_id="trace-abc",
                    latest_user="review this",
                    latest_assistant="first answer",
                    usage_judgment={"judgments": [{"name": "code-review", "used": True}]},
                    evolution_action="merge",
                )
                traces = se_mod.load_skill_traces("code-review")
                self.assertEqual(len(traces), 1, "同一 trace_id 多次回写应合并为一条")
                self.assertEqual(traces[0]["trace_id"], "trace-abc")
                self.assertEqual(traces[0]["usage_judgment"]["judgments"][0]["used"], True)
                self.assertEqual(traces[0]["evolution_action"], "merge")
            finally:
                se_mod.get_evolution_dir = orig_dir

    def test_record_trace_empty_trace_id_skipped(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            orig_dir = se_mod.get_evolution_dir
            se_mod.get_evolution_dir = lambda: tmp_root
            try:
                se_mod.record_skill_trace(skill_name="code-review", trace_id="")
                self.assertEqual(se_mod.load_skill_traces("code-review"), [])
            finally:
                se_mod.get_evolution_dir = orig_dir


if __name__ == "__main__":
    unittest.main()
