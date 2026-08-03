# tests/test_verification.py
# 针对 agents/verification.py 三层验证架构与 agent.py 集成点的回归测试：
#   1. 规则解析与校验（id/level/type）
#   2. 各检查器：file_exists / glob_exists / dir_nonempty / file_contains / command_success
#   3. run_verification 报告结构与失败汇总
#   4. 自动收集 L1 规则（本轮写过的产物）
#   5. agent 集成：_verify_before_done 检查点行为 / 工具分发 / 会话存档
#   6. run_verification 权限：default allow / plan deny
# 仅使用标准库 unittest，运行方式：python -m unittest discover -s tests

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agents import agent as agent_mod
from agents import tools
from agents.agent import Agent
from agents.verification import (
    VerificationRule,
    check_rule,
    collect_written_file_rules,
    format_verification_feedback,
    load_verification_rules,
    run_verification,
)


def _make_agent() -> Agent:
    return Agent(
        model="deepseek-chat",
        api_key="test-key",
        api_base="http://localhost:9/v1",
        custom_system_prompt="test prompt",
    )


class TestRuleParsing(unittest.TestCase):
    """规则解析：合法规则保留，非法规则（缺 id / 越界 level / 未知 type）过滤。"""

    def _parse(self, raw):
        from agents.verification import _parse_rule
        return _parse_rule(raw)

    def test_valid_rule_parsed(self):
        r = self._parse({"id": "r1", "level": 2, "type": "file_contains",
                         "target": "a.py", "pattern": "def main", "timeout": 30})
        self.assertIsNotNone(r)
        self.assertEqual(r.id, "r1")
        self.assertEqual(r.level, 2)

    def test_missing_id_rejected(self):
        self.assertIsNone(self._parse({"level": 1, "type": "file_exists"}))

    def test_out_of_range_level_rejected(self):
        self.assertIsNone(self._parse({"id": "r", "level": 9, "type": "file_exists"}))

    def test_unknown_type_rejected(self):
        self.assertIsNone(self._parse({"id": "r", "level": 1, "type": "no_such_type"}))

    def test_load_skips_invalid_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "verification.json"
            cfg.write_text(json.dumps({
                "rules": [
                    {"id": "ok", "level": 1, "type": "file_exists", "target": "x"},
                    {"id": "bad", "level": 9, "type": "file_exists", "target": "x"},
                ]
            }), encoding="utf-8")
            rules = load_verification_rules(cfg)
        self.assertEqual([r.id for r in rules], ["ok"])

    def test_load_missing_config_returns_empty(self):
        self.assertEqual(load_verification_rules(Path("no_such_file.json")), [])


class TestCheckers(unittest.TestCase):
    """各检查器正/负样例。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._dir = Path(self._tmp.name)
        (self._dir / "out.txt").write_text("hello world", encoding="utf-8")
        (self._dir / "sub").mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _rule(self, **kw):
        return VerificationRule(kw)

    def test_file_exists_pass_and_fail(self):
        self.assertEqual(check_rule(self._rule(id="r", level=1, type="file_exists", target="out.txt"), self._dir)["status"], "pass")
        self.assertEqual(check_rule(self._rule(id="r", level=1, type="file_exists", target="nope.txt"), self._dir)["status"], "fail")

    def test_file_contains_pass_fail_and_missing(self):
        r = self._rule(id="r", level=2, type="file_contains", target="out.txt", pattern="hello")
        self.assertEqual(check_rule(r, self._dir)["status"], "pass")
        r2 = self._rule(id="r", level=2, type="file_contains", target="out.txt", pattern="bye")
        self.assertEqual(check_rule(r2, self._dir)["status"], "fail")
        r3 = self._rule(id="r", level=2, type="file_contains", target="nope.txt", pattern="x")
        self.assertEqual(check_rule(r3, self._dir)["status"], "fail")

    def test_file_contains_regex_prefix(self):
        r = self._rule(id="r", level=2, type="file_contains", target="out.txt", pattern="re:^hello")
        self.assertEqual(check_rule(r, self._dir)["status"], "pass")
        r2 = self._rule(id="r", level=2, type="file_contains", target="out.txt", pattern="re:^bye")
        self.assertEqual(check_rule(r2, self._dir)["status"], "fail")

    def test_glob_exists(self):
        self.assertEqual(check_rule(self._rule(id="r", level=1, type="glob_exists", target="*.txt"), self._dir)["status"], "pass")
        self.assertEqual(check_rule(self._rule(id="r", level=1, type="glob_exists", target="*.md"), self._dir)["status"], "fail")

    def test_dir_nonempty(self):
        self.assertEqual(check_rule(self._rule(id="r", level=1, type="dir_nonempty", target="sub"), self._dir)["status"], "fail")
        (self._dir / "sub" / "f").write_text("x", encoding="utf-8")
        self.assertEqual(check_rule(self._rule(id="r", level=1, type="dir_nonempty", target="sub"), self._dir)["status"], "pass")

    def test_command_success(self):
        ok = f'"{sys.executable}" -c "print()"'
        self.assertEqual(check_rule(self._rule(id="r", level=3, type="command_success", command=ok), self._dir)["status"], "pass")
        bad = f'"{sys.executable}" -c "import sys; sys.exit(3)"'
        res = check_rule(self._rule(id="r", level=3, type="command_success", command=bad), self._dir)
        self.assertEqual(res["status"], "fail")
        self.assertIn("exit 3", res["detail"])

    def test_command_timeout(self):
        slow = f'"{sys.executable}" -c "import time; time.sleep(3)"'
        res = check_rule(self._rule(id="r", level=3, type="command_success", command=slow, timeout=1), self._dir)
        self.assertEqual(res["status"], "fail")
        self.assertIn("timeout", res["detail"])

    def test_llm_judge_skipped_by_default(self):
        r = self._rule(id="r", level=3, type="llm_judge")
        self.assertEqual(check_rule(r, self._dir)["status"], "skip")


class TestRunVerification(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_all_pass(self):
        (self._dir / "a.txt").write_text("done", encoding="utf-8")
        rules = [
            VerificationRule({"id": "r1", "level": 1, "type": "file_exists", "target": "a.txt"}),
            VerificationRule({"id": "r2", "level": 2, "type": "file_contains", "target": "a.txt", "pattern": "done"}),
        ]
        report = run_verification(rules, cwd=self._dir)
        self.assertTrue(report["passed"])
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["failures"], [])

    def test_failures_collected(self):
        rules = [
            VerificationRule({"id": "r1", "level": 1, "type": "file_exists", "target": "missing.txt"}),
            VerificationRule({"id": "r2", "level": 2, "type": "file_contains", "target": "missing.txt", "pattern": "x"}),
        ]
        report = run_verification(rules, cwd=self._dir)
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["failures"]), 2)
        self.assertEqual([f["id"] for f in report["failures"]], ["r1", "r2"])

    def test_invalid_rule_skipped_not_failed(self):
        rules = [VerificationRule({"id": "r", "level": 1, "type": "no_such_type"})]
        report = run_verification(rules, cwd=self._dir)
        self.assertTrue(report["passed"])
        self.assertEqual(report["results"][0]["status"], "skip")


class TestAutoCollect(unittest.TestCase):
    def test_written_files_become_l1_rules(self):
        rules = collect_written_file_rules({"a.py", "sub/b.py"}, root=Path.cwd())
        self.assertEqual(len(rules), 2)
        for r in rules:
            self.assertEqual(r.level, 1)
            self.assertEqual(r.type, "file_exists")
            self.assertTrue(r.id.startswith("auto-exists:"))


class TestFeedbackFormat(unittest.TestCase):
    def test_feedback_contains_failure_details(self):
        report = {
            "passed": False,
            "failures": [
                {"id": "r1", "level": 2, "type": "file_contains", "severity": "error",
                 "description": "必须包含 main", "detail": "pattern not found in a.py"},
            ],
        }
        text = format_verification_feedback(report, attempt=1, max_attempts=3)
        self.assertIn("Verification Report", text)
        self.assertIn("[L2] r1", text)
        self.assertIn("pattern not found", text)
        self.assertIn("1/3", text)


class TestAgentVerificationIntegration(unittest.TestCase):
    """agent.py 集成：检查点行为 / 工具分发 / 会话存档。"""

    def _chdir_tmp(self, tmp: str):
        self._old_cwd = os.getcwd()
        os.chdir(tmp)

    def _restore_cwd(self):
        os.chdir(self._old_cwd)

    def test_write_file_records_written_path(self):
        a = _make_agent()
        with patch.object(agent_mod, "execute_tool", new=AsyncMock(return_value="ok")):
            asyncio.run(a._execute_tool_call("write_file", {"file_path": "a.txt", "content": "x"}))
        self.assertIn("a.txt", a._written_files)

    def test_edit_file_records_written_path(self):
        a = _make_agent()
        with patch.object(agent_mod, "execute_tool", new=AsyncMock(return_value="ok")):
            asyncio.run(a._execute_tool_call("edit_file", {"file_path": "b.txt", "old_string": "a", "new_string": "b"}))
        self.assertIn("b.txt", a._written_files)

    def test_verify_before_done_passes_without_rules(self):
        a = _make_agent()
        with patch("agents.agent.load_verification_rules", return_value=[]):
            self.assertTrue(asyncio.run(a._verify_before_done()))

    def test_verify_before_done_failure_injects_feedback(self):
        rule = VerificationRule({"id": "r1", "level": 1, "type": "file_exists", "target": "missing.txt"})
        with tempfile.TemporaryDirectory() as tmp:
            self._chdir_tmp(tmp)
            try:
                a = _make_agent()
                with patch("agents.agent.load_verification_rules", return_value=[rule]):
                    result = asyncio.run(a._verify_before_done())
                self.assertFalse(result)
                injected = [m["content"] for m in a._openai_messages if isinstance(m.get("content"), str)]
                self.assertTrue(any("[Verification Report]" in c for c in injected))
                self.assertEqual(len(a._verification_log), 1)
                self.assertFalse(a._verification_log[0]["passed"])
            finally:
                self._restore_cwd()

    def test_verify_before_done_releases_after_max_attempts(self):
        rule = VerificationRule({"id": "r1", "level": 1, "type": "file_exists", "target": "missing.txt"})
        with tempfile.TemporaryDirectory() as tmp:
            self._chdir_tmp(tmp)
            try:
                a = _make_agent()
                with patch("agents.agent.load_verification_rules", return_value=[rule]):
                    results = [asyncio.run(a._verify_before_done()) for _ in range(4)]
                # 前两次失败返回 False（第 3 次尝试也失败但已达上限放行）
                self.assertEqual(results, [False, False, True, True])
                # 每次验证都留下存档记录（含第 3 次失败与第 4 次重复调用）
                self.assertEqual(len(a._verification_log), 4)
            finally:
                self._restore_cwd()

    def test_verify_before_done_skips_sub_agent_and_plan_mode(self):
        a = _make_agent()
        a.is_sub_agent = True
        self.assertTrue(asyncio.run(a._verify_before_done()))
        b = _make_agent()
        b.permission_mode = "plan"
        self.assertTrue(asyncio.run(b._verify_before_done()))

    def test_auto_save_includes_verification_log(self):
        a = _make_agent()
        a._verification_log = [{"attempt": 1, "passed": True, "total": 2, "failures": []}]
        with patch("agents.agent.save_session") as mock_save:
            a._auto_save()
        data = mock_save.call_args[0][1]
        self.assertEqual(data["verification"], [{"attempt": 1, "passed": True, "total": 2, "failures": []}])

    def test_run_verification_tool_returns_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._chdir_tmp(tmp)
            try:
                Path("out.txt").write_text("done", encoding="utf-8")
                a = _make_agent()
                a._written_files.add("out.txt")
                with patch("agents.agent.load_verification_rules", return_value=[]):
                    res = asyncio.run(a._run_verification_tool({}))
                data = json.loads(res)
                self.assertTrue(data["passed"])
                self.assertEqual(len(data["results"]), 1)  # 自动收集的 L1 规则
            finally:
                self._restore_cwd()

    def test_run_verification_tool_filters_by_rule_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._chdir_tmp(tmp)
            try:
                Path("out.txt").write_text("done", encoding="utf-8")
                a = _make_agent()
                a._written_files.add("out.txt")
                with patch("agents.agent.load_verification_rules", return_value=[]):
                    res = asyncio.run(a._run_verification_tool({"rule_ids": ["no-such-rule"]}))
                self.assertIn("No verification rules configured", res)
            finally:
                self._restore_cwd()


class TestRunVerificationPermission(unittest.TestCase):
    """run_verification 权限：default 放行（只读验证），plan 模式拒绝（无产物可验）。"""

    def setUp(self):
        self._saved_rules = tools._cached_rules
        tools._cached_rules = {"allow": [], "deny": []}

    def tearDown(self):
        tools._cached_rules = self._saved_rules

    def test_default_mode_allows(self):
        result = tools.check_permission("run_verification", {}, mode="default")
        self.assertEqual(result["action"], "allow")

    def test_accept_edits_mode_allows(self):
        result = tools.check_permission("run_verification", {}, mode="acceptEdits")
        self.assertEqual(result["action"], "allow")

    def test_plan_mode_denies(self):
        result = tools.check_permission("run_verification", {}, mode="plan")
        self.assertEqual(result["action"], "deny")


if __name__ == "__main__":
    unittest.main()
