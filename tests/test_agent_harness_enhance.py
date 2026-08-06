# tests/test_agent_harness_enhance.py
# 安全加固任务（Task 3/4/8/11/12/13/14）的回归测试：
#   1. _is_retryable 状态码扩充（429/500/502/503/504/529）
#   2. get_max_verification_attempts 环境变量读取（OTTER_VERIFY_MAX_ATTEMPTS）
#   3. format_verification_feedback 已通过规则摘要（Passed rules: ...）
#   4. _check_timeout wall-clock 超时
#   5. 子代理权限收窄（acceptEdits）与 confirm_fn=None 时拒绝危险操作
#   6. _filter_l1_rules 纯函数（子代理只跑 L1 验证）
#   7. _cleanup_stale_tool_results 临时文件清理（旧删新留 + 目录缺失容错）
#   8. _read_file_state 会话持久化（save/restore）
#   9. 中途 L1 检查点行为（主代理生效 / 子代理跳过）
# 仅使用标准库 unittest，运行方式：python -m unittest discover -s tests

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_mod
from agents import tools
from agents.agent import Agent
from agents.verification import (
    VerificationRule,
    format_verification_feedback,
    get_max_verification_attempts,
)


def _make_agent(**overrides) -> Agent:
    """构造不触发网络/文件副作用的 Agent 实例（与既有测试风格一致）。"""
    kwargs = dict(
        model="deepseek-chat",
        api_key="test-key",
        api_base="http://localhost:9/v1",
        custom_system_prompt="test prompt",
    )
    kwargs.update(overrides)
    return Agent(**kwargs)


class FakeError(Exception):
    def __init__(self, status_code=None, message=""):
        self.status_code = status_code
        super().__init__(message)


class TestIsRetryable(unittest.TestCase):
    """Task 12: 退避重试状态码扩充。"""

    def test_retryable_status_codes(self):
        for code in (429, 500, 502, 503, 504, 529):
            self.assertTrue(agent_mod._is_retryable(FakeError(status_code=code)), f"code {code}")

    def test_non_retryable_status_code(self):
        self.assertFalse(agent_mod._is_retryable(FakeError(status_code=400)))

    def test_retryable_by_message_keyword(self):
        self.assertTrue(agent_mod._is_retryable(FakeError(status_code=200, message="overloaded")))
        self.assertTrue(agent_mod._is_retryable(FakeError(status_code=200, message="ECONNRESET")))
        self.assertTrue(agent_mod._is_retryable(FakeError(status_code=200, message="ETIMEDOUT")))
        self.assertFalse(agent_mod._is_retryable(FakeError(status_code=200, message="bad request")))


class TestMaxVerificationAttempts(unittest.TestCase):
    """Task 13: 验证轮数改为环境变量可配。"""

    def _restore_env(self, old):
        if old is None:
            os.environ.pop("OTTER_VERIFY_MAX_ATTEMPTS", None)
        else:
            os.environ["OTTER_VERIFY_MAX_ATTEMPTS"] = old

    def test_env_override(self):
        old = os.environ.get("OTTER_VERIFY_MAX_ATTEMPTS")
        try:
            os.environ["OTTER_VERIFY_MAX_ATTEMPTS"] = "5"
            self.assertEqual(get_max_verification_attempts(), 5)
        finally:
            self._restore_env(old)

    def test_default_when_unset(self):
        old = os.environ.get("OTTER_VERIFY_MAX_ATTEMPTS")
        try:
            os.environ.pop("OTTER_VERIFY_MAX_ATTEMPTS", None)
            self.assertEqual(get_max_verification_attempts(), 3)
        finally:
            self._restore_env(old)

    def test_invalid_value_falls_back_to_default(self):
        old = os.environ.get("OTTER_VERIFY_MAX_ATTEMPTS")
        try:
            os.environ["OTTER_VERIFY_MAX_ATTEMPTS"] = "abc"
            self.assertEqual(get_max_verification_attempts(), 3)
            os.environ["OTTER_VERIFY_MAX_ATTEMPTS"] = "0"
            self.assertEqual(get_max_verification_attempts(), 3)
            os.environ["OTTER_VERIFY_MAX_ATTEMPTS"] = "-2"
            self.assertEqual(get_max_verification_attempts(), 3)
        finally:
            self._restore_env(old)


class TestFormatVerificationFeedback(unittest.TestCase):
    """Task 13: 反馈报告附带已通过规则摘要。"""

    def test_passed_rules_summary_included(self):
        report = {
            "passed": False,
            "results": [
                {"id": "r-pass-1", "status": "pass"},
                {"id": "r-pass-2", "status": "pass"},
                {"id": "r-fail", "status": "fail"},
                {"id": "r-skip", "status": "skip"},
            ],
            "failures": [
                {"id": "r-fail", "level": 1, "type": "file_exists", "severity": "error",
                 "description": "d", "detail": "not found"},
            ],
        }
        text = format_verification_feedback(report, attempt=1, max_attempts=3)
        self.assertIn("Passed rules", text)
        self.assertIn("r-pass-1", text)
        self.assertIn("r-pass-2", text)
        # 失败规则仍在列表中
        self.assertIn("[L1] r-fail", text)
        self.assertIn("not found", text)
        # 未通过/跳过的不出现在 Passed rules 行中
        passed_line = [ln for ln in text.splitlines() if ln.startswith("Passed rules:")][0]
        self.assertNotIn("r-fail", passed_line)
        self.assertNotIn("r-skip", passed_line)

    def test_no_passed_rules_line_when_none_pass(self):
        report = {
            "passed": False,
            "results": [{"id": "r1", "status": "fail"}],
            "failures": [{"id": "r1", "level": 1, "type": "file_exists", "severity": "error",
                          "description": "d", "detail": "missing"}],
        }
        text = format_verification_feedback(report, attempt=1, max_attempts=3)
        self.assertNotIn("Passed rules", text)


class TestCheckTimeout(unittest.TestCase):
    """Task 14: wall-clock 超时。"""

    def test_no_duration_never_times_out(self):
        a = _make_agent(max_duration_s=None)
        self.assertFalse(a._check_timeout())

    def test_short_duration_times_out(self):
        # Agent 构造本身耗时可能超过 0.001s，因此只断言 sleep 后必然超时。
        a = _make_agent(max_duration_s=0.001)
        time.sleep(0.02)
        self.assertTrue(a._check_timeout())

    def test_long_duration_does_not_time_out(self):
        a = _make_agent(max_duration_s=600)
        self.assertFalse(a._check_timeout())


class TestSubAgentPermissionNarrowing(unittest.TestCase):
    """Task 3: 子代理权限收窄为 acceptEdits（plan 派生保持 plan）。"""

    def test_default_mode_yields_accept_edits(self):
        a = _make_agent(permission_mode="default")
        self.assertEqual(a._sub_agent_permission_mode(), "acceptEdits")

    def test_bypass_permissions_yields_accept_edits(self):
        a = _make_agent(permission_mode="bypassPermissions")
        self.assertEqual(a._sub_agent_permission_mode(), "acceptEdits")

    def test_accept_edits_yields_accept_edits(self):
        a = _make_agent(permission_mode="acceptEdits")
        self.assertEqual(a._sub_agent_permission_mode(), "acceptEdits")

    def test_plan_mode_yields_plan(self):
        a = _make_agent(permission_mode="plan")
        self.assertEqual(a._sub_agent_permission_mode(), "plan")

    def test_confirm_dangerous_rejects_without_confirm_fn(self):
        # 子代理 confirm_fn 为 None 时必须拒绝（不能阻塞等待输入）。
        a = _make_agent(is_sub_agent=True, confirm_fn=None)
        self.assertFalse(asyncio.run(a._confirm_dangerous("git push")))

    def test_confirm_dangerous_uses_confirm_fn_when_provided(self):
        async def _fn(cmd):
            return cmd == "git push"

        a = _make_agent(is_sub_agent=True, confirm_fn=_fn)
        self.assertTrue(asyncio.run(a._confirm_dangerous("git push")))
        self.assertFalse(asyncio.run(a._confirm_dangerous("rm -rf /")))

    def test_accept_edits_dangerous_shell_still_confirms(self):
        # acceptEdits 只放行编辑类工具；危险 shell 仍需 confirm。
        result = tools.check_permission("run_shell", {"command": "git push"}, mode="acceptEdits")
        self.assertEqual(result["action"], "confirm")

    def test_unverified_marker_appended(self):
        self.assertEqual(
            Agent._append_unverified_marker("done", False),
            "done\n\n[unverified] 子代理产物未通过 L1 验证",
        )
        self.assertEqual(Agent._append_unverified_marker("done", True), "done")
        self.assertEqual(Agent._append_unverified_marker("done", None), "done")


class TestFilterL1Rules(unittest.TestCase):
    """Task 4: 子代理只跑 L1 规则。"""

    def test_filters_to_level_one_only(self):
        rules = [
            VerificationRule({"id": "l1", "level": 1, "type": "file_exists", "target": "a"}),
            VerificationRule({"id": "l1b", "level": 1, "type": "glob_exists", "target": "*.py"}),
            VerificationRule({"id": "l2", "level": 2, "type": "file_contains", "target": "a", "pattern": "x"}),
            VerificationRule({"id": "l3", "level": 3, "type": "command_success", "command": "true"}),
        ]
        filtered = agent_mod._filter_l1_rules(rules)
        self.assertEqual([r.id for r in filtered], ["l1", "l1b"])
        for r in filtered:
            self.assertEqual(r.level, 1)

    def test_empty_list(self):
        self.assertEqual(agent_mod._filter_l1_rules([]), [])


class TestSubAgentL1Verification(unittest.TestCase):
    """Task 4: _verify_before_done 对子代理只跑 L1（失败注入 feedback、attempt 计数）。"""

    def test_sub_agent_l1_only(self):
        l1 = VerificationRule({"id": "l1", "level": 1, "type": "file_exists", "target": "missing.txt"})
        l2 = VerificationRule({"id": "l2", "level": 2, "type": "file_contains",
                               "target": "missing.txt", "pattern": "x"})
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                a = _make_agent()
                a.is_sub_agent = True
                with patch("agents.agent.load_verification_rules", return_value=[l1, l2]):
                    result = asyncio.run(a._verify_before_done())
                self.assertFalse(result)
                self.assertFalse(a._last_verification_passed)
                self.assertEqual(len(a._verification_log), 1)
                # 只有 L1 规则被计入失败列表（L2 被过滤）
                self.assertEqual([f["id"] for f in a._verification_log[0]["failures"]], ["l1"])
                self.assertEqual(a._verification_log[0]["total"], 1)
            finally:
                os.chdir(old_cwd)

    def test_sub_agent_l1_pass(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                Path("out.txt").write_text("x", encoding="utf-8")
                a = _make_agent()
                a.is_sub_agent = True
                a._written_files.add("out.txt")
                with patch("agents.agent.load_verification_rules", return_value=[]):
                    result = asyncio.run(a._verify_before_done())
                self.assertTrue(result)
                self.assertTrue(a._last_verification_passed)
            finally:
                os.chdir(old_cwd)


class TestCheckpointVerification(unittest.TestCase):
    """Task 13: 中途 L1 检查点。"""

    def _restore_env(self, key, old):
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old

    def test_checkpoint_interval_env(self):
        a = _make_agent()
        old = os.environ.get("OTTER_VERIFY_CHECKPOINT_EVERY")
        try:
            os.environ["OTTER_VERIFY_CHECKPOINT_EVERY"] = "3"
            self.assertEqual(a._checkpoint_interval(), 3)
            os.environ["OTTER_VERIFY_CHECKPOINT_EVERY"] = "0"
            self.assertEqual(a._checkpoint_interval(), 5)
            os.environ["OTTER_VERIFY_CHECKPOINT_EVERY"] = "abc"
            self.assertEqual(a._checkpoint_interval(), 5)
            os.environ.pop("OTTER_VERIFY_CHECKPOINT_EVERY", None)
            self.assertEqual(a._checkpoint_interval(), 5)
        finally:
            self._restore_env("OTTER_VERIFY_CHECKPOINT_EVERY", old)

    def test_checkpoint_injects_feedback_on_failure(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                a = _make_agent()
                a._written_files.add("missing.txt")
                asyncio.run(a._run_checkpoint_verification())
                injected = [m["content"] for m in a._openai_messages if isinstance(m.get("content"), str)]
                self.assertTrue(any("[Verification Report]" in c for c in injected))
            finally:
                os.chdir(old_cwd)

    def test_checkpoint_skipped_for_sub_agent(self):
        a = _make_agent(is_sub_agent=True)
        a._written_files.add("missing.txt")
        asyncio.run(a._run_checkpoint_verification())
        injected = [m["content"] for m in a._openai_messages if isinstance(m.get("content"), str)]
        self.assertFalse(any("[Verification Report]" in c for c in injected))


class TestCleanupStaleToolResults(unittest.TestCase):
    """Task 11: 大结果临时文件过期清理。"""

    def test_old_removed_new_kept(self):
        a = _make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            old = d / "old.txt"
            new = d / "new.txt"
            old.write_text("x", encoding="utf-8")
            new.write_text("y", encoding="utf-8")
            old_ts = time.time() - 8 * 86400
            os.utime(old, (old_ts, old_ts))
            removed = a._cleanup_stale_tool_results(directory=d)
            self.assertEqual(removed, 1)
            self.assertFalse(old.exists())
            self.assertTrue(new.exists())

    def test_missing_directory_is_tolerated(self):
        a = _make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(a._cleanup_stale_tool_results(directory=Path(tmp) / "nope"), 0)

    def test_fresh_files_untouched(self):
        a = _make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            f = d / "recent.txt"
            f.write_text("x", encoding="utf-8")
            self.assertEqual(a._cleanup_stale_tool_results(directory=d), 0)
            self.assertTrue(f.exists())


class TestReadFileStatePersistence(unittest.TestCase):
    """Task 8: _read_file_state 会话持久化（read-before-edit 保护跨会话保留）。"""

    def test_auto_save_serializes_read_file_state(self):
        a = _make_agent()
        a._read_file_state = {"a.py": 123.0, "b.py": 456.5}
        with patch("agents.agent.save_session") as mock_save:
            a._auto_save()
        data = mock_save.call_args[0][1]
        self.assertEqual(data["readFileState"], {"a.py": 123.0, "b.py": 456.5})

    def test_restore_session_recovers_read_file_state(self):
        a = _make_agent()
        a.restore_session({
            "readFileState": {"b.py": 456.0},
            "anthropicMessages": [],
            "openaiMessages": [],
        })
        self.assertEqual(a._read_file_state, {"b.py": 456.0})

    def test_restore_filters_invalid_values(self):
        a = _make_agent()
        a.restore_session({"readFileState": {"ok.py": 1.0, "bad": "string", "bool": True}})
        self.assertEqual(a._read_file_state, {"ok.py": 1.0})

    def test_restore_without_read_file_state_keeps_default(self):
        a = _make_agent()
        a.restore_session({"anthropicMessages": [], "openaiMessages": []})
        self.assertEqual(a._read_file_state, {})

    def test_auto_save_keeps_custom_title(self):
        # webui 重命名写入 metadata.title 后，_auto_save 不得覆盖掉
        a = _make_agent()
        with patch("agents.agent.load_session") as mock_load, patch("agents.agent.save_session") as mock_save:
            mock_load.return_value = {"metadata": {"id": "x", "title": "自定义标题"}}
            a._auto_save()
        data = mock_save.call_args[0][1]
        self.assertEqual(data["metadata"]["title"], "自定义标题")

    def test_auto_save_no_title_when_absent(self):
        # 旧会话无 title 时保持原行为（不新增字段）
        a = _make_agent()
        with patch("agents.agent.load_session") as mock_load, patch("agents.agent.save_session") as mock_save:
            mock_load.return_value = {"metadata": {"id": "x"}}
            a._auto_save()
        data = mock_save.call_args[0][1]
        self.assertNotIn("title", data["metadata"])

    def test_auto_save_task_and_outcome_fields(self):
        # 任务级单元：metadata 含 task（首条 user 消息）与 outcome（最近验证结果）
        a = _make_agent()
        a._anthropic_messages = [{"role": "user", "content": "修复登录 bug，这是一个很长的任务描述"}]
        a._verification_log = [{"attempt": 1, "passed": False, "total": 3, "failures": []}]
        with patch("agents.agent.load_session") as mock_load, patch("agents.agent.save_session") as mock_save:
            mock_load.return_value = None
            a._auto_save()
        data = mock_save.call_args[0][1]
        self.assertTrue(data["metadata"]["task"].startswith("修复登录 bug"))
        self.assertEqual(data["metadata"]["outcome"], "fail")
        # 无验证记录 → unknown
        a._verification_log = []
        with patch("agents.agent.load_session") as mock_load, patch("agents.agent.save_session") as mock_save:
            mock_load.return_value = None
            a._auto_save()
        data = mock_save.call_args[0][1]
        self.assertEqual(data["metadata"]["outcome"], "unknown")

    def test_verification_log_has_message_index(self):
        # 失败轨迹关联键：验证条目记录触发时的消息位置
        l1 = VerificationRule({"id": "l1", "level": 1, "type": "file_exists", "target": "missing.txt"})
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                a = _make_agent()
                # use_openai 默认 True：失败轨迹关联键取自 openai 消息列表
                a._openai_messages = [
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "content": "y"},
                ]
                with patch("agents.agent.load_verification_rules", return_value=[l1]):
                    asyncio.run(a._verify_before_done())
                entry = a._verification_log[-1]
                self.assertFalse(entry["passed"])
                self.assertEqual(entry["message_index"], 2)
            finally:
                os.chdir(old_cwd)


class TestRunOnceVerifiedDefault(unittest.TestCase):
    """Task 4: run_once 未触发验证时 verified 默认 True（不依赖完整 chat 循环）。"""

    def test_verified_defaults_true_without_verification(self):
        a = _make_agent()
        # 未触发验证时 _last_verification_passed 为 None → run_once 默认 True 的等价逻辑
        verified = a._last_verification_passed if a._last_verification_passed is not None else True
        self.assertTrue(verified)


class TestExecuteToolCallStructuredContract(unittest.TestCase):
    """Task 18: _execute_tool_call 适配 execute_tool 的 dict 契约（仍对外返回 str）。"""

    def test_returns_content_on_success(self):
        a = _make_agent()
        with patch.object(
            agent_mod, "execute_tool",
            new=AsyncMock(return_value={"content": "ok-text", "error": None, "retryable": False}),
        ) as mock_exec:
            result = asyncio.run(a._execute_tool_call("read_file", {"file_path": "a.py"}))
        self.assertEqual(result, "ok-text")
        mock_exec.assert_awaited_once_with("read_file", {"file_path": "a.py"}, a._read_file_state)

    def test_returns_error_text_when_content_empty(self):
        a = _make_agent()
        with patch.object(
            agent_mod, "execute_tool",
            new=AsyncMock(return_value={"content": "", "error": "boom", "retryable": False}),
        ):
            result = asyncio.run(a._execute_tool_call("read_file", {"file_path": "a.py"}))
        self.assertEqual(result, "boom")

    def test_skill_create_refresh_uses_dict_content(self):
        a = _make_agent()
        a._refresh_runtime_system_prompt = MagicMock()
        with patch.object(
            agent_mod, "execute_tool",
            new=AsyncMock(return_value={"content": '{"ok": true}', "error": None, "retryable": False}),
        ):
            result = asyncio.run(a._execute_tool_call("skill_create", {"name": "demo"}))
        self.assertEqual(result, '{"ok": true}')
        a._refresh_runtime_system_prompt.assert_called_once()

    def test_write_file_still_records_written_path_with_dict_result(self):
        a = _make_agent()
        with patch.object(
            agent_mod, "execute_tool",
            new=AsyncMock(return_value={"content": "ok", "error": None, "retryable": False}),
        ):
            asyncio.run(a._execute_tool_call("write_file", {"file_path": "a.txt", "content": "x"}))
        self.assertIn("a.txt", a._written_files)


class _FakeChunk:
    """最小可用的流式 chunk 桩（仅暴露 _call_openai_stream 会访问的属性）。"""

    def __init__(self, content=None, tool_calls=None, finish_reason=None, usage=None):
        self.usage = usage
        self.choices = [] if finish_reason is None else [
            type("_Choice", (), {
                "delta": type("_Delta", (), {"content": content, "tool_calls": tool_calls})(),
                "finish_reason": finish_reason,
            })()
        ]


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class TestOpenAIMultiToolCallOnce(unittest.TestCase):
    """Task 1: 模型单次返回多个 tool_calls 时，每个工具只执行一次、每条 tool 消息只回写一次。"""

    @staticmethod
    def _tc(tc_id, name, args=None):
        return {
            "id": tc_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})},
        }

    @staticmethod
    def _resp(tool_calls):
        return {
            "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": tool_calls}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    def test_openai_three_tool_calls_each_executed_once(self):
        a = _make_agent(is_sub_agent=True)
        resp1 = self._resp([
            self._tc("tc1", "tool_a", {"x": 1}),
            self._tc("tc2", "tool_b", {"x": 2}),
            self._tc("tc3", "tool_c", {"x": 3}),
        ])
        resp2 = self._resp(None)  # 无 tool_calls，结束循环
        exec_mock = AsyncMock(return_value="ok-result")
        with (
            patch.object(Agent, "_call_openai_stream", new=AsyncMock(side_effect=[resp1, resp2])),
            patch("agents.agent.check_permission", return_value={"action": "allow"}),
            patch.object(Agent, "_execute_tool_call", new=exec_mock),
            patch.object(Agent, "_verify_before_done", new=AsyncMock(return_value=True)),
        ):
            asyncio.run(a._chat_openai("do three things"))

        # 每个工具恰好执行一次
        self.assertEqual(exec_mock.call_count, 3)
        names = [c.args[0] for c in exec_mock.call_args_list]
        self.assertEqual(sorted(names), ["tool_a", "tool_b", "tool_c"])
        # 每条 tool 消息只回写一次
        tool_msgs = [m for m in a._openai_messages if m.get("role") == "tool"]
        ids = [m["tool_call_id"] for m in tool_msgs]
        self.assertEqual(ids, ["tc1", "tc2", "tc3"])
        for tid in ("tc1", "tc2", "tc3"):
            self.assertEqual(ids.count(tid), 1)

    def test_openai_denied_tool_writes_result_once(self):
        # 拒绝分支的 result 在收集阶段写入，执行阶段只 append 一次，且不执行工具。
        a = _make_agent(is_sub_agent=True)

        def _perm(name, inp, mode, plan):
            return {"action": "deny", "message": "blocked"} if name == "tool_b" else {"action": "allow"}

        resp1 = self._resp([
            self._tc("tc1", "tool_a", {"x": 1}),
            self._tc("tc2", "tool_b", {"x": 2}),
            self._tc("tc3", "tool_c", {"x": 3}),
        ])
        resp2 = self._resp(None)
        exec_mock = AsyncMock(return_value="ok-result")
        with (
            patch.object(Agent, "_call_openai_stream", new=AsyncMock(side_effect=[resp1, resp2])),
            patch("agents.agent.check_permission", side_effect=_perm),
            patch.object(Agent, "_execute_tool_call", new=exec_mock),
            patch.object(Agent, "_verify_before_done", new=AsyncMock(return_value=True)),
        ):
            asyncio.run(a._chat_openai("do three things"))

        # 被拒绝的工具不执行，其余各执行一次
        self.assertEqual(exec_mock.call_count, 2)
        names = [c.args[0] for c in exec_mock.call_args_list]
        self.assertEqual(sorted(names), ["tool_a", "tool_c"])
        # 三条 tool 消息各一次，拒绝消息保留
        tool_msgs = [m for m in a._openai_messages if m.get("role") == "tool"]
        ids = [m["tool_call_id"] for m in tool_msgs]
        self.assertEqual(ids, ["tc1", "tc2", "tc3"])
        denied = [m for m in tool_msgs if m["tool_call_id"] == "tc2"]
        self.assertEqual(denied[0]["content"], "Action denied: blocked")


class TestOpenAISendDedupe(unittest.TestCase):
    """Task 2: _call_openai_stream 发送前对消息历史做 tool_call_id 去重（不改动原历史）。"""

    def test_openai_dedup_tool_call_ids_before_send(self):
        a = _make_agent()
        a._openai_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                {"id": "tc2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                {"id": "tc1", "type": "function", "function": {"name": "c", "arguments": "{}"}},  # 重复 id
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "first"},
            {"role": "tool", "tool_call_id": "tc1", "content": "second"},  # 重复 tool_call_id
            {"role": "tool", "tool_call_id": "tc2", "content": "x"},
        ]
        original = [dict(m) for m in a._openai_messages]
        with patch.object(
            a._openai_client.chat.completions, "create",
            new=AsyncMock(return_value=_FakeStream([])),
        ) as mock_create:
            asyncio.run(a._call_openai_stream())

        sent = mock_create.call_args.kwargs["messages"]
        # tool 消息：同 tool_call_id 只保留最先出现者
        tool_msgs = [m for m in sent if m.get("role") == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["tc1", "tc2"])
        self.assertEqual(tool_msgs[0]["content"], "first")
        # assistant tool_calls：同 id 只保留首个
        asst = [m for m in sent if m.get("role") == "assistant"]
        self.assertEqual(len(asst), 1)
        self.assertEqual([tc["id"] for tc in asst[0]["tool_calls"]], ["tc1", "tc2"])
        # 原历史不被修改
        self.assertEqual(a._openai_messages, original)
        self.assertEqual(len([m for m in a._openai_messages if m.get("role") == "tool"]), 3)


class TestCompactSummaryVersioning(unittest.TestCase):
    """Task 17: compact summary 记忆 name 带时间戳版本化，不再覆盖历史。"""

    def test_two_persist_calls_produce_distinct_timestamped_names(self):
        a = _make_agent()
        with patch.object(agent_mod, "save_memory_structured") as mock_save:
            a._persist_compact_summary("first summary")
            a._persist_compact_summary("second summary")

        self.assertEqual(mock_save.call_count, 2)
        names = [c.kwargs["name"] for c in mock_save.call_args_list]
        for name in names:
            self.assertTrue(name.startswith("conversation-compact-summary-"))
            self.assertRegex(name, r"^conversation-compact-summary-\d{14}-\d{9}$")
        self.assertNotEqual(names[0], names[1])
        for c in mock_save.call_args_list:
            self.assertEqual(c.kwargs["description"], "Auto-saved conversation summary from context compaction")
            self.assertEqual(c.kwargs["type"], "project")
            self.assertEqual(c.kwargs["session_id"], a.session_id)

    def test_skipped_when_empty_or_sub_agent(self):
        a = _make_agent()
        with patch.object(agent_mod, "save_memory_structured") as mock_save:
            a._persist_compact_summary("")
            a.is_sub_agent = True
            a._persist_compact_summary("some text")
        mock_save.assert_not_called()


class TestCompactRequestSanitization(unittest.TestCase):
    """Task 1: 摘要压缩请求清洗 —— 工具轮后触发压缩时，发送给摘要模型的
    请求不含孤立 tool_use/tool_result/tool_calls，且角色严格交替。"""

    def test_compact_strip_anthropic_removes_unpaired_tool_blocks(self):
        messages = [
            {"role": "user", "content": "请读取 a.py"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "好的"},
                {"type": "tool_use", "id": "tu1", "name": "read_file", "input": {}},
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "内容"}]},
            {"role": "user", "content": "继续"},
        ]
        out = agent_mod._strip_unpaired_tool_blocks(messages)
        # 无孤立 tool_use/tool_result block
        for msg in out:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    self.assertNotIn(block.get("type"), ("tool_use", "tool_result"))
        # 过滤后无内容的消息丢弃；角色交替
        self.assertEqual([m.get("content") for m in out],
                         ["请读取 a.py", [{"type": "text", "text": "好的"}], "继续"])
        roles = [m.get("role") for m in out]
        self.assertEqual(roles, ["user", "assistant", "user"])

    def test_compact_strip_openai_removes_tool_calls_and_tool_role(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "请读取 a.py"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "内容"},
            {"role": "assistant", "content": "已读取"},
            {"role": "user", "content": "继续"},
        ]
        out = agent_mod._strip_unpaired_tool_blocks(messages)
        for msg in out:
            self.assertNotIn("tool_calls", msg)
            self.assertNotEqual(msg.get("role"), "tool")
            self.assertTrue(msg.get("content"))
        roles = [m.get("role") for m in out]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_compact_strip_merges_consecutive_same_role(self):
        messages = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},  # 连续同角色：丢弃后者
        ]
        out = agent_mod._strip_unpaired_tool_blocks(messages)
        self.assertEqual([m.get("content") for m in out], ["a"])

    @staticmethod
    def _assert_protocol_legal(messages):
        """请求/历史协议合法：有 role、无空消息、无工具 block、角色交替。"""
        roles = []
        for msg in messages:
            assert isinstance(msg, dict)
            roles.append(msg.get("role"))
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    assert isinstance(block, dict), f"non-dict block: {block}"
                    assert block.get("type") not in ("tool_use", "tool_result"), f"orphan block: {block}"
            else:
                assert content, f"empty content in message: {msg}"
        for r1, r2 in zip(roles, roles[1:]):
            assert r1 != r2, f"consecutive same role: {roles}"

    def test_compact_anthropic_request_protocol_legal_after_tool_round(self):
        a = _make_agent(api_base=None, is_sub_agent=True)
        # 历史尾为 [assistant(tool_use), user(tool_results)] 的多轮工具历史
        a._anthropic_messages = [
            {"role": "user", "content": "请读取项目结构"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1", "name": "read_file",
                                               "input": {"file_path": "a.py"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "src/main.py"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu2", "name": "grep_search", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu2", "content": "hits"}]},
        ]
        a._anthropic_client = MagicMock()
        a._anthropic_client.messages.create = AsyncMock(
            return_value=MagicMock(content=[MagicMock(type="text", text="SUMMARY")])
        )
        asyncio.run(a._compact_anthropic())
        sent = a._anthropic_client.messages.create.call_args.kwargs["messages"]
        self._assert_protocol_legal(sent)
        self.assertIn("Summarize the conversation", " ".join(str(m) for m in sent))
        # 压缩后历史同样无孤儿 tool_result、无连续同角色
        self._assert_protocol_legal(a._anthropic_messages)

    def test_compact_openai_request_protocol_legal_after_tool_round(self):
        a = _make_agent(is_sub_agent=True)
        a._openai_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "请读取 a.py"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "file content"},
            {"role": "assistant", "content": "已读取"},
            {"role": "user", "content": "继续"},
        ]
        a._openai_client = MagicMock()
        a._openai_client.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="SUMMARY"))])
        )
        asyncio.run(a._compact_openai())
        sent = a._openai_client.chat.completions.create.call_args.kwargs["messages"]
        for msg in sent:
            self.assertNotIn("tool_calls", msg)
            self.assertNotEqual(msg.get("role"), "tool")
            self.assertTrue(msg.get("content"))
        roles = [m.get("role") for m in sent]
        self.assertNotIn("tool", roles)
        for r1, r2 in zip(roles, roles[1:]):
            self.assertNotEqual(r1, r2)
        # 压缩后历史无 tool_calls / tool 角色 / 连续同角色
        for msg in a._openai_messages:
            self.assertNotIn("tool_calls", msg)
            self.assertNotEqual(msg.get("role"), "tool")
        roles = [m.get("role") for m in a._openai_messages]
        for r1, r2 in zip(roles, roles[1:]):
            self.assertNotEqual(r1, r2)
        self.assertEqual(a._openai_messages[1]["content"], "[Previous conversation summary]\nSUMMARY")
        self.assertEqual(a._openai_messages[-1]["content"], "继续")


class TestNormalizeOrphanToolResultCascade(unittest.TestCase):
    """Task 2: 半截轮（部分 tool_result）恢复时，紧随被跳过 assistant 的
    纯孤儿 tool_result 消息（无文本）一并丢弃。"""

    def test_normalize_half_round_orphan_tool_result_cascaded_drop(self):
        a = _make_agent()
        msgs = [
            {"role": "user", "content": "开始"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu1", "name": "read_file", "input": {}},
                {"type": "tool_use", "id": "tu2", "name": "grep_search", "input": {}},
            ]},
            # 只有 tu1 的结果，tu2 缺失 → 半截轮
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "r1"}]},
            {"role": "user", "content": "后续对话"},
        ]
        out = a._normalize_anthropic_messages(msgs)
        # 输出历史中不存在指向已删除 tool_use 的 tool_result 消息
        for msg in out:
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    self.assertNotEqual(block.get("type"), "tool_result")
        self.assertEqual([m.get("content") for m in out], ["开始", "后续对话"])

    def test_normalize_mixed_follower_with_text_is_kept(self):
        # 半截轮中孤儿结果旁有文本 → 消息保留（文本有价值），仅 assistant 被跳过
        a = _make_agent()
        msgs = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu1", "name": "read_file", "input": {}},
                {"type": "tool_use", "id": "tu2", "name": "grep_search", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu1", "content": "r1"},
                {"type": "text", "text": "请继续"},
            ]},
        ]
        out = a._normalize_anthropic_messages(msgs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["role"], "user")
        self.assertEqual(len(out[0]["content"]), 2)

    def test_normalize_restore_session_drops_half_round_with_orphan_result(self):
        # 会话恢复走 restore_session → normalize，恢复后无孤儿 tool_result
        a = _make_agent()
        a.restore_session({
            "anthropicMessages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tu1", "name": "read_file", "input": {}},
                    {"type": "tool_use", "id": "tu2", "name": "grep_search", "input": {}},
                ]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "r1"}]},
            ],
            "openaiMessages": [],
        })
        self.assertEqual(a._anthropic_messages, [])


class TestBudgetBackfillAndUserMerge(unittest.TestCase):
    """Task 3: OpenAI budget 超限回填 skipped tool 消息（无孤儿 tool_calls）；
    budget 以 user 结尾后再次 chat 追加 user 文本时合并，避免连续 user。"""

    @staticmethod
    def _tc(tc_id, name, args=None):
        return {
            "id": tc_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})},
        }

    @staticmethod
    def _resp(tool_calls):
        return {
            "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": tool_calls}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    def test_openai_budget_backfills_skipped_tool_messages(self):
        a = _make_agent(is_sub_agent=True, max_cost_usd=0.0)
        resp1 = self._resp([self._tc("tc1", "tool_a", {"x": 1}), self._tc("tc2", "tool_b", {"x": 2})])
        exec_mock = AsyncMock(return_value="should-not-run")
        with (
            patch.object(Agent, "_call_openai_stream", new=AsyncMock(return_value=resp1)),
            patch.object(Agent, "_execute_tool_call", new=exec_mock),
        ):
            asyncio.run(a._chat_openai("do things"))

        # 预算超限不执行任何工具
        exec_mock.assert_not_called()
        tool_msgs = [m for m in a._openai_messages if m.get("role") == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["tc1", "tc2"])
        for m in tool_msgs:
            self.assertIn("Tool execution skipped", m["content"])
        # 无孤儿 tool_calls：每个 assistant tool_call 都有对应 tool 消息
        asst = [m for m in a._openai_messages if m.get("role") == "assistant"]
        tc_ids = {tc["id"] for m in asst for tc in (m.get("tool_calls") or [])}
        self.assertEqual(tc_ids, {"tc1", "tc2"})
        self.assertEqual({m["tool_call_id"] for m in tool_msgs}, tc_ids)

    def test_anthropic_chat_after_budget_user_tail_merges(self):
        a = _make_agent(api_base=None, is_sub_agent=True, max_cost_usd=0.0)
        # 模拟 Anthropic budget 分支结尾：assistant(tool_use) + user(tool_result skipped)
        a._anthropic_messages = [
            {"role": "user", "content": "开始"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1", "name": "read_file", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1",
                                          "content": "Tool execution skipped: Cost limit reached"}]},
        ]
        fake = MagicMock()
        fake.usage.input_tokens = 10
        fake.usage.output_tokens = 5
        fake.content = []
        with (
            patch.object(Agent, "_call_anthropic_stream", new=AsyncMock(return_value=fake)),
            patch.object(Agent, "_verify_before_done", new=AsyncMock(return_value=True)),
        ):
            asyncio.run(a._chat_anthropic("继续干活"))

        # 消息列表不存在连续两条 user
        roles = [m.get("role") for m in a._anthropic_messages]
        self.assertFalse(any(r1 == r2 == "user" for r1, r2 in zip(roles, roles[1:])))
        # 新 user 文本合并进最后一条 user（list 追加 text block），不新增消息
        user_msgs = [m for m in a._anthropic_messages if m.get("role") == "user"]
        last_user = user_msgs[-1]
        text_blocks = [b for b in last_user["content"]
                       if isinstance(b, dict) and b.get("type") == "text"]
        self.assertTrue(any("继续干活" in b["text"] for b in text_blocks))
        self.assertEqual(len(a._anthropic_messages), 4)

    def test_budget_append_user_text_openai_str_merge(self):
        a = _make_agent(is_sub_agent=True)
        a._openai_messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "first"}]
        a._append_user_text("second")
        self.assertEqual(len(a._openai_messages), 2)
        self.assertEqual(a._openai_messages[-1]["content"], "first\n\nsecond")
        # 末尾不是 user 时正常追加
        a._openai_messages.append({"role": "assistant", "content": "reply"})
        a._append_user_text("third")
        self.assertEqual(a._openai_messages[-1], {"role": "user", "content": "third"})

    def test_budget_append_user_text_anthropic_merge(self):
        a = _make_agent(api_base=None, is_sub_agent=True)
        a._anthropic_messages = [{"role": "user", "content": "first"}]
        a._append_user_text("second")
        self.assertEqual(a._anthropic_messages[-1]["content"], "first\n\nsecond")
        # list 内容：追加 text block
        a._anthropic_messages[-1]["content"] = [{"type": "tool_result", "tool_use_id": "tu1", "content": "r"}]
        a._append_user_text("third")
        last = a._anthropic_messages[-1]
        self.assertEqual([b["type"] for b in last["content"]], ["tool_result", "text"])
        self.assertEqual(last["content"][-1]["text"], "third")


class TestOpenAIExecutionErrorProtection(unittest.TestCase):
    """Task 4: OpenAI 工具执行异常（并发与串行）捕获并回填错误，整轮继续。"""

    @staticmethod
    def _tc(tc_id, name, args=None):
        return {
            "id": tc_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})},
        }

    @staticmethod
    def _resp(tool_calls):
        return {
            "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": tool_calls}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    def test_openai_concurrent_batch_error_backfilled_and_round_continues(self):
        a = _make_agent(is_sub_agent=True)
        resp1 = self._resp([
            self._tc("tc1", "read_file", {"file_path": "a.py"}),
            self._tc("tc2", "grep_search", {"pattern": "x"}),
            self._tc("tc3", "read_file", {"file_path": "b.py"}),
        ])
        resp2 = self._resp(None)
        exec_count = 0

        async def _exec(self, name, inp):
            nonlocal exec_count
            exec_count += 1
            if name == "grep_search":
                raise RuntimeError("MCP disconnected")
            return f"ok-{name}"

        with (
            patch.object(Agent, "_call_openai_stream", new=AsyncMock(side_effect=[resp1, resp2])),
            patch("agents.agent.check_permission", return_value={"action": "allow"}),
            patch.object(Agent, "_execute_tool_call", new=_exec),
            patch.object(Agent, "_verify_before_done", new=AsyncMock(return_value=True)),
        ):
            # 整轮不抛异常即通过
            asyncio.run(a._chat_openai("run tools"))

        self.assertEqual(exec_count, 3)
        tool_msgs = [m for m in a._openai_messages if m.get("role") == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["tc1", "tc2", "tc3"])
        by_id = {m["tool_call_id"]: m["content"] for m in tool_msgs}
        self.assertIn("Error executing tool: MCP disconnected", by_id["tc2"])
        self.assertEqual(by_id["tc1"], "ok-read_file")
        self.assertEqual(by_id["tc3"], "ok-read_file")

    def test_openai_serial_branch_error_backfilled(self):
        a = _make_agent(is_sub_agent=True)
        resp1 = self._resp([
            self._tc("tc1", "run_shell", {"command": "echo hi"}),
            self._tc("tc2", "tool_b", {"x": 2}),
        ])
        resp2 = self._resp(None)

        async def _exec(self, name, inp):
            if name == "run_shell":
                raise ValueError("boom")
            return "ok"

        with (
            patch.object(Agent, "_call_openai_stream", new=AsyncMock(side_effect=[resp1, resp2])),
            patch("agents.agent.check_permission", return_value={"action": "allow"}),
            patch.object(Agent, "_execute_tool_call", new=_exec),
            patch.object(Agent, "_verify_before_done", new=AsyncMock(return_value=True)),
        ):
            asyncio.run(a._chat_openai("run"))

        tool_msgs = [m for m in a._openai_messages if m.get("role") == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["tc1", "tc2"])
        self.assertIn("Error executing tool: boom", tool_msgs[0]["content"])
        self.assertEqual(tool_msgs[1]["content"], "ok")


class TestToolIdDedupAndEarlyTaskReplay(unittest.TestCase):
    """Task 5: 执行层按 id 去重（重复 id 只执行一次、只回填一次）；
    流式提前执行防重放（同 id 两次回调只建一个 early task）。"""

    @staticmethod
    def _tc(tc_id, name, args=None):
        return {
            "id": tc_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})},
        }

    @staticmethod
    def _resp(tool_calls):
        return {
            "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": tool_calls}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    def test_dedup_openai_duplicate_tool_call_id_executed_once(self):
        a = _make_agent(is_sub_agent=True)
        resp1 = self._resp([
            self._tc("tc1", "tool_a", {"x": 1}),
            self._tc("tc1", "tool_a", {"x": 1}),  # 网关重放重复 id
            self._tc("tc2", "tool_b", {"x": 2}),
        ])
        resp2 = self._resp(None)
        exec_mock = AsyncMock(return_value="ok")
        with (
            patch.object(Agent, "_call_openai_stream", new=AsyncMock(side_effect=[resp1, resp2])),
            patch("agents.agent.check_permission", return_value={"action": "allow"}),
            patch.object(Agent, "_execute_tool_call", new=exec_mock),
            patch.object(Agent, "_verify_before_done", new=AsyncMock(return_value=True)),
        ):
            asyncio.run(a._chat_openai("run"))

        # 重复 id 只执行一次
        tool_a_calls = [c for c in exec_mock.call_args_list if c.args[0] == "tool_a"]
        self.assertEqual(len(tool_a_calls), 1)
        self.assertEqual(exec_mock.call_count, 2)
        # tool 消息只回填一次
        tool_msgs = [m for m in a._openai_messages if m.get("role") == "tool"]
        ids = [m["tool_call_id"] for m in tool_msgs]
        self.assertEqual(ids, ["tc1", "tc2"])

    def test_dedup_anthropic_duplicate_tool_use_id_executed_once(self):
        a = _make_agent(api_base=None, is_sub_agent=True)
        tu = SimpleNamespace(type="tool_use", id="tu1", name="read_file", input={"file_path": "a.py"})
        fake1 = MagicMock()
        fake1.usage.input_tokens = 10
        fake1.usage.output_tokens = 5
        fake1.content = [tu, tu]  # 网关重放重复 id
        fake2 = MagicMock()
        fake2.usage.input_tokens = 1
        fake2.usage.output_tokens = 1
        fake2.content = []
        exec_mock = AsyncMock(return_value="ok")
        with (
            patch.object(Agent, "_call_anthropic_stream", new=AsyncMock(side_effect=[fake1, fake2])),
            patch("agents.agent.check_permission", return_value={"action": "allow"}),
            patch.object(Agent, "_execute_tool_call", new=exec_mock),
            patch.object(Agent, "_verify_before_done", new=AsyncMock(return_value=True)),
        ):
            asyncio.run(a._chat_anthropic("read"))

        # 该 id 的工具只执行 1 次
        self.assertEqual(exec_mock.call_count, 1)
        # tool_result 只回填 1 条
        user_msgs = [m for m in a._anthropic_messages
                     if m.get("role") == "user" and isinstance(m.get("content"), list)]
        results = [b for m in user_msgs for b in m["content"]
                   if isinstance(b, dict) and b.get("type") == "tool_result"]
        self.assertEqual(len(results), 1)
        # 写入历史的 assistant tool_use 也去重（保留首个）
        asst = [m for m in a._anthropic_messages if m.get("role") == "assistant"]
        use_blocks = [b for m in asst for b in (m.get("content") or [])
                      if isinstance(b, dict) and b.get("type") == "tool_use"]
        self.assertEqual(len(use_blocks), 1)

    def test_dedup_on_tool_block_duplicate_id_creates_single_task(self):
        a = _make_agent(api_base=None, is_sub_agent=True)
        calls = {"n": 0}
        block = {"type": "tool_use", "id": "tu1", "name": "read_file", "input": {"file_path": "a.py"}}
        tu = SimpleNamespace(type="tool_use", id="tu1", name="read_file", input={"file_path": "a.py"})

        def _resp(content):
            f = MagicMock()
            f.usage.input_tokens = 10
            f.usage.output_tokens = 5
            f.content = content
            return f

        async def _fake_stream(self, on_tool_block_complete=None):
            calls["n"] += 1
            if calls["n"] == 1 and on_tool_block_complete:
                on_tool_block_complete(block)
                on_tool_block_complete(block)  # 同一 id 重复触发 content_block_stop
                return _resp([tu])
            return _resp([])

        exec_mock = AsyncMock(return_value="ok")
        with (
            patch.object(Agent, "_call_anthropic_stream", new=_fake_stream),
            patch("agents.agent.check_permission", return_value={"action": "allow"}),
            patch.object(Agent, "_execute_tool_call", new=exec_mock),
            patch.object(Agent, "_verify_before_done", new=AsyncMock(return_value=True)),
        ):
            asyncio.run(a._chat_anthropic("read"))

        # 同 id 两次回调只建一个 early task → 工具只执行一次
        self.assertEqual(exec_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
