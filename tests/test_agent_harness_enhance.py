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


if __name__ == "__main__":
    unittest.main()
