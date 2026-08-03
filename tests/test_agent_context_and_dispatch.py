# tests/test_agent_context_and_dispatch.py
# 针对 agents/agent.py 中风险最高的两块纯逻辑路径的最小聚焦测试：
#   1. 上下文压缩触发条件：_check_and_compact / _compact_anthropic|_openai /
#      _budget_tool_results_* / _snip_stale_results_* / _microcompact_*
#   2. tool-call 解析与调度：_normalize_anthropic_messages / _find_tool_use_by_id /
#      _execute_tool_call 的路由分支
# 仅使用标准库 unittest，运行方式：python -m unittest discover -s tests

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_mod
from agents.agent import (
    Agent,
    KEEP_RECENT_RESULTS,
    MICROCOMPACT_IDLE_S,
    SNIP_PLACEHOLDER,
)


def _make_agent(use_openai: bool = False) -> Agent:
    """构造不触发任何网络/文件副作用的 Agent 实例。

    custom_system_prompt 避免走 build_system_prompt 的磁盘扫描；
    api_key/api_base 只用于客户端对象构造，测试中不会真正发请求。
    """
    kwargs = dict(
        model="claude-sonnet-4-6",
        api_key="test-key",
        custom_system_prompt="test prompt",
    )
    if use_openai:
        kwargs["model"] = "deepseek-chat"
        kwargs["api_base"] = "http://localhost:9/v1"
    return Agent(**kwargs)


class TestCheckAndCompactTrigger(unittest.TestCase):
    """_check_and_compact：仅当 last_input_token_count > effective_window * 0.85 时压缩。"""

    def test_triggers_above_85_percent(self):
        a = _make_agent()
        a.effective_window = 1000
        a.last_input_token_count = 851
        a._compact_conversation = AsyncMock()
        asyncio.run(a._check_and_compact())
        a._compact_conversation.assert_awaited_once()

    def test_does_not_trigger_at_or_below_85_percent(self):
        a = _make_agent()
        a.effective_window = 1000
        a._compact_conversation = AsyncMock()
        for tokens in (850, 500, 0):
            with self.subTest(tokens=tokens):
                a.last_input_token_count = tokens
                asyncio.run(a._check_and_compact())
                a._compact_conversation.assert_not_awaited()


class TestBudgetToolResults(unittest.TestCase):
    """第一级压缩：利用率 <0.5 不动；0.5~0.7 预算 30000；>0.7 预算 15000。"""

    def _anthropic_result_msg(self, content: str) -> dict:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": content}],
        }

    def test_anthropic_no_change_below_half_utilization(self):
        a = _make_agent()
        a.effective_window = 1000
        a.last_input_token_count = 400  # 0.4
        big = "x" * 40000
        a._anthropic_messages = [self._anthropic_result_msg(big)]
        a._budget_tool_results_anthropic()
        self.assertEqual(a._anthropic_messages[0]["content"][0]["content"], big)

    def test_anthropic_budget_30000_between_half_and_70_percent(self):
        a = _make_agent()
        a.effective_window = 1000
        a.last_input_token_count = 600  # 0.6 -> budget 30000
        a._anthropic_messages = [
            self._anthropic_result_msg("x" * 31000),  # 超预算，应截断
            self._anthropic_result_msg("y" * 20000),  # 未超预算，应保留
        ]
        a._budget_tool_results_anthropic()
        truncated = a._anthropic_messages[0]["content"][0]["content"]
        self.assertIn("[... budgeted:", truncated)
        self.assertLess(len(truncated), 31000)
        self.assertEqual(a._anthropic_messages[1]["content"][0]["content"], "y" * 20000)

    def test_anthropic_budget_tightens_to_15000_above_70_percent(self):
        a = _make_agent()
        a.effective_window = 1000
        a.last_input_token_count = 750  # 0.75 -> budget 15000
        a._anthropic_messages = [self._anthropic_result_msg("z" * 20000)]
        a._budget_tool_results_anthropic()
        self.assertIn("[... budgeted:", a._anthropic_messages[0]["content"][0]["content"])

    def test_openai_budget_truncates_tool_role_message(self):
        a = _make_agent(use_openai=True)
        a.effective_window = 1000
        a.last_input_token_count = 600
        a._openai_messages = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "tool_call_id": "c1", "content": "x" * 31000},
        ]
        a._budget_tool_results_openai()
        self.assertIn("[... budgeted:", a._openai_messages[1]["content"])
        self.assertEqual(a._openai_messages[0]["content"], "sys")


class TestSnipStaleResults(unittest.TestCase):
    """第二级压缩：0.60 阈值闸门 + 保留最近 3 条 + 重复 read_file 去重。"""

    def _openai_agent_with_tools(self, count: int) -> Agent:
        a = _make_agent(use_openai=True)
        a.effective_window = 1000
        a._openai_messages = [{"role": "system", "content": "sys"}] + [
            {"role": "tool", "tool_call_id": f"c{i}", "content": f"result-{i}"}
            for i in range(count)
        ]
        return a

    def test_openai_no_snip_below_threshold(self):
        a = self._openai_agent_with_tools(5)
        a.last_input_token_count = 500  # 0.5 < SNIP_THRESHOLD(0.60)
        a._snip_stale_results_openai()
        for i, msg in enumerate(a._openai_messages[1:]):
            self.assertEqual(msg["content"], f"result-{i}")

    def test_openai_snips_older_keeps_recent_three(self):
        a = self._openai_agent_with_tools(5)
        a.last_input_token_count = 700  # 0.7 >= 阈值
        a._snip_stale_results_openai()
        contents = [m["content"] for m in a._openai_messages[1:]]
        self.assertEqual(contents[:2], [SNIP_PLACEHOLDER, SNIP_PLACEHOLDER])
        self.assertEqual(contents[2:], ["result-2", "result-3", "result-4"])

    def test_anthropic_dedups_repeated_read_file_of_same_path(self):
        a = _make_agent()
        a.effective_window = 1000
        a.last_input_token_count = 700
        files = ["x.py", "a.py", "b.py", "a.py"]  # a.py 被读两次
        a._anthropic_messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"t{i}", "name": "read_file",
                     "input": {"file_path": f}}
                    for i, f in enumerate(files)
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"t{i}", "content": f"content-{i}"}
                    for i in range(len(files))
                ],
            },
        ]
        a._snip_stale_results_anthropic()
        blocks = a._anthropic_messages[1]["content"]
        # index 0：超出"保留最近 3 条"窗口被修剪
        self.assertEqual(blocks[0]["content"], SNIP_PLACEHOLDER)
        # index 1：虽在最近窗口内，但 a.py 之后又被重读，旧结果应被去重修剪
        self.assertEqual(blocks[1]["content"], SNIP_PLACEHOLDER)
        # 最近的 b.py 与最后一次 a.py 保留
        self.assertEqual(blocks[2]["content"], "content-2")
        self.assertEqual(blocks[3]["content"], "content-3")


class TestMicrocompact(unittest.TestCase):
    """微压缩：仅当距上次 API 调用空闲超过 MICROCOMPACT_IDLE_S 才清理旧结果。"""

    def _openai_agent_with_tools(self, count: int) -> Agent:
        a = _make_agent(use_openai=True)
        a._openai_messages = [{"role": "system", "content": "sys"}] + [
            {"role": "tool", "tool_call_id": f"c{i}", "content": f"result-{i}"}
            for i in range(count)
        ]
        return a

    def test_noop_when_never_called_api(self):
        a = self._openai_agent_with_tools(5)
        a.last_api_call_time = 0
        a._microcompact_openai()
        self.assertEqual(a._openai_messages[1]["content"], "result-0")

    def test_noop_when_recently_active(self):
        a = self._openai_agent_with_tools(5)
        a.last_api_call_time = time.time()  # 刚刚调用过
        a._microcompact_openai()
        self.assertEqual(a._openai_messages[1]["content"], "result-0")

    def test_clears_old_results_after_idle(self):
        a = self._openai_agent_with_tools(5)
        a.last_api_call_time = time.time() - MICROCOMPACT_IDLE_S - 10
        a._microcompact_openai()
        contents = [m["content"] for m in a._openai_messages[1:]]
        self.assertEqual(contents[:2], ["[Old result cleared]", "[Old result cleared]"])
        self.assertEqual(contents[2:], ["result-2", "result-3", "result-4"])
        self.assertEqual(len(contents) - 2, KEEP_RECENT_RESULTS)


class TestCompactConversation(unittest.TestCase):
    """摘要压缩：消息过短不触发；压缩后历史折叠为 [摘要+确认+末条用户消息]，
    且 last_input_token_count 必须归零（为下一轮压缩触发判断提供干净基准）。"""

    @staticmethod
    def _summary_content_mock():
        return MagicMock(content=[MagicMock(type="text", text="SUMMARY")])

    def test_anthropic_noop_below_four_messages(self):
        a = _make_agent()
        a._anthropic_messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        a._anthropic_client = MagicMock()
        a._anthropic_client.messages.create = AsyncMock()
        asyncio.run(a._compact_anthropic())
        a._anthropic_client.messages.create.assert_not_awaited()
        self.assertEqual(len(a._anthropic_messages), 2)

    def test_anthropic_compacts_and_resets_token_counter(self):
        a = _make_agent()
        a.last_input_token_count = 12345
        a._anthropic_messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "world"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "what now?"},
        ]
        a._anthropic_client = MagicMock()
        a._anthropic_client.messages.create = AsyncMock(return_value=self._summary_content_mock())
        asyncio.run(a._compact_anthropic())
        # 历史折叠为 [摘要 + 确认 + 最后一条用户消息]
        self.assertEqual(a._anthropic_messages[0]["content"], "[Previous conversation summary]\nSUMMARY")
        self.assertEqual(a._anthropic_messages[1]["role"], "assistant")
        self.assertEqual(a._anthropic_messages[-1]["content"], "what now?")
        self.assertEqual(len(a._anthropic_messages), 3)
        # 计数器归零：回归 last_input_tokens 拼写笔误
        self.assertEqual(a.last_input_token_count, 0)

    def test_openai_compacts_and_resets_token_counter(self):
        a = _make_agent(use_openai=True)
        a.last_input_token_count = 9999
        a._openai_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "what now?"},
        ]
        a._openai_client = MagicMock()
        # 必须走 chat.completions.create（Chat Completions）；legacy completions.create 不接受 messages 参数
        a._openai_client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="SUMMARY"))]
        ))
        a._openai_client.completions.create = AsyncMock()
        asyncio.run(a._compact_openai())
        a._openai_client.chat.completions.create.assert_awaited_once()
        a._openai_client.completions.create.assert_not_awaited()
        self.assertEqual(a._openai_messages[0]["content"], "sys")
        self.assertEqual(a._openai_messages[1]["content"], "[Previous conversation summary]\nSUMMARY")
        self.assertEqual(a._openai_messages[-1]["content"], "what now?")
        self.assertEqual(a.last_input_token_count, 0)

    def test_openai_summary_without_content_falls_back_to_empty(self):
        # 响应缺失 message.content（如 choices 为空或 content 为 None）时不得崩溃
        a = _make_agent(use_openai=True)
        a._openai_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        a._openai_client = MagicMock()
        for label, resp in (
            ("empty-choices", MagicMock(choices=[])),
            ("none-content", MagicMock(choices=[MagicMock(message=MagicMock(content=None))])),
        ):
            with self.subTest(label=label):
                a._openai_client.chat.completions.create = AsyncMock(return_value=resp)
                asyncio.run(a._compact_openai())
                self.assertEqual(
                    a._openai_messages[1]["content"], "[Previous conversation summary]\n"
                )


class TestNormalizeAnthropicMessages(unittest.TestCase):
    """tool-call 解析：角色修正、孤儿 tool_use 丢弃、成对消息保留。"""

    def test_plain_messages_pass_through(self):
        a = _make_agent()
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        self.assertEqual(a._normalize_anthropic_messages(msgs), msgs)

    def test_user_message_with_tool_use_becomes_assistant(self):
        a = _make_agent()
        msgs = [
            {"role": "user",  # 角色错误：tool_use 只能来自 assistant
             "content": [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {}}]},
            {"role": "user",
             "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
        ]
        out = a._normalize_anthropic_messages(msgs)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["role"], "assistant")

    def test_orphan_tool_use_is_dropped(self):
        a = _make_agent()
        msgs = [
            {"role": "assistant",
             "content": [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {}}]},
            # 后面没有对应的 tool_result
            {"role": "user", "content": "unrelated text"},
        ]
        out = a._normalize_anthropic_messages(msgs)
        self.assertEqual(out, [{"role": "user", "content": "unrelated text"}])

    def test_matched_tool_use_result_pair_is_kept(self):
        a = _make_agent()
        msgs = [
            {"role": "assistant",
             "content": [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {}}]},
            {"role": "user",
             "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
        ]
        out = a._normalize_anthropic_messages(msgs)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["role"], "assistant")
        self.assertEqual(out[1]["content"][0]["tool_use_id"], "t1")


class TestFindToolUseById(unittest.TestCase):
    def test_found_returns_name_and_input(self):
        a = _make_agent()
        a._anthropic_messages = [
            {"role": "assistant",
             "content": [{"type": "tool_use", "id": "t9", "name": "read_file",
                          "input": {"file_path": "a.py"}}]},
        ]
        info = a._find_tool_use_by_id("t9")
        self.assertEqual(info["name"], "read_file")
        self.assertEqual(info["input"], {"file_path": "a.py"})

    def test_missing_returns_none(self):
        a = _make_agent()
        a._anthropic_messages = []
        self.assertIsNone(a._find_tool_use_by_id("nope"))


class TestExecuteToolCallDispatch(unittest.TestCase):
    """_execute_tool_call 的路由分支：plan 工具 / MCP / 普通工具 / skill_* 刷新提示词。"""

    def test_plan_mode_tool_routed(self):
        a = _make_agent()
        a._execute_plan_mode_tool = AsyncMock(return_value="plan-ok")
        result = asyncio.run(a._execute_tool_call("enter_plan_mode", {}))
        self.assertEqual(result, "plan-ok")
        a._execute_plan_mode_tool.assert_awaited_once_with("enter_plan_mode")

    def test_mcp_tool_routed_to_mcp_manager(self):
        a = _make_agent()
        a._mcp_manager.is_mcp_tool = MagicMock(return_value=True)
        a._mcp_manager.call_tool = AsyncMock(return_value="mcp-ok")
        result = asyncio.run(a._execute_tool_call("mcp__srv__tool", {"k": 1}))
        self.assertEqual(result, "mcp-ok")
        a._mcp_manager.call_tool.assert_awaited_once_with("mcp__srv__tool", {"k": 1})

    def test_normal_tool_routed_to_execute_tool(self):
        a = _make_agent()
        a._refresh_runtime_system_prompt = MagicMock()
        with patch.object(agent_mod, "execute_tool",
                          new=AsyncMock(return_value="file content")) as mock_exec:
            result = asyncio.run(a._execute_tool_call("read_file", {"file_path": "a.py"}))
        self.assertEqual(result, "file content")
        mock_exec.assert_awaited_once_with("read_file", {"file_path": "a.py"}, a._read_file_state)
        a._refresh_runtime_system_prompt.assert_not_called()

    def test_skill_create_success_refreshes_system_prompt(self):
        a = _make_agent()
        a._refresh_runtime_system_prompt = MagicMock()
        with patch.object(agent_mod, "execute_tool",
                          new=AsyncMock(return_value='{"ok": true}')):
            result = asyncio.run(a._execute_tool_call("skill_create", {"name": "demo"}))
        self.assertEqual(result, '{"ok": true}')
        a._refresh_runtime_system_prompt.assert_called_once()

    def test_skill_create_failure_does_not_refresh(self):
        a = _make_agent()
        a._refresh_runtime_system_prompt = MagicMock()
        with patch.object(agent_mod, "execute_tool",
                          new=AsyncMock(return_value="not-json")):
            asyncio.run(a._execute_tool_call("skill_create", {"name": "demo"}))
        a._refresh_runtime_system_prompt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
