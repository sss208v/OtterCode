# tests/test_plan_mode_and_deferred.py
# 针对 agent.py 的 plan 模式退出路径与 tools.py 的 deferred 工具范围的回归测试：
#   1. exit_plan_mode 审批分支必须真正切换 permission_mode。曾因只更新打印消息与
#      无效赋值（_pre_plan_mode = target_mode 后立即清空），导致批准后仍停留在
#      plan 模式，编辑类工具与 run_shell 全被拒绝。
#   2. deferred 工具范围：当前仅 enter_plan_mode / exit_plan_mode 声明为 deferred。
# 仅使用标准库 unittest，运行方式：python -m unittest discover -s tests

import asyncio
import unittest

from agents import tools
from agents.agent import Agent


def _make_agent() -> Agent:
    """use_openai 路径构造实例；plan 模式工具执行不涉及模型调用，无网络副作用。"""
    return Agent(
        model="deepseek-chat",
        api_key="test-key",
        api_base="http://localhost:9/v1",
        custom_system_prompt="test prompt",
    )


class TestExitPlanModePermissionSwitch(unittest.TestCase):
    """exit_plan_mode 审批分支：批准后 permission_mode 必须从 plan 切换走。"""

    def _enter_plan(self) -> Agent:
        a = _make_agent()
        asyncio.run(a._execute_plan_mode_tool("enter_plan_mode"))
        self.assertEqual(a.permission_mode, "plan")
        return a

    def test_approve_execute_switches_to_accept_edits(self):
        a = self._enter_plan()
        a._plan_approval_fn = lambda content: {"choice": "execute"}
        asyncio.run(a._execute_plan_mode_tool("exit_plan_mode"))
        self.assertEqual(a.permission_mode, "acceptEdits")

    def test_approve_clear_and_execute_switches_to_accept_edits(self):
        a = self._enter_plan()
        a._plan_approval_fn = lambda content: {"choice": "clear-and-execute"}
        asyncio.run(a._execute_plan_mode_tool("exit_plan_mode"))
        self.assertEqual(a.permission_mode, "acceptEdits")

    def test_manual_execute_restores_pre_plan_mode(self):
        a = self._enter_plan()
        a._plan_approval_fn = lambda content: {"choice": "manual-execute"}
        asyncio.run(a._execute_plan_mode_tool("exit_plan_mode"))
        self.assertEqual(a.permission_mode, "default")

    def test_keep_planning_stays_in_plan_mode(self):
        a = self._enter_plan()
        a._plan_approval_fn = lambda content: {"choice": "keep-planning", "feedback": "revise"}
        asyncio.run(a._execute_plan_mode_tool("exit_plan_mode"))
        self.assertEqual(a.permission_mode, "plan")

    def test_no_approval_fn_restores_pre_plan_mode(self):
        a = self._enter_plan()
        asyncio.run(a._execute_plan_mode_tool("exit_plan_mode"))
        self.assertEqual(a.permission_mode, "default")


class TestDeferredToolsScope(unittest.TestCase):
    """deferred 工具范围：当前仅 enter_plan_mode / exit_plan_mode 声明为 deferred。"""

    def test_only_plan_mode_tools_are_deferred(self):
        deferred = [t["name"] for t in tools.tool_definitions if t.get("deferred")]
        self.assertEqual(deferred, ["enter_plan_mode", "exit_plan_mode"])

    def test_active_definitions_exclude_deferred_by_default(self):
        active = tools.get_active_tool_definitions()
        names = [t["name"] for t in active]
        self.assertNotIn("enter_plan_mode", names)
        self.assertNotIn("exit_plan_mode", names)


if __name__ == "__main__":
    unittest.main()
