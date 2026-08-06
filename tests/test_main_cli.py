"""CLI 参数解析回归测试。

背景：main.py 构造 Agent 时引用 args.max_duration（agent.py:178 的 max_duration_s），
但 --max-duration 参数未注册，导致 CLI 启动即崩（AttributeError: 'Namespace' object
has no attribute 'max_duration'）。本文件保证 parse_args 的字段与使用点保持一致。

注意：导入 agents.main 会连带导入 agents.agent（依赖 anthropic 等），须在 .venv 下运行。
"""
import sys
import unittest
from unittest.mock import patch

from agents.main import parse_args


class TestParseArgs(unittest.TestCase):
    def _parse(self, *argv: str):
        with patch.object(sys, "argv", ["otter-code", *argv]):
            return parse_args()

    def test_max_duration_flag_registered(self):
        # 回归：main.py:421 构造 Agent 时使用 args.max_duration，参数必须已注册且类型为 float。
        args = self._parse("--max-duration", "120.5", "任务")
        self.assertEqual(args.max_duration, 120.5)

    def test_max_duration_default_none(self):
        # 未传 --max-duration 时默认 None，Agent 侧 wall-clock 超时检查不生效（agent.py:1324）。
        args = self._parse("任务")
        self.assertIsNone(args.max_duration)

    def test_budget_flags_default_none(self):
        # 三个预算参数（成本/轮数/时长）默认值一致，避免遗漏导致 AttributeError。
        args = self._parse()
        self.assertIsNone(args.max_cost)
        self.assertIsNone(args.max_turns)
        self.assertIsNone(args.max_duration)


if __name__ == "__main__":
    unittest.main()
