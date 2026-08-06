# -*- coding: utf-8 -*-
"""tools.export_failure_trajectories 的回归测试：轨迹提取与过滤行为。"""

import json
import tempfile
import unittest
from pathlib import Path

from tools.export_failure_trajectories import _is_tbench_session, _trajectory, export


def _session(id_: str, outcome: str, *, messages=None, verification=None,
             task: str = "任务", start_time: str = "2026-08-01T00:00:00Z",
             message_count: int = 0) -> dict:
    return {
        "metadata": {
            "id": id_, "model": "deepseek-v4-flash",
            "startTime": start_time, "messageCount": message_count,
            "task": task, "outcome": outcome,
        },
        "anthropicMessages": messages or [],
        "verification": verification or [],
    }


class TrajectoryExtractionTest(unittest.TestCase):
    def test_text_messages_and_tool_result_backfill(self):
        # 文本 → user/assistant；tool_use + tool_result 按 tool_use_id 回填
        msgs = [
            {"role": "user", "content": "修复登录 bug"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "先看代码"},
                {"type": "tool_use", "id": "tu1", "name": "read_file",
                 "input": {"path": "a.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu1",
                 "content": [{"type": "text", "text": "def main(): pass"}]},
            ]},
        ]
        traj = _trajectory({"anthropicMessages": msgs})
        self.assertEqual([t["role"] for t in traj], ["user", "assistant", "tool"])
        self.assertEqual(traj[0]["text"], "修复登录 bug")
        self.assertEqual(traj[1]["text"], "先看代码")
        self.assertEqual(traj[2]["name"], "read_file")
        self.assertEqual(traj[2]["result"], "def main(): pass")

    def test_interrupted_tool_marked(self):
        # 无对应 tool_result 的工具调用标记为中断（失败轨迹关键场景）
        msgs = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu9", "name": "run_shell",
                 "input": {"command": "rm -rf /"}},
            ]},
        ]
        traj = _trajectory({"anthropicMessages": msgs})
        self.assertEqual(traj[0]["result"], "[no result / interrupted]")

    def test_tool_error_kept_as_result(self):
        # 工具报错文本保留在 result 中
        msgs = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu2", "name": "edit_file",
                 "input": {"path": "x.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu2",
                 "content": [{"type": "text", "text": "Error: 越界路径被拒绝"}]},
            ]},
        ]
        traj = _trajectory({"anthropicMessages": msgs})
        self.assertIn("拒绝", traj[0]["result"])


class ExportFilterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.session_dir = Path(self.tmp.name)
        fail = _session("s-fail", "fail",
                        messages=[{"role": "user", "content": "修复登录"}],
                        verification=[{"attempt": 1, "passed": False}],
                        task="修复登录 bug", message_count=3)
        ok = _session("s-pass", "pass", task="写测试", message_count=2)
        (self.session_dir / "s-fail.json").write_text(json.dumps(fail), encoding="utf-8")
        (self.session_dir / "s-pass.json").write_text(json.dumps(ok), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_only_fail(self):
        rows = export(self.session_dir)
        self.assertEqual([r["session_id"] for r in rows], ["s-fail"])
        self.assertEqual(rows[0]["outcome"], "fail")
        self.assertEqual(rows[0]["task"], "修复登录 bug")
        self.assertEqual(rows[0]["message_count"], 3)
        self.assertFalse(rows[0]["last_verification"]["passed"])

    def test_all_outcomes(self):
        rows = export(self.session_dir, all_outcomes=True)
        self.assertEqual({r["session_id"] for r in rows}, {"s-fail", "s-pass"})

    def test_task_keyword_filter(self):
        rows = export(self.session_dir, task="登录")
        self.assertEqual([r["session_id"] for r in rows], ["s-fail"])

    def test_since_filter(self):
        rows = export(self.session_dir, since="2026-08-02")
        self.assertEqual(rows, [])

    def test_corrupt_file_skipped(self):
        (self.session_dir / "broken.json").write_text("{not json", encoding="utf-8")
        rows = export(self.session_dir)
        self.assertEqual([r["session_id"] for r in rows], ["s-fail"])


class TbenchFilterTest(unittest.TestCase):
    def test_is_tbench_session(self):
        self.assertTrue(_is_tbench_session({
            "anthropicMessages": [
                {"role": "system", "content": "你在 [T-Bench 容器环境] 中"},
            ]}))
        self.assertFalse(_is_tbench_session({
            "anthropicMessages": [{"role": "user", "content": "hi"}]}))


if __name__ == "__main__":
    unittest.main()
