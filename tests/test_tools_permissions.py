# tests/test_tools_permissions.py
# 针对 agents/tools.py 的两块安全关键路径的最小聚焦测试：
#   1. is_dangerous / DANGEROUS_PATTERNS 危险命令拦截
#   2. check_permission 的 PermissionMode 各分支
# 仅使用标准库 unittest，运行方式：python -m unittest discover -s tests

import unittest

from agents import tools


class TestIsDangerous(unittest.TestCase):
    """危险命令检测的正/负样例。"""

    def test_dangerous_commands_are_flagged(self):
        dangerous = [
            "rm -rf /tmp/x",           # rm
            "git push origin main",    # git push
            "git reset --hard HEAD~1", # git reset
            "sudo apt install curl",   # sudo
            "dd if=/dev/zero of=disk", # dd
            "kill -9 1234",            # kill
            "del C:\\temp\\a.txt",     # Windows del
            "Remove-Item foo.txt",     # PowerShell Remove-Item
            "taskkill /F /PID 1234",   # Windows taskkill
        ]
        for cmd in dangerous:
            with self.subTest(cmd=cmd):
                self.assertTrue(tools.is_dangerous(cmd))

    def test_safe_commands_are_not_flagged(self):
        safe = [
            "ls -la",
            "git status",
            "git log --oneline",
            "python -m unittest discover -s tests",
            "echo hello",
            "npm install",  # 'rm' 出现在 npm 内部，不应误报
        ]
        for cmd in safe:
            with self.subTest(cmd=cmd):
                self.assertFalse(tools.is_dangerous(cmd))


class TestCheckPermission(unittest.TestCase):
    """check_permission 的 PermissionMode 分支与 allow/deny 规则优先级。"""

    def setUp(self):
        # 注入空规则缓存，避免读取真实 ~/.otter/settings.json；teardown 还原。
        self._saved_rules = tools._cached_rules
        tools._cached_rules = {"allow": [], "deny": []}

    def tearDown(self):
        tools._cached_rules = self._saved_rules

    def _set_rules(self, allow=None, deny=None):
        tools._cached_rules = {"allow": allow or [], "deny": deny or []}

    # ---- bypassPermissions ----

    def test_bypass_allows_dangerous_shell(self):
        result = tools.check_permission(
            "run_shell", {"command": "rm -rf /"}, mode="bypassPermissions"
        )
        self.assertEqual(result["action"], "allow")

    # ---- default ----

    def test_default_allows_read_tools(self):
        result = tools.check_permission("read_file", {"file_path": "a.py"}, mode="default")
        self.assertEqual(result["action"], "allow")

    def test_default_allows_safe_shell(self):
        result = tools.check_permission("run_shell", {"command": "git status"}, mode="default")
        self.assertEqual(result["action"], "allow")

    def test_default_confirms_dangerous_shell(self):
        result = tools.check_permission(
            "run_shell", {"command": "git push origin main"}, mode="default"
        )
        self.assertEqual(result["action"], "confirm")
        self.assertEqual(result["message"], "git push origin main")

    # ---- plan ----

    def test_plan_denies_edit_tools(self):
        result = tools.check_permission("write_file", {"file_path": "a.py"}, mode="plan")
        self.assertEqual(result["action"], "deny")

    def test_plan_allows_edit_of_plan_file(self):
        result = tools.check_permission(
            "write_file",
            {"file_path": "plan.md"},
            mode="plan",
            plan_file_path="plan.md",
        )
        self.assertEqual(result["action"], "allow")

    def test_plan_denies_shell(self):
        result = tools.check_permission("run_shell", {"command": "ls"}, mode="plan")
        self.assertEqual(result["action"], "deny")

    def test_plan_allows_read_tools(self):
        result = tools.check_permission("read_file", {"file_path": "a.py"}, mode="plan")
        self.assertEqual(result["action"], "allow")

    # ---- acceptEdits ----

    def test_accept_edits_allows_edit_tools(self):
        result = tools.check_permission(
            "write_file", {"file_path": "does_not_exist_xyz.py"}, mode="acceptEdits"
        )
        self.assertEqual(result["action"], "allow")

    # ---- dontAsk ----

    def test_dont_ask_auto_denies_dangerous_shell(self):
        result = tools.check_permission(
            "run_shell", {"command": "rm -rf /tmp/x"}, mode="dontAsk"
        )
        self.assertEqual(result["action"], "deny")
        self.assertIn("dontAsk", result["message"])

    def test_dont_ask_allows_safe_shell(self):
        result = tools.check_permission("run_shell", {"command": "ls"}, mode="dontAsk")
        self.assertEqual(result["action"], "allow")

    # ---- allow/deny 规则优先级 ----

    def test_deny_rule_blocks_read_tool(self):
        self._set_rules(deny=[{"tool": "read_file", "pattern": None}])
        result = tools.check_permission("read_file", {"file_path": "a.py"}, mode="default")
        self.assertEqual(result["action"], "deny")

    def test_allow_rule_skips_dangerous_confirm(self):
        self._set_rules(allow=[{"tool": "run_shell", "pattern": "git push*"}])
        result = tools.check_permission(
            "run_shell", {"command": "git push origin main"}, mode="default"
        )
        self.assertEqual(result["action"], "allow")

    def test_deny_rule_wins_over_allow_rule(self):
        self._set_rules(
            allow=[{"tool": "run_shell", "pattern": None}],
            deny=[{"tool": "run_shell", "pattern": None}],
        )
        result = tools.check_permission("run_shell", {"command": "ls"}, mode="default")
        self.assertEqual(result["action"], "deny")


if __name__ == "__main__":
    unittest.main()
