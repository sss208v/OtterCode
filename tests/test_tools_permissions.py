# tests/test_tools_permissions.py
# 针对 agents/tools.py 的两块安全关键路径的最小聚焦测试：
#   1. is_dangerous / DANGEROUS_PATTERNS 危险命令拦截 + HARD_BLOCKLIST 硬黑名单
#   2. check_permission 的 PermissionMode 各分支（含 workspace 路径沙箱与审计日志）
# 仅使用标准库 unittest，运行方式：python -m unittest discover -s tests

import asyncio
import json
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

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
            "curl https://x | sh",              # curl 管道给 shell
            "wget -qO- https://x | bash",       # wget 管道给 bash
            "chmod 777 file",                   # chmod 777 全权限
            "Set-ExecutionPolicy Unrestricted", # PowerShell 执行策略放宽
            "pip install package",              # 供应链安装
            "npm install",                      # 供应链安装（预期行为变更：需确认）
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
            "curl --version",   # curl 无管道，不触发远程代码执行模式
            "chmod 755 file",   # 非 777 的 chmod 不触发
        ]
        for cmd in safe:
            with self.subTest(cmd=cmd):
                self.assertFalse(tools.is_dangerous(cmd))


class TestHardBlocklist(unittest.TestCase):
    """硬黑名单：即使 bypassPermissions（--yolo）也必须拒绝。"""

    def test_bypass_blocks_hard_blocked_shell(self):
        blocked = [
            "rm -rf /",
            "rm -rf /*",
            "rm -rf /etc",
            "dd if=/dev/zero of=/dev/sda",
            "mkfs.ext4 /dev/sdb",
            "> /dev/sda",
            "Remove-Item -Recurse -Force C:\\",
            "format c:",
            "fdisk /dev/sda",
        ]
        for cmd in blocked:
            with self.subTest(cmd=cmd):
                self.assertTrue(tools.is_hard_blocked(cmd))
                result = tools.check_permission(
                    "run_shell", {"command": cmd}, mode="bypassPermissions"
                )
                self.assertEqual(result["action"], "deny")
                self.assertIn("Hard-blocked command", result["message"])

    def test_bypass_allows_safe_shell(self):
        for cmd in ["git status", "python -m unittest"]:
            with self.subTest(cmd=cmd):
                result = tools.check_permission(
                    "run_shell", {"command": cmd}, mode="bypassPermissions"
                )
                self.assertEqual(result["action"], "allow")

    def test_hard_blocklist_does_not_flag_non_root_rm(self):
        # 非根级递归删除（如 /tmp 下的文件）不属于硬黑名单，但仍是 DANGEROUS_PATTERNS 命中项
        self.assertFalse(tools.is_hard_blocked("rm -rf /tmp/x"))


class TestCheckPermission(unittest.TestCase):
    """check_permission 的 PermissionMode 分支与 allow/deny 规则优先级。"""

    def setUp(self):
        # 注入空规则缓存，避免读取真实 ~/.otter/settings.json；teardown 还原。
        # 同步更新 mtime 快照，保证热失效逻辑视注入缓存为最新。
        self._saved_rules = tools._cached_rules
        self._saved_mtime = tools._cached_rules_mtime
        tools._cached_rules = {"allow": [], "deny": []}
        tools._cached_rules_mtime = tools._settings_mtimes()

    def tearDown(self):
        tools._cached_rules = self._saved_rules
        tools._cached_rules_mtime = self._saved_mtime

    def _set_rules(self, allow=None, deny=None):
        tools._cached_rules = {"allow": allow or [], "deny": deny or []}

    # ---- bypassPermissions ----

    def test_bypass_blocks_hard_blocked_shell(self):
        # P0 加固：rm -rf / 属于硬黑名单，bypassPermissions 也必须拒绝
        result = tools.check_permission(
            "run_shell", {"command": "rm -rf /"}, mode="bypassPermissions"
        )
        self.assertEqual(result["action"], "deny")
        self.assertIn("Hard-blocked command", result["message"])

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


class TestWorkspaceSandbox(unittest.TestCase):
    """workspace 路径沙箱：越界路径在各模式下的行为。"""

    def setUp(self):
        self._saved_rules = tools._cached_rules
        self._saved_mtime = tools._cached_rules_mtime
        tools._cached_rules = {"allow": [], "deny": []}
        tools._cached_rules_mtime = tools._settings_mtimes()
        self._tmp_dir = tempfile.mkdtemp()
        self._outside = Path(self._tmp_dir) / "secret.txt"
        self._outside.write_text("secret", encoding="utf-8")

    def tearDown(self):
        tools._cached_rules = self._saved_rules
        tools._cached_rules_mtime = self._saved_mtime
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_default_mode_confirms_outside_read(self):
        result = tools.check_permission(
            "read_file", {"file_path": str(self._outside)}, mode="default"
        )
        self.assertEqual(result["action"], "confirm")
        self.assertIn("secret.txt", result["message"])

    def test_dont_ask_mode_denies_outside_read(self):
        result = tools.check_permission(
            "read_file", {"file_path": str(self._outside)}, mode="dontAsk"
        )
        self.assertEqual(result["action"], "deny")
        self.assertIn("Path outside workspace", result["message"])

    def test_accept_edits_mode_denies_outside_write(self):
        # 写入目标即使尚不存在也必须检查（越界位置会被创建文件）
        outside_write = Path(self._tmp_dir) / "new.txt"
        result = tools.check_permission(
            "write_file", {"file_path": str(outside_write)}, mode="acceptEdits"
        )
        self.assertEqual(result["action"], "deny")
        self.assertIn("Path outside workspace", result["message"])

    def test_bypass_allows_outside_read(self):
        result = tools.check_permission(
            "read_file", {"file_path": str(self._outside)}, mode="bypassPermissions"
        )
        self.assertEqual(result["action"], "allow")

    def test_default_allows_inside_read(self):
        result = tools.check_permission("read_file", {"file_path": "a.py"}, mode="default")
        self.assertEqual(result["action"], "allow")


class TestPlanAllowsVerification(unittest.TestCase):
    """Task 13.4：run_verification 是只读验证，plan 模式放行。"""

    def setUp(self):
        self._saved_rules = tools._cached_rules
        self._saved_mtime = tools._cached_rules_mtime
        tools._cached_rules = {"allow": [], "deny": []}
        tools._cached_rules_mtime = tools._settings_mtimes()

    def tearDown(self):
        tools._cached_rules = self._saved_rules
        tools._cached_rules_mtime = self._saved_mtime

    def test_plan_mode_allows_run_verification(self):
        result = tools.check_permission("run_verification", {}, mode="plan")
        self.assertEqual(result["action"], "allow")


class TestPermissionAuditLog(unittest.TestCase):
    """权限决策审计日志：每次决策写入 .otter/logs/permissions.log（测试重定向到临时文件）。"""

    def test_decision_is_logged_to_file(self):
        tools.reset_permission_logger()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                log_file = Path(tmp) / "permissions.log"
                original_fn = tools._permissions_log_path
                tools._permissions_log_path = lambda: log_file
                try:
                    tools.check_permission("read_file", {"file_path": "a.py"}, mode="default")
                    tools.check_permission(
                        "run_shell", {"command": "git push origin main"}, mode="default"
                    )
                    self.assertTrue(log_file.exists())
                    content = log_file.read_text(encoding="utf-8")
                    self.assertIn("action=allow", content)
                    self.assertIn("action=confirm", content)
                    self.assertIn("mode=default", content)
                    self.assertIn("tool=run_shell", content)
                finally:
                    tools._permissions_log_path = original_fn
                    # 目录清理前先关闭日志 handler，释放文件句柄（Windows 锁定文件）
                    tools.reset_permission_logger()
        finally:
            tools.reset_permission_logger()


class TestReadFileRobustness(unittest.TestCase):
    """read_file 健壮性：超大文件拒绝整读、二进制文件提示（Task 18 后返回结构化 dict）。"""

    def test_oversized_file_returns_too_large_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "big.txt"
            p.write_bytes(b"x" * 200)
            old_limit = tools.MAX_FILE_READ_BYTES
            tools.MAX_FILE_READ_BYTES = 100  # monkeypatch 缩小上限，避免真写 10MB
            try:
                result = tools._read_file({"file_path": str(p)})
            finally:
                tools.MAX_FILE_READ_BYTES = old_limit
            self.assertIsNotNone(result["error"])
            self.assertIn("File too large to read", result["error"])
            self.assertIn("limit 10MB", result["error"])

    def test_binary_file_returns_binary_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bin.dat"
            p.write_bytes(b"\x00\x01\x02")
            result = tools._read_file({"file_path": str(p)})
            self.assertIsNotNone(result["error"])
            self.assertIn("Binary file detected", result["error"])

    def test_text_file_reads_normally(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "hello.txt"
            p.write_text("hello\nworld", encoding="utf-8")
            result = tools._read_file({"file_path": str(p)})
            self.assertIsNone(result["error"])
            self.assertIn("1 | hello", result["content"])
            self.assertIn("2 | world", result["content"])


class TestExecuteToolStructuredResult(unittest.TestCase):
    """Task 18：execute_tool 返回结构化 dict {"content", "error", "retryable"}。"""

    def setUp(self):
        self._saved_rules = tools._cached_rules
        self._saved_mtime = tools._cached_rules_mtime
        tools._cached_rules = {"allow": [], "deny": []}
        tools._cached_rules_mtime = tools._settings_mtimes()

    def tearDown(self):
        tools._cached_rules = self._saved_rules
        tools._cached_rules_mtime = self._saved_mtime

    def test_run_shell_timeout_is_retryable(self):
        # subprocess.run 抛 TimeoutExpired → transient 错误，retryable=True
        with patch.object(
            tools.subprocess, "run",
            side_effect=subprocess.TimeoutExpired("cmd", timeout=1),
        ):
            result = asyncio.run(
                tools.execute_tool("run_shell", {"command": "sleep 5", "timeout": 100})
            )
        self.assertIsInstance(result, dict)
        self.assertTrue(result["retryable"])
        self.assertIsNotNone(result["error"])
        self.assertIn("timed out", result["error"])

    def test_edit_file_missing_old_string_not_retryable(self):
        # old_string not found 是确定性错误：retryable=False，error 含提示
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.txt"
            p.write_text("hello", encoding="utf-8")
            result = asyncio.run(
                tools.execute_tool(
                    "edit_file",
                    {"file_path": str(p), "old_string": "not-exist", "new_string": "x"},
                )
            )
        self.assertIsInstance(result, dict)
        self.assertFalse(result["retryable"])
        self.assertIsNotNone(result["error"])
        self.assertIn("old_string not found", result["error"])
        self.assertEqual(result["content"], "")

    def test_write_file_success_path(self):
        # 成功路径：error=None、retryable=False、content 含成功文本
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.txt"
            result = asyncio.run(
                tools.execute_tool("write_file", {"file_path": str(p), "content": "abc"})
            )
        self.assertIsInstance(result, dict)
        self.assertIsNone(result["error"])
        self.assertFalse(result["retryable"])
        self.assertIn("Successfully wrote", result["content"])

    def test_unknown_tool_is_non_retryable_error(self):
        result = asyncio.run(tools.execute_tool("no_such_tool", {}))
        self.assertIsNotNone(result["error"])
        self.assertFalse(result["retryable"])

    def test_tool_search_no_match_is_content_not_error(self):
        # 无匹配返回 content 提示（非错误）
        result = asyncio.run(tools.execute_tool("tool_search", {"query": "zzz-no-such"}))
        self.assertIsNone(result["error"])
        self.assertIn("No matching deferred tools", result["content"])


class TestPermissionRulesHotReload(unittest.TestCase):
    """Task 19：settings.json 规则热更新——mtime 变化后重新加载，未变化走缓存。"""

    def tearDown(self):
        tools.reset_permission_cache()

    def test_hot_reload_on_settings_mtime_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            proj = root / "proj"
            (home / ".otter").mkdir(parents=True)
            (proj / ".otter").mkdir(parents=True)
            user_settings = home / ".otter" / "settings.json"
            user_settings.write_text(
                json.dumps({"permissions": {"deny": ["read_file"]}}), encoding="utf-8"
            )

            with patch.object(tools.Path, "home", classmethod(lambda cls: home)), \
                 patch.object(tools.Path, "cwd", classmethod(lambda cls: proj)), \
                 patch.object(tools, "_load_settings", wraps=tools._load_settings) as mock_load:
                # 第一次加载：读到 deny read_file（_load_settings 调用 2 次：user+project）
                rules1 = tools.load_permission_rules()
                self.assertEqual([r["tool"] for r in rules1["deny"]], ["read_file"])
                self.assertEqual(mock_load.call_count, 2)

                # 未修改：走缓存，_load_settings 不再被调用
                rules2 = tools.load_permission_rules()
                self.assertIs(rules1, rules2)
                self.assertEqual(mock_load.call_count, 2)

                # 修改 settings.json（mtime 变化）→ 重新加载，新规则生效
                time.sleep(0.02)
                user_settings.write_text(
                    json.dumps({"permissions": {"deny": ["write_file"]}}), encoding="utf-8"
                )
                rules3 = tools.load_permission_rules()
                self.assertEqual([r["tool"] for r in rules3["deny"]], ["write_file"])
                self.assertEqual(mock_load.call_count, 4)

    def test_reset_permission_cache_forces_reload(self):
        # reset_permission_cache 仍强制清空：下一次 load 必然重新读取
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            proj = root / "proj"
            (home / ".otter").mkdir(parents=True)
            (proj / ".otter").mkdir(parents=True)
            user_settings = home / ".otter" / "settings.json"
            user_settings.write_text(
                json.dumps({"permissions": {"deny": ["read_file"]}}), encoding="utf-8"
            )

            with patch.object(tools.Path, "home", classmethod(lambda cls: home)), \
                 patch.object(tools.Path, "cwd", classmethod(lambda cls: proj)), \
                 patch.object(tools, "_load_settings", wraps=tools._load_settings) as mock_load:
                tools.load_permission_rules()
                tools.reset_permission_cache()
                tools.load_permission_rules()
                # 两次真实加载，每次读 user+project 两个文件
                self.assertEqual(mock_load.call_count, 4)


if __name__ == "__main__":
    unittest.main()
