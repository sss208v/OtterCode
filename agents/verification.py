#!/usr/bin/env python3
"""三层验证架构：产物存在性(L1) → 产物正确性(L2) → 业务状态(L3)。

设计依据（来自外部调研，详见 fix 需求文档）：
- 声明式规则前置声明：验收标准在任务开始前冻结（Proof Loop 的 AC / SWE-bench
  的测试集），验证不是事后自评而是事前承诺。
- 程序化检查器优先：LLM-as-judge 存在位置偏见、自增强偏见等系统性偏差；
  程序化检查（文件存在/内容匹配/命令退出码）可审计、零模型成本。
- 失败回传修复（fix loop / evaluator-optimizer）：验证失败信息注入对话，
  模型修复后重新验证，限次防止死循环。

本模块只依赖标准库，符合 AGENTS.md"验证命令只依赖标准库"的约束。
"""

from __future__ import annotations

import glob as _glob
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

# 项目级验证规则配置文件（相对仓库根）
DEFAULT_CONFIG_PATH = Path(".otter") / "verification.json"

# 验证最多执行的轮数（含首轮）；超过后放行并标记未通过。
# 运行时请使用 get_max_verification_attempts()（支持 OTTER_VERIFY_MAX_ATTEMPTS 环境变量覆盖）。
MAX_VERIFICATION_ATTEMPTS = 3


def get_max_verification_attempts() -> int:
    """读取验证最大轮数：环境变量 OTTER_VERIFY_MAX_ATTEMPTS（默认 3；需为 >=1 的整数，非法值回落 3）。

    每轮验证时调用以获取最新值，避免 import 常量后固定死。
    """
    raw = os.environ.get("OTTER_VERIFY_MAX_ATTEMPTS", "3").strip()
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return MAX_VERIFICATION_ATTEMPTS
    return n if n >= 1 else MAX_VERIFICATION_ATTEMPTS

VALID_LEVELS = (1, 2, 3)
VALID_TYPES = (
    "file_exists",     # L1: 目标文件/路径存在
    "glob_exists",     # L1: glob 模式至少匹配一个文件
    "dir_nonempty",    # L1: 目录存在且非空
    "file_contains",   # L2: 文件内容包含指定文本（支持正则前缀 re:）
    "command_success", # L2/L3: 命令退出码为 0（编译/测试/业务状态断言）
    "llm_judge",       # L3: 模型语义判定（预留，默认不启用——偏见风险高）
)


class VerificationRule:
    """单条验证规则。字段全部可序列化，便于写入会话存档。"""

    __slots__ = ("id", "level", "type", "target", "pattern", "command",
                 "timeout", "severity", "description")

    def __init__(self, raw: dict[str, Any]):
        self.id = str(raw.get("id", "")).strip()
        self.level = int(raw.get("level", 2))
        self.type = str(raw.get("type", "")).strip()
        self.target = str(raw.get("target", "")).strip()
        self.pattern = str(raw.get("pattern", ""))
        self.command = str(raw.get("command", "")).strip()
        self.timeout = int(raw.get("timeout", 60))
        self.severity = str(raw.get("severity", "error")).strip() or "error"
        self.description = str(raw.get("description", "")).strip()

    @property
    def valid(self) -> bool:
        return bool(self.id) and self.level in VALID_LEVELS and self.type in VALID_TYPES

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "level": self.level, "type": self.type,
            "target": self.target, "pattern": self.pattern,
            "command": self.command, "timeout": self.timeout,
            "severity": self.severity, "description": self.description,
        }


def _parse_rule(raw: dict[str, Any]) -> VerificationRule | None:
    rule = VerificationRule(raw)
    return rule if rule.valid else None


def load_verification_rules(
    config_path: Path | str | None = None,
) -> list[VerificationRule]:
    """从配置文件加载规则。配置文件缺失/损坏时返回空列表（零侵入）。"""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))  # utf-8-sig: 兼容 Windows 记事本/PowerShell 的 BOM
    except Exception:
        return []
    rules = []
    for raw in data.get("rules", []):
        rule = _parse_rule(raw)
        if rule:
            rules.append(rule)
    return rules


def collect_written_file_rules(
    written_files: list[str] | set[str],
    root: Path | str | None = None,
) -> list[VerificationRule]:
    """自动收集 L1 规则：本轮写过的文件必须真实存在（防"声称写了却没写"）。"""
    root_path = Path(root) if root else Path.cwd()
    rules = []
    for f in written_files:
        p = Path(f)
        if not p.is_absolute():
            p = root_path / p
        target = p.relative_to(root_path) if p.is_relative_to(root_path) else p
        rules.append(VerificationRule({
            "id": f"auto-exists:{target}",
            "level": 1,
            "type": "file_exists",
            "target": str(target),
            "description": f"本会话声明写入的产物必须存在: {target}",
        }))
    return rules


def _run_command(command: str, cwd: Path, timeout: int) -> dict[str, str]:
    try:
        proc = subprocess.run(
            command, shell=True, cwd=str(cwd), timeout=timeout,
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            return {"status": "pass", "detail": "exit 0"}
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {"status": "fail", "detail": f"exit {proc.returncode}: {'; '.join(tail[:3])}"}
    except subprocess.TimeoutExpired:
        return {"status": "fail", "detail": f"timeout after {timeout}s"}
    except Exception as e:
        return {"status": "fail", "detail": f"{type(e).__name__}: {e}"}


def check_rule(rule: VerificationRule, cwd: Path) -> dict[str, str]:
    """执行单条规则，返回 {"status": pass|fail|skip, "detail": str}。"""
    if not rule.valid:
        return {"status": "skip", "detail": f"invalid rule: {rule.id or '<no-id>'}"}

    if rule.type == "file_exists":
        p = Path(rule.target)
        if not p.is_absolute():
            p = cwd / p
        return {"status": "pass" if p.exists() else "fail",
                "detail": "exists" if p.exists() else f"not found: {rule.target}"}

    if rule.type == "glob_exists":
        matches = _glob.glob(str(cwd / rule.target)) if rule.target else []
        return {"status": "pass" if matches else "fail",
                "detail": f"{len(matches)} match(es)" if matches else f"no match: {rule.target}"}

    if rule.type == "dir_nonempty":
        p = Path(rule.target)
        if not p.is_absolute():
            p = cwd / p
        count = len(list(p.iterdir())) if p.is_dir() else 0
        return {"status": "pass" if count > 0 else "fail",
                "detail": f"{count} entries" if p.is_dir() else f"not a dir: {rule.target}"}

    if rule.type == "file_contains":
        p = Path(rule.target)
        if not p.is_absolute():
            p = cwd / p
        if not p.is_file():
            return {"status": "fail", "detail": f"file missing: {rule.target}"}
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return {"status": "fail", "detail": f"unreadable: {e}"}
        if rule.pattern.startswith("re:"):
            matched = re.search(rule.pattern[3:], content) is not None
        else:
            matched = rule.pattern in content
        return {"status": "pass" if matched else "fail",
                "detail": f"pattern found" if matched else f"pattern not found in {rule.target}"}

    if rule.type == "command_success":
        if not rule.command:
            return {"status": "skip", "detail": "empty command"}
        return _run_command(rule.command, cwd, rule.timeout)

    if rule.type == "llm_judge":
        # 预留类型：LLM 判定存在偏见与幻觉验证风险，默认不启用。
        return {"status": "skip", "detail": "llm_judge not enabled (programmatic verification preferred)"}

    return {"status": "skip", "detail": f"unsupported type: {rule.type}"}


def run_verification(
    rules: list[VerificationRule],
    cwd: Path | str | None = None,
) -> dict[str, Any]:
    """运行全部规则，返回结构化报告：
    {"passed", "total", "failures": [{id, level, type, severity, description, detail}],
     "results": [{id, level, type, status, detail}]}
    """
    root = Path(cwd) if cwd else Path.cwd()
    results = []
    failures = []
    for rule in rules:
        res = check_rule(rule, root)
        entry = {
            "id": rule.id, "level": rule.level, "type": rule.type,
            "status": res["status"], "detail": res["detail"],
        }
        results.append(entry)
        if res["status"] == "fail":
            failures.append({
                "id": rule.id, "level": rule.level, "type": rule.type,
                "severity": rule.severity, "description": rule.description,
                "detail": res["detail"],
            })
    return {
        "passed": len(failures) == 0,
        "total": len(results),
        "failures": failures,
        "results": results,
    }


def format_verification_feedback(
    report: dict[str, Any],
    attempt: int,
    max_attempts: int,
) -> str:
    """把失败报告格式化为回传给模型的修复指令（fix loop 的反馈载体）。

    报告开头附带"已通过规则摘要"（Passed rules: ...），避免模型重复已完成的工作；
    失败列表保持不变。
    """
    passed_ids = [str(r.get("id")) for r in report.get("results", []) if r.get("status") == "pass"]
    lines = [
        "[Verification Report] Task is NOT verified yet.",
        f"Attempt {attempt}/{max_attempts}. Fix these issues and re-complete the task:",
    ]
    if passed_ids:
        lines.insert(1, f"Passed rules: {', '.join(passed_ids)}")
    for f in report.get("failures", []):
        lines.append(f"- [L{f.get('level', '?')}] {f.get('id', '?')}"
                     f" ({f.get('type', '?')}): {f.get('detail', '')}"
                     f"{' — ' + f.get('description', '') if f.get('description') else ''}")
    lines.append("After fixing, verify again and respond to the user.")
    return "\n".join(lines)
