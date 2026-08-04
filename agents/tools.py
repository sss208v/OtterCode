#!/usr/bin/env python3
from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
import subprocess
from pathlib import Path

from agents.memory import get_memory_dir, save_memory_structured, update_memory_index

# Windows 平台判断只依赖标准库，保证 AGENTS.md 中"验证命令只依赖标准库"的断言成立
IS_WIN = os.name == "nt"

ToolDef = dict  # Anthropic tool schema dict
#权限模式
PermissionMode = str  # "default" | "plan" | "acceptEdits" | "bypassPermissions" | "dontAsk"

READ_TOOLS = {"read_file", "list_files", "grep_search", "file_stats"}
EDIT_TOOLS = {"write_file", "edit_file", "skill_evolve", "skill_create"}


#并发安全的工具可以并行运行（只读，无副作用）
CONCURRENCY_SAFE_TOOLS = {"read_file", "list_files", "grep_search", "file_stats"}


#工具定义
tool_definitions: list[ToolDef] = [
    {
        "name": "read_file",
        "description": "Read the contents of a file. Returns the file content with line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path to the file to read"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "file_stats",
        "description": "Get statistics for a file: line count, character count, and size in bytes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path to the file to inspect"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path to the file to write"},
                "content": {"type": "string", "description": "The content to write to the file"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Edit a file by replacing an exact string match with new content. The old_string must match exactly (including whitespace and indentation).",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path to the file to edit"},
                "old_string": {"type": "string", "description": "The exact string to find and replace"},
                "new_string": {"type": "string", "description": "The string to replace it with"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "list_files",
        "description": "List files matching a glob pattern. Returns matching file paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": 'Glob pattern to match files (e.g., "**/*.ts", "src/**/*")'},
                "path": {"type": "string", "description": "Base directory to search from. Defaults to current directory."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep_search",
        "description": "Search for a pattern in files. Returns matching lines with file paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The regex pattern to search for"},
                "path": {"type": "string", "description": "Directory or file to search in. Defaults to current directory."},
                "include": {"type": "string", "description": 'File glob pattern to include (e.g., "*.ts", "*.py")'},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_shell",
        "description": "Execute a shell command and return its output. Use this for running tests, installing packages, git operations, etc. Reserve this exclusively for system commands and terminal operations that require shell execution. For file operations, always use the dedicated tools instead: read_file (not cat/head/tail/sed), edit_file (not sed/awk), write_file (not echo redirection or heredoc), list_files (not find/ls), grep_search (not grep/rg). Dedicated tools allow the user to better understand and review your work.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "timeout": {"type": "number", "description": "Timeout in milliseconds (default: 30000)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "skill",
        "description": "Invoke a registered skill by name. Skills are prompt templates loaded from .otter/skills/. Returns the skill's resolved prompt to follow.",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "The name of the skill to invoke"},
                "args": {"type": "string", "description": "Optional arguments to pass to the skill"},
            },
            "required": ["skill_name"],
        },
    },
    {
        "name": "skill_evolve",
        "description": "Persist an explicit reusable user correction or workflow preference into an existing skill. Creates a version snapshot before editing the skill.",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "The registered skill name to evolve"},
                "lesson": {"type": "string", "description": "Durable reusable rule to add to the skill"},
                "rationale": {"type": "string", "description": "Why this lesson should affect future similar tasks"},
                "target": {
                    "type": "string",
                    "enum": ["active", "project", "user"],
                    "description": "Which skill file to update. Defaults to active.",
                },
            },
            "required": ["skill_name", "lesson"],
        },
    },
    {
        "name": "skill_create",
        "description": "Create a new reusable skill from explicit durable workflow guidance when no suitable existing skill exists.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Concise reusable skill name"},
                "description": {"type": "string", "description": "One-sentence description of what the skill does and when to use it"},
                "instructions": {"type": "string", "description": "Reusable SKILL.md body. Focus on durable method, constraints, and workflow, not one-off task content."},
                "when_to_use": {"type": "string", "description": "Trigger condition for auto-invocation"},
                "target": {
                    "type": "string",
                    "enum": ["project", "user"],
                    "description": "Where to create the skill. Defaults to project.",
                },
                "context": {
                    "type": "string",
                    "enum": ["inline", "fork"],
                    "description": "Skill execution mode. Defaults to inline.",
                },
                "user_invocable": {"type": "boolean", "description": "Whether users can invoke it manually with /<skill>. Defaults to false."},
                "allowed_tools": {"type": "string", "description": "Optional comma-separated allowed tools for fork mode"},
                "evidence": {"type": "string", "description": "Short user-provided evidence showing why this is reusable"},
            },
            "required": ["name", "description", "instructions"],
        },
    },
    {
        "name": "enter_plan_mode",
        "description": "Enter plan mode to switch to a read-only planning phase. In plan mode, you can only read files and write to the plan file.",
        "input_schema": {"type": "object", "properties": {}},
        "deferred": True,
    },
    {
        "name": "exit_plan_mode",
        "description": "Exit plan mode after you have finished writing your plan to the plan file.",
        "input_schema": {"type": "object", "properties": {}},
        "deferred": True,
    },
    {
        "name": "agent",
        "description": "Launch a sub-agent to handle a task autonomously. Sub-agents have isolated context and return their result. Types: 'explore' (read-only), 'plan' (read-only, structured planning), 'general' (full tools).",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Short (3-5 word) description of the sub-agent's task"},
                "prompt": {"type": "string", "description": "Detailed task instructions for the sub-agent"},
                "type": {"type": "string", "enum": ["explore", "plan", "general"], "description": "Agent type. Default: general"},
            },
            "required": ["description", "prompt"],
        },
    },
    # ─── Tool search (deferred tool loader) ─────────────────────
    {
        "name": "run_verification",
        "description": "Run the configured three-layer verification (L1 artifact existence -> L2 artifact correctness -> L3 business state) and return a structured report. Rules come from .otter/verification.json plus auto-collected files written this session. Commands executed by verification rules are restricted to those declared in the config. Use this when the task claims completion and you want to prove the deliverable actually exists, is correct, and the business state is reached.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rule_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional: only run rules with these ids"},
            },
        },
    },
    {
        "name": "tool_search",
        "description": "Search for available tools by name or keyword. Returns full schema definitions for matching deferred tools so you can use them.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Tool name or search keywords"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_save",
        "description": "Save a structured memory entry (type, name, description, content). Validates type, dedups by name, and auto-updates the MEMORY.md index. Prefer this over write_file for saving memories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Short memory name, e.g. 'user-prefers-chinese'"},
                "description": {"type": "string", "description": "One-line description of the memory"},
                "type": {"type": "string", "description": "Memory type: user|feedback|project|reference"},
                "content": {"type": "string", "description": "Memory body content"}
            },
            "required": ["name", "description", "type", "content"]
        }
    },
]





#----------------------工具失败日志----------------------------
# 工具失败写入 .otter/logs/tools.log，留下可归因的持久记录；
# 日志只做旁路观测，工具对模型仍返回可读错误字符串，契约不变。

logger = logging.getLogger("otter.tools")


def _ensure_tool_logger() -> None:
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return
    try:
        log_dir = Path.cwd() / ".otter" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_dir / "tools.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.INFO)
        logger.propagate = False
    except Exception:
        pass  # 日志设施不可用时退回默认 logging 行为，不影响工具执行


def _summarize_input(inp: dict) -> str:
    try:
        summary = json.dumps(inp, ensure_ascii=False, default=str)
    except Exception:
        summary = repr(inp)
    return summary if len(summary) <= 200 else summary[:200] + "...(truncated)"


def _log_tool_failure(tool_name: str, inp: dict, exc: BaseException, *, level: int = logging.WARNING) -> None:
    try:
        _ensure_tool_logger()
        logger.log(
            level,
            "tool=%s input=%s error=%s: %s",
            tool_name,
            _summarize_input(inp),
            type(exc).__name__,
            exc,
        )
    except Exception:
        pass  # 记录失败不能影响工具返回契约


#----------------------工具调用----------------------------

def _resolve_tool_path(raw_path: str, *, must_exist: bool = True) -> Path:
    # 安全解析：绝对路径直接返回，相对路径保持原样。
    # 不做"cwd 外 fallback 拼接"——旧实现会把不存在的 /abs/path 尝试映射到
    # cwd 子树内，从而绕过 workspace 路径沙箱；保留 must_exist 参数仅为签名兼容。
    return Path(raw_path)


# 结构化工具返回：{"content": str, "error": str|None, "retryable": bool}
# content 为成功文本（失败时可为空串）；error 为失败时的可读错误消息（成功为 None）；
# retryable 标记错误是否可重试（transient 错误为 True，确定性错误为 False）。
def _tool_result(content: str, error: str | None = None, retryable: bool = False) -> dict:
    return {"content": content, "error": error, "retryable": retryable}


#读取文件并且在读取文件的基础上添加行号
# 单次读取文件的大小上限：超过 10MB 的文件拒绝整读（防 OOM / 防误读超大文件）
MAX_FILE_READ_BYTES = 10 * 1024 * 1024

def _read_file(inp:dict) -> dict:
    try:
        path = _resolve_tool_path(inp["file_path"])
        size = path.stat().st_size
        if size > MAX_FILE_READ_BYTES:
            return _tool_result(
                "",
                error=(
                    f"Error: File too large to read ({size} bytes, limit 10MB). "
                    "Use grep_search or read specific sections."
                ),
            )
        # 二进制检测：只读文件头 8192 字节，命中 NUL 即视为二进制，
        # 不读取整个文件，避免大文件 OOM。
        with open(path, "rb") as f:
            head = f.read(8192)
        if b"\x00" in head:
            return _tool_result("", error="Error: Binary file detected. Use file_stats or grep_search instead.")
        # Windows 默认 GBK，必须显式 UTF-8，否则中文文件解码崩溃。
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")
        numbered = "\n".join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines))
        return _tool_result(numbered)
    except Exception as e:
        _log_tool_failure("read_file", inp, e)
        return _tool_result("", error=f"Error reading file: {e}")

def _write_file(inp:dict) -> dict:
    try:
        path = _resolve_tool_path(inp["file_path"], must_exist=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(inp["content"])
        if str(path).startswith(str(get_memory_dir())):
            update_memory_index()
        lines = inp["content"].split("\n")
        line_count = len(lines)
        preview = "\n".join(f"{i + 1:4d} | {l}" for i, l in enumerate(lines[:30]))
        trunc = f"\n  ... ({line_count} lines total)" if line_count > 30 else ""
        return _tool_result(f"Successfully wrote to {inp['file_path']} ({line_count} lines)\n\n{preview}{trunc}")
    except Exception as e:
        _log_tool_failure("write_file", inp, e)
        # Windows 下 PermissionError（[Errno 13]）多为文件被占用/瞬时锁，属可重试错误
        retryable = isinstance(e, PermissionError)
        return _tool_result("", error=f"Error writing file: {e}", retryable=retryable)

def _memory_save(inp: dict) -> dict:
    try:
        filename = save_memory_structured(
            name=str(inp.get("name", "")),
            description=str(inp.get("description", "")),
            type=str(inp.get("type", "")),
            content=str(inp.get("content", "")),
        )
        return _tool_result(filename)
    except ValueError as e:
        return _tool_result("", error=f"Error saving memory: {e}")
    except Exception as e:
        _log_tool_failure("memory_save", inp, e)
        return _tool_result("", error=f"Error saving memory: {e}")

#-------------------------编辑助手，符号规范化+差异化------------------------

#将各种Unicode引号字符统一转换为标准的ASCII直引号。
def _normalize_quotes(s: str) -> str:
    s = re.sub("[\u2018\u2019\u2032]", "'", s)
    s = re.sub('[\u201c\u201d\u2033]', '"', s)
    return s

#查询匹配到的字符串
def _find_actual_string(file_content: str, search_string: str) -> str | None:
    if search_string in file_content:
        return search_string
    norm_search = _normalize_quotes(search_string)
    norm_file = _normalize_quotes(file_content)
    idx = norm_file.find(norm_search)
    if idx != -1:
        return file_content[idx:idx + len(search_string)]
    return None

#生成一个简单的文本差异格式
def _generate_diff(old_content: str, old_string: str, new_string: str) -> str:
    before_change = old_content.split(old_string)[0]
    line_num = before_change.count("\n") + 1
    old_lines = old_string.split("\n")
    new_lines = new_string.split("\n")

    parts = [f"@@ -{line_num},{len(old_lines)} +{line_num},{len(new_lines)} @@"]
    for l in old_lines:
        parts.append(f"- {l}")
    for l in new_lines:
        parts.append(f"+ {l}")
    return "\n".join(parts)

#编辑文件
def _edit_file(inp: dict) -> dict:
    try:
        path = _resolve_tool_path(inp["file_path"])
        content = path.read_text(errors="replace")

        actual = _find_actual_string(content, inp["old_string"])
        if not actual:
            return _tool_result("", error=f"Error: old_string not found in {inp['file_path']}")

        occurrences = content.count(inp["old_string"])
        if occurrences > 1:
            return _tool_result("", error=f"Error: old_string found {occurrences} times in {inp['file_path']}. Must be unique.")

        new_content = content.replace(actual, inp["new_string"], 1)
        path.write_text(new_content)

        diff = _generate_diff(content, actual, inp["new_string"])

        quote_note = " (matched via quote normalization)" if actual != inp["old_string"] else ""

        return _tool_result(f"Successfully edited {inp['file_path']}{quote_note}\n\n{diff}")
    except Exception as e:
        _log_tool_failure("edit_file", inp, e)
        return _tool_result("", error=f"Error editing file: {e}")


def _list_files(inp: dict) -> dict:
    try:
        base = _resolve_tool_path(inp.get("path") or ".")
        pattern = inp["pattern"]
        files = []
        for p in base.glob(pattern):
            if p.is_file():
                rel = str(p.relative_to(base) if base != Path(".") else p)

                if "node_modules" in rel or ".git" in rel.split(os.sep):
                    continue
                files.append(rel)
                if len(files) >= 200:
                    break
        if not files:
            return _tool_result("No files found matching the pattern.")
        result = "\n".join(files[:200])

        if len(files) > 200:
            result += f"\n...and {len(files) - 200} more files are found ..."
        return _tool_result(result)
    except Exception as e:
        _log_tool_failure("list_files", inp, e)
        return _tool_result("", error=f"Error listing files: {e}")


# 文件统计：行数/字符数/字节大小。失败返回可读错误字符串，不抛异常。
def _file_stats(inp: dict) -> dict:
    try:
        p = _resolve_tool_path(inp["file_path"])
        # Windows 默认 GBK，必须显式 UTF-8，否则中文文件解码崩溃。
        content = p.read_text(encoding="utf-8", errors="replace")
        lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        return _tool_result(
            f"File: {p}\n"
            f"Lines: {lines}\n"
            f"Characters: {len(content)}\n"
            f"Size: {p.stat().st_size} bytes"
        )
    except FileNotFoundError as e:
        _log_tool_failure("file_stats", inp, e, level=logging.INFO)
        return _tool_result("", error=f"Error: File not found: {inp.get('file_path', '')}")
    except Exception as e:
        _log_tool_failure("file_stats", inp, e)
        return _tool_result("", error=f"Error: {e}")


def _grep_search(inp: dict) -> dict:
    pattern = inp["pattern"]
    path = str(_resolve_tool_path(inp.get("path") or "."))
    include = inp.get("include")

    if not IS_WIN:
        try:
            args = ["grep", "--line-number", "--color=never", "-r"]
            if include:
                args.append(f"--include={include}")
            args.extend(["--", pattern, path])

            # Windows 默认 GBK，必须显式 UTF-8，否则中文输出解码崩溃。
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=10, encoding="utf-8", errors="replace"
            )
            if result.returncode == 1:
                return _tool_result("No matches found.")
            if result.returncode == 0:
                lines = [l for l in result.stdout.split("\n") if l]
                output = "\n".join(lines[:100])
                if len(lines) >100:
                    output += f"\n... and {len(lines) - 100} more matches"
                return _tool_result(output)
        except Exception as e:
            # 系统 grep 不可用属预期回退，记 INFO 便于归因
            _log_tool_failure("grep_search", inp, e, level=logging.INFO)

    return _grep_python(pattern, path, include)


def _grep_python(pattern: str, directory: str, include: str | None) -> dict:
    try:
        regex = re.compile(pattern)
    except re.error as e:
        _log_tool_failure("grep_search", {"pattern": pattern, "path": directory}, e)
        return _tool_result("", error=f"Error: invalid grep pattern: {e}")
    include_pattern = include
    matches: list[str] = []

    def walk(d: str) -> None:
        if len(matches) >= 200:
            return
        try:
            entries = os.listdir(d)
        except Exception as e:
            _log_tool_failure("grep_search", {"pattern": pattern, "path": d}, e, level=logging.DEBUG)
            return
        for name in entries:
            if name.startswith(".") or name == "node_modules":
                continue
            full = os.path.join(d, name)
            if os.path.isdir(full):
                walk(full)
                continue
            if include_pattern and not fnmatch.fnmatch(name, include_pattern):
                continue
            try:
                # 大文件保护：超过 5MB 的文件直接跳过，避免 grep Python 回退
                # 路径把整个文件读入内存导致 OOM（大文件应走系统 grep 或定向检索）。
                if Path(full).stat().st_size > 5 * 1024 * 1024:
                    continue
                text = Path(full).read_text(errors="replace")
                for i, line in enumerate(text.split("\n")):
                    if regex.search(line):
                        matches.append(f"{full}:{i+1}:{line}")
                        if len(matches) >= 200:
                            return
            except Exception as e:
                _log_tool_failure("grep_search", {"pattern": pattern, "file": full}, e, level=logging.DEBUG)

    walk(directory)
    if not matches:
        return _tool_result("No matches found.")
    output = "\n".join(matches[:100])
    if len(matches) > 100:
        output += f"\n... and {len(matches) - 100} more matches"
    return _tool_result(output)



#--------截断过长的工具调用结果

MAX_RESULT_CHARS = 50000


def _truncate_result(result:str) -> str:
    if len(result) <= MAX_RESULT_CHARS:
        return result
    keep_each = (MAX_RESULT_CHARS - 60) // 2
    return (
        result[:keep_each]
        +f"\n\n[... truncated {len(result) - keep_each * 2} chars ...]\n\n"
        +result[-keep_each:]
    )






#----------延迟工具激活---------------------------
_activated_tools: set[str] = set()



def reset_activated_tools() -> None:
    _activated_tools.clear()

def get_active_tool_definitions(all_tools: list[ToolDef] | None = None) -> list[ToolDef]:
    """过滤并返回当前可用的工具定义列表，主要用于 API 调用前剔除尚未激活的“延迟工具”（deferred tools），并删除无关的元数据字段。"""
    tools = all_tools if all_tools is not None else tool_definitions
    return [
        {k: v for k, v in t.items() if k != "deferred"}
        for t in tools
        if not t.get("deferred") or t["name"] in _activated_tools
    ]

def get_deferred_tool_names(all_tools: list[ToolDef] | None = None) -> list[str]:
    tools = all_tools if all_tools is not None else tool_definitions
    return [t["name"] for t in tools if t.get("deferred") and t["name"] not in _activated_tools]

#执行shell命令
def _run_shell(inp: dict) -> dict:
    try:
        timeout_ms = inp.get("timeout", 30000)
        timeout_s = timeout_ms / 1000
        result = subprocess.run(
            inp["command"],
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            encoding="utf-8",  # Windows 默认 GBK，必须显式 UTF-8，否则中文输出解码崩溃。
            errors="replace"
        )
        output = result.stdout or ""
        if result.returncode != 0:
            stderr = f"\nStderr: {result.stderr}" if result.stderr else ""
            stdout = f"\nStdout: {result.stdout}" if result.stdout else ""
            return _tool_result("", error=f"Command failed (exit code {result.returncode}){stdout}{stderr}")
        return _tool_result(output or "(no output)")
    except subprocess.TimeoutExpired as e:
        _log_tool_failure("run_shell", inp, e)
        # 超时属 transient 错误：命令可能仍在执行/稍后重试可成功
        return _tool_result("", error=f"Command timed out after {inp.get('timeout', 30000)}ms", retryable=True)
    except Exception as e:
        _log_tool_failure("run_shell", inp, e)
        return _tool_result("", error=f"Error: {e}")


#危险命令检测模式列表

DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s"),
    re.compile(r"\bgit\s+(push|reset|clean|checkout\s+\.)"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s"),
    re.compile(r">\s*/dev/"),
    re.compile(r"\bkill\b"),
    re.compile(r"\bpkill\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\bdel\s", re.IGNORECASE),
    re.compile(r"\brmdir\s", re.IGNORECASE),
    re.compile(r"\bformat\s", re.IGNORECASE),
    re.compile(r"\btaskkill\s", re.IGNORECASE),
    re.compile(r"\bRemove-Item\s", re.IGNORECASE),
    re.compile(r"\bStop-Process\s", re.IGNORECASE),
    # --- 以下为 P0 安全加固新增模式（2026-08）---
    # 远程代码执行：curl/wget 输出管道直接交给 shell（sh / bash）执行
    re.compile(r"\bcurl\s+.*\|\s*(?:ba)?sh\b", re.IGNORECASE),
    re.compile(r"\bwget\s+.*\|\s*(?:ba)?sh\b", re.IGNORECASE),
    # 权限修改：chmod 777（全权限放行）、chown（改变属主）需确认
    re.compile(r"\bchmod\s+777\b"),
    re.compile(r"\bchown\s"),
    # PowerShell 执行策略放宽，属于安全边界弱化
    re.compile(r"\bSet-ExecutionPolicy\b", re.IGNORECASE),
    # Windows 持久化/系统修改：注册表写入、计划任务、WMI 远程执行
    re.compile(r"\breg\s+add\b", re.IGNORECASE),
    re.compile(r"\bschtasks\b", re.IGNORECASE),
    re.compile(r"\bwmic\b", re.IGNORECASE),
    # 供应链安装：pip/npm 安装依赖属于外部代码引入，需用户确认
    re.compile(r"\bpip\s+install\b"),
    re.compile(r"\bnpm\s+install\b"),
]
def is_dangerous(command: str) -> bool:
    return any(p.search(command) for p in DANGEROUS_PATTERNS)


# 硬黑名单：不可逆/毁灭性系统操作。即使 bypassPermissions（--yolo）也必须拒绝。
# 与 DANGEROUS_PATTERNS 的区别：DANGEROUS_PATTERNS 只是"需确认"（default 下 confirm），
# 硬黑名单是"绝对禁止"——没有合法工作流需要它们，执行即造成不可恢复的破坏。
_RM_RF_FLAGS = r"(?:[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*|[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*)"

HARD_BLOCKLIST: list[re.Pattern] = [
    # 1. Unix 根级递归删除：rm -rf / 、rm -rf /*（flags 同时含 r 和 f，目标为根）
    re.compile(rf"\brm\b\s+-{_RM_RF_FLAGS}\s+/\*?(?:\s|$)"),
    # 2. Unix 根级递归删除：rm -rf /xxx（根下第一层目录/文件，路径不含更深子路径）
    re.compile(rf"\brm\b\s+-{_RM_RF_FLAGS}\s+/(?=[^/\s]+(?:\s|$))[^/\s]+"),
    # 3. 文件系统格式化：mkfs、mkfs.ext4 等（抹除整个文件系统，不可恢复）
    re.compile(r"\bmkfs(?:\.\w+)?\b"),
    # 4. dd 直接写块设备：dd if=... of=/dev/...（越过文件系统写裸设备）
    re.compile(r"\bdd\b[^\n|;]*\bof\s*=\s*/dev/"),
    # 5. shell 重定向写入块设备：> /dev/sdX（直接破坏磁盘块）
    re.compile(r">\s*/dev/sd[a-z]"),
    # 6. Windows：Remove-Item -Recurse -Force C:\（递归强制删除系统盘根）
    re.compile(r"\bRemove-Item\b[^\n]*(?:-Recurse|-Force)[^\n]*(?:-Recurse|-Force)[^\n]*C:\\", re.IGNORECASE),
    # 7. Windows：格式化/擦除卷与磁盘
    re.compile(r"\bFormat-Volume\b", re.IGNORECASE),
    re.compile(r"\bClear-Disk\b", re.IGNORECASE),
    # 8. Windows：diskpart 磁盘分区工具、format c: 格式化系统盘
    re.compile(r"\bdiskpart\b", re.IGNORECASE),
    re.compile(r"\bformat\s+c:", re.IGNORECASE),
    # 9. fdisk 直接操作物理磁盘分区表
    re.compile(r"\bfdisk\s+/dev/sd[a-z]"),
]


def is_hard_blocked(command: str) -> bool:
    """判断命令是否命中硬黑名单：返回 True 时任何权限模式都必须拒绝执行。"""
    return any(p.search(command) for p in HARD_BLOCKLIST)

#权限规则
def _parse_rule(rule: str) -> dict:
    m = re.match(r"^([a-z_]+)\((.+)\)$", rule)
    if m:
        return {"tool": m.group(1), "pattern": m.group(2)}
    return {"tool": rule, "pattern": None}

def _load_settings(file_path: Path) -> dict | None:
    if not file_path.exists():
        return None
    try:
        return json.loads(file_path.read_text())
    except Exception as e:
        _log_tool_failure("load_settings", {"file": str(file_path)}, e)
        return None

_cached_rules: dict | None = None
# 与 _cached_rules 配套的 settings.json 文件 mtime（纳秒）快照：
# {"user": int|None, "project": int|None}，文件不存在记为 None。
# 任一文件 mtime 与快照不一致即视为配置已改动，触发重新加载。
_cached_rules_mtime: dict[str, int | None] | None = None


def _file_mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _settings_mtimes() -> dict[str, int | None]:
    return {
        "user": _file_mtime_ns(Path.home() / ".otter" / "settings.json"),
        "project": _file_mtime_ns(Path.cwd() / ".otter" / "settings.json"),
    }


def load_permission_rules() -> dict:
    global _cached_rules, _cached_rules_mtime
    current = _settings_mtimes()
    if _cached_rules is not None and _cached_rules_mtime == current:
        return _cached_rules

    allow: list[dict] = []
    deny: list[dict] = []

    user_settings = _load_settings(Path.home() / ".otter" / "settings.json")
    project_settings = _load_settings(Path.cwd() / ".otter" / "settings.json")

    for settings in [user_settings, project_settings]:
        if not settings or "permissions" not in settings:
            continue
        perms = settings["permissions"]
        for r in perms.get("allow", []):
            allow.append(_parse_rule(r))
        for r in perms.get("deny", []):
            deny.append(_parse_rule(r))

    _cached_rules = {"allow": allow, "deny": deny}
    _cached_rules_mtime = current
    return _cached_rules


def _matches_rule(rule: dict, tool_name: str, inp: dict) -> bool:
    if rule["tool"] != tool_name:
        return False
    if rule["pattern"] is None:
        return True

    value = ""
    if tool_name == "run_shell":
        value = inp.get("command", "")
    elif "file_path" in inp:
        value = inp["file_path"]
    else:
        return True

    pattern = rule["pattern"]
    if pattern.endswith("*"):
        return value.startswith(pattern[:-1])
    return value == pattern


def _check_permission_rules(tool_name: str, inp: dict) -> str | None:
    rules = load_permission_rules()
    for rule in rules["deny"]:
        if _matches_rule(rule, tool_name, inp):
            return "deny"
    for rule in rules["allow"]:
        if _matches_rule(rule, tool_name, inp):
            return "allow"
    return None


#----------------------权限决策审计日志----------------------------
# 每次权限决策（allow/deny/confirm）写入 .otter/logs/permissions.log，
# 留下可归因的持久记录；日志只做旁路观测，不改变 check_permission 返回契约。

permission_logger = logging.getLogger("otter.permissions")


def _permissions_log_path() -> Path:
    return Path.cwd() / ".otter" / "logs" / "permissions.log"


def _ensure_permission_logger() -> None:
    if any(isinstance(h, logging.FileHandler) for h in permission_logger.handlers):
        return
    try:
        log_path = _permissions_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        permission_logger.addHandler(handler)
        if permission_logger.level == logging.NOTSET:
            permission_logger.setLevel(logging.INFO)
        permission_logger.propagate = False
    except Exception:
        pass  # 审计日志设施不可用时退回默认行为，不影响权限决策


def reset_permission_logger() -> None:
    """移除已有 handler 并重置 logger 级别，用于测试隔离。"""
    for h in list(permission_logger.handlers):
        permission_logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    permission_logger.setLevel(logging.NOTSET)


def _log_permission_decision(tool_name: str, inp: dict, action: str, mode: str, message: str = "") -> None:
    try:
        _ensure_permission_logger()
        permission_logger.info(
            "tool=%s action=%s mode=%s input=%s message=%s",
            tool_name,
            action,
            mode,
            _summarize_input(inp),
            message,
        )
    except Exception:
        pass  # 审计日志失败不能影响权限决策


#----------------------workspace 路径沙箱----------------------------

def _is_within_workspace(path: Path) -> bool:
    """判断 path.resolve() 是否位于 workspace（Path.cwd()）子树内。

    用 os.path.commonpath + os.path.normcase 实现，Windows 下路径大小写不敏感；
    不同盘符（如 C: 与 D:）时 commonpath 抛 ValueError，视为越界。
    """
    try:
        cwd = os.path.normcase(Path.cwd().resolve())
        resolved = os.path.normcase(path.resolve())
        return os.path.commonpath([resolved, cwd]) == cwd
    except ValueError:
        return False


def _check_workspace_path(tool_name: str, inp: dict) -> str | None:
    """解析工具的路径参数：越界返回绝对路径字符串，否则返回 None。

    - read_file/write_file/edit_file/file_stats 取 file_path；
    - list_files/grep_search 取 path，缺省时不做检查（默认 "." 在 cwd 内）；
    - 读取类工具只检查"确实存在"的越界路径（不存在的越界路径读取本身会失败，跳过避免误伤）；
    - 写入路径即使尚不存在也检查（write_file 会在越界位置创建文件）。
    """
    raw = None
    if tool_name in ("read_file", "write_file", "edit_file", "file_stats"):
        raw = inp.get("file_path")
    elif tool_name in ("list_files", "grep_search"):
        raw = inp.get("path")
    if not raw:
        return None
    path = _resolve_tool_path(raw, must_exist=(tool_name != "write_file"))
    resolved = path.resolve()
    if _is_within_workspace(resolved):
        return None
    if tool_name != "write_file" and not resolved.exists():
        return None
    return str(resolved)


def check_permission(
    tool_name: str,
    inp: dict,
    mode: str = "default",
    plan_file_path: str | None = None,
) -> dict:
    """Returns {"action": "allow"|"deny"|"confirm", "message": ...}"""
    # 硬黑名单优先于一切模式（含 bypassPermissions）：不可逆/毁灭性操作一律拒绝。
    if tool_name == "run_shell" and is_hard_blocked(inp.get("command", "")):
        message = f"Hard-blocked command: {inp.get('command', '')}"
        _log_permission_decision(tool_name, inp, "deny", mode, message)
        return {"action": "deny", "message": message}

    if mode == "bypassPermissions":
        _log_permission_decision(tool_name, inp, "allow", mode)
        return {"action": "allow"}

    rule_result = _check_permission_rules(tool_name, inp)
    if rule_result == "deny":
        message = f"Denied by permission rule for {tool_name}"
        _log_permission_decision(tool_name, inp, "deny", mode, message)
        return {"action": "deny", "message": message}
    if rule_result == "allow":
        _log_permission_decision(tool_name, inp, "allow", mode)
        return {"action": "allow"}

    # workspace 路径沙箱：读/写工具的目标路径必须位于 cwd 子树内。
    # bypassPermissions 已提前放行；其余模式对越界路径 deny（dontAsk/plan/acceptEdits）
    # 或 confirm（default 走用户确认）。
    outside = _check_workspace_path(tool_name, inp)
    if outside is not None:
        if mode in ("dontAsk", "plan", "acceptEdits"):
            message = f"Path outside workspace: {outside}"
            _log_permission_decision(tool_name, inp, "deny", mode, message)
            return {"action": "deny", "message": message}
        _log_permission_decision(tool_name, inp, "confirm", mode, outside)
        return {"action": "confirm", "message": outside}

    if tool_name in READ_TOOLS:
        _log_permission_decision(tool_name, inp, "allow", mode)
        return {"action": "allow"}

    if mode == "plan":
        if tool_name in EDIT_TOOLS:
            file_path = inp.get("file_path") or inp.get("path")
            if plan_file_path and file_path == plan_file_path:
                _log_permission_decision(tool_name, inp, "allow", mode)
                return {"action": "allow"}
            message = f"Blocked in plan mode: {tool_name}"
            _log_permission_decision(tool_name, inp, "deny", mode, message)
            return {"action": "deny", "message": message}
        if tool_name == "run_shell":
            message = "Shell commands blocked in plan mode"
            _log_permission_decision(tool_name, inp, "deny", mode, message)
            return {"action": "deny", "message": message}
        # run_verification 是只读验证（规则命令受 verification 配置约束），plan 模式放行

    if tool_name in ("enter_plan_mode", "exit_plan_mode"):
        _log_permission_decision(tool_name, inp, "allow", mode)
        return {"action": "allow"}

    if mode == "acceptEdits" and tool_name in EDIT_TOOLS:
        _log_permission_decision(tool_name, inp, "allow", mode)
        return {"action": "allow"}

    needs_confirm = False
    confirm_message = ""

    if tool_name == "run_shell" and is_dangerous(inp.get("command", "")):
        needs_confirm = True
        confirm_message = inp.get("command", "")
    elif tool_name == "write_file" and not _resolve_tool_path(inp.get("file_path", ""), must_exist=False).exists():
        needs_confirm = True
        confirm_message = f"write new file: {inp.get('file_path', '')}"
    elif tool_name == "edit_file" and not _resolve_tool_path(inp.get("file_path", "")).exists():
        needs_confirm = True
        confirm_message = f"edit non-existent file: {inp.get('file_path', '')}"
    elif tool_name == "skill_evolve":
        needs_confirm = True
        confirm_message = f"evolve skill: {inp.get('skill_name', '')}"
    elif tool_name == "skill_create":
        needs_confirm = True
        confirm_message = f"create skill: {inp.get('name', '')}"

    if needs_confirm:
        if mode == "dontAsk":
            message = f"Auto-denied (dontAsk mode): {confirm_message}"
            _log_permission_decision(tool_name, inp, "deny", mode, message)
            return {"action": "deny", "message": message}
        _log_permission_decision(tool_name, inp, "confirm", mode, confirm_message)
        return {"action": "confirm", "message": confirm_message}

    _log_permission_decision(tool_name, inp, "allow", mode)
    return {"action": "allow"}






#----------------执行工具调用-----------------------
# 'agent' 和 'skill' 这两个工具在 agent.py 中处理，以避免循环依赖。"

async def execute_tool(
    name: str, inp: dict, read_file_state: dict[str, float] | None = None
) -> dict:
    """执行工具并返回结构化结果 {"content": str, "error": str|None, "retryable": bool}。"""
    if name == "memory_save":
        return _memory_save(inp)

    if name == "read_file":
        result = _read_file(inp)
        if read_file_state is not None and result["error"] is None:
            abs_path = str(_resolve_tool_path(inp["file_path"]).resolve())
            try:
                read_file_state[abs_path] =  os.path.getmtime(abs_path)
            except OSError:
                pass
        result["content"] = _truncate_result(result["content"])
        return result

    if name in ("write_file", "edit_file") and read_file_state is not None:
        abs_path = str(_resolve_tool_path(inp["file_path"], must_exist=(name == "edit_file")).resolve())
        if os.path.exists(abs_path):
            if abs_path not in read_file_state:
                verb = "writing" if name == "write_file" else "editing"
                return _tool_result("", error=f"Error: You must read this file before {verb}. Use read_file first to see its current contents.")
            if os.path.getmtime(abs_path) != read_file_state[abs_path]:
                verb = "writing" if name == "write_file" else "editing"
                return _tool_result("", error=f"Warning: {inp['file_path']} was modified externally since your last read. Please read_file again before {verb}.")

    #搜索和激活延迟加载的工具。
    if name == "tool_search":
        query = (inp.get("query") or "").lower()
        deferred = [t for t in tool_definitions if t.get("deferred")]
        matches = [
            t for t in deferred
            if query in t["name"].lower() or query in (t.get("description") or "").lower()
        ]
        if not matches:
            return _tool_result("No matching deferred tools found.")

        for m in matches:
            _activated_tools.add(m["name"])

        return _tool_result(
            json.dumps(
                [{"name": t["name"], "description": t.get("description", ""), "input_schema": t["input_schema"]} for t in
                 matches],
                indent=2,
            )
        )
    if name == "skill_evolve":
        from .skills import evolve_skill

        result = evolve_skill(
            skill_name=inp.get("skill_name", ""),
            lesson=inp.get("lesson", ""),
            rationale=inp.get("rationale", ""),
            target=inp.get("target", "active"),
        )
        return _tool_result(_truncate_result(json.dumps(result, ensure_ascii=False, indent=2)))

    if name == "skill_create":
        from .skills import create_skill

        result = create_skill(
            name=inp.get("name", ""),
            description=inp.get("description", ""),
            instructions=inp.get("instructions", ""),
            when_to_use=inp.get("when_to_use", "") or inp.get("when-to-use", ""),
            target=inp.get("target", "project"),
            context=inp.get("context", "inline"),
            user_invocable=bool(inp.get("user_invocable", False)),
            allowed_tools=inp.get("allowed_tools"),
            evidence=inp.get("evidence", ""),
        )
        return _tool_result(_truncate_result(json.dumps(result, ensure_ascii=False, indent=2)))

    handlers: dict = {
        "write_file": _write_file,
        "edit_file": _edit_file,
        "list_files": _list_files,
        "grep_search": _grep_search,
        "file_stats": _file_stats,
        "run_shell": _run_shell,
    }
    handler = handlers.get(name)

    if not handler:
        return _tool_result("", error=f"Unknown tool: {name}")

    result = handler(inp)

    # 更新时间
    if name in ("write_file", "edit_file") and read_file_state is not None and result["error"] is None:
        abs_path = str(_resolve_tool_path(inp["file_path"], must_exist=False).resolve())
        try:
            read_file_state[abs_path] = os.path.getmtime(abs_path)
        except OSError:
            pass

    if result["error"] is None:
        result["content"] = _truncate_result(result["content"])
    return result



def reset_permission_cache() -> None:
    global _cached_rules, _cached_rules_mtime
    _cached_rules = None
    _cached_rules_mtime = None




