#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import copy
import json
import os
import random
import re
import time
import uuid
from pathlib import Path
from typing import Callable, Awaitable, Any

import anthropic
import openai

from agents.mcp_client import McpManager
try:
    from agents.memory import (MemoryPrefetch, start_memory_prefetch,
                               format_memories_for_injection, save_memory_structured)
except ImportError:
    # memory.py 的 save_memory_structured 由并行子代理补充实现；缺失时降级为 None，
    # 相关抽取/回写路径内部自行判空跳过，避免模块级导入失败。
    from agents.memory import MemoryPrefetch, start_memory_prefetch, format_memories_for_injection
    save_memory_structured = None
from agents.prompt import build_system_prompt
from agents.session import save_session
from agents.subagent import get_sub_agent_config
from agents.tools import ToolDef, tool_definitions, execute_tool, CONCURRENCY_SAFE_TOOLS, check_permission, \
    get_active_tool_definitions
from agents.verification import (collect_written_file_rules, format_verification_feedback,
                                get_max_verification_attempts, load_verification_rules,
                                run_verification)
from agents.ui import print_info, print_divider, print_assistant_text, print_sub_agent_start, print_sub_agent_end, \
    start_spinner, stop_spinner, print_cost, print_tool_call, print_tool_result, print_confirmation, print_retry, \
    print_error, print_verification


# 指数退避重试


def _is_retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status in (429, 500, 502, 503, 504, 529):
        return True
    msg = str(error)
    if "overloaded" in msg or "ECONNRESET" in msg or "ETIMEDOUT" in msg:
        return True
    return False


def _safe_utf8_text(value: object) -> str:
    return str(value).encode("utf-8", errors="replace").decode("utf-8")


def _sanitize_for_utf8(value: Any) -> Any:
    if isinstance(value, str):
        return _safe_utf8_text(value)
    if isinstance(value, list):
        return [_sanitize_for_utf8(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_for_utf8(item) for item in value)
    if isinstance(value, dict):
        return {
            _sanitize_for_utf8(key): _sanitize_for_utf8(item)
            for key, item in value.items()
        }
    return value


def _dedupe_openai_messages(messages: list[dict]) -> list[dict]:
    """发送前防御性去重：assistant 消息内 tool_calls 相同 id 只保留首个；
    tool 角色消息相同 tool_call_id 只保留最先出现者（其余消息原样保留）。

    覆盖恢复会话/旧存档中的脏数据；不修改原消息数组（self._openai_messages
    保持不变），仅对发送数组生效。
    """
    deduped: list[dict] = []
    seen_tool_call_ids: set[str] = set()
    for msg in messages:
        if not isinstance(msg, dict):
            deduped.append(msg)
            continue
        if msg.get("role") == "tool":
            tid = msg.get("tool_call_id")
            if tid is not None and tid in seen_tool_call_ids:
                continue
            if tid is not None:
                seen_tool_call_ids.add(tid)
            deduped.append(msg)
            continue
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            seen_ids: set[str] = set()
            kept: list[dict] = []
            has_dup = False
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id") if isinstance(tc, dict) else None
                if tc_id is not None and tc_id in seen_ids:
                    has_dup = True
                    continue
                if tc_id is not None:
                    seen_ids.add(tc_id)
                kept.append(tc)
            if has_dup:
                new_msg = dict(msg)
                new_msg["tool_calls"] = kept
                deduped.append(new_msg)
                continue
        deduped.append(msg)
    return deduped


def _is_openai_style_messages(messages: list) -> bool:
    """格式判定：存在 tool 角色消息或 assistant 的 tool_calls 字段 → OpenAI 格式。"""
    return any(
        isinstance(m, dict) and (m.get("role") == "tool" or m.get("tool_calls"))
        for m in messages
    )


def _strip_unpaired_tool_blocks(messages: list[dict]) -> list[dict]:
    """清洗发送给摘要模型的请求消息，保证协议合法（无孤立工具块、角色交替）：

    - Anthropic 格式：逐条消息移除 tool_use/tool_result block，过滤后无内容的消息丢弃；
    - OpenAI 格式：移除 assistant 消息的 tool_calls 字段、丢弃 tool 角色消息，
      内容为空的消息丢弃；
    - 两种格式均做相邻同角色消息合并（丢弃后者），保证角色交替。

    不修改传入列表（逐条浅拷贝）。
    """
    if not messages:
        return []
    is_openai = _is_openai_style_messages(messages)
    cleaned: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        msg = dict(msg)
        if is_openai:
            if msg.get("role") == "tool":
                continue
            msg.pop("tool_calls", None)
            content = msg.get("content")
            if content is None or content == "":
                continue
        else:
            content = msg.get("content")
            if isinstance(content, list):
                kept = [
                    block for block in content
                    if not (isinstance(block, dict) and block.get("type") in ("tool_use", "tool_result"))
                ]
                if not kept:
                    continue  # 过滤后无内容 → 丢弃消息
                msg["content"] = kept
            elif not content:
                continue  # 空内容消息丢弃
        cleaned.append(msg)

    # 相邻同角色消息合并（丢弃后者），保证角色交替约束。
    merged: list[dict] = []
    for msg in cleaned:
        if merged and merged[-1].get("role") == msg.get("role"):
            continue
        merged.append(msg)
    return merged


def _append_user_text_merged(messages: list[dict], text: str) -> list[dict]:
    """向消息列表追加 user 文本；末尾已是 user 时合并（list 追加 text block /
    str 拼接）而非新增一条，避免连续两条 user 消息违反角色交替约束
    （参照 _inject_midloop_feedback 的合并写法）。直接修改并返回 messages。"""
    if messages and messages[-1].get("role") == "user":
        last = messages[-1]
        content = last.get("content")
        if isinstance(content, list):
            content.append({"type": "text", "text": text})
        else:
            last["content"] = (content or "") + "\n\n" + text
    else:
        messages.append({"role": "user", "content": text})
    return messages


async def _with_retry(fn, max_retries: int = 3):
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as error:
            if attempt >= max_retries or not _is_retryable(error):
                raise
            delay = min(1000 * (2 ** attempt), 30000) / 1000 + (random.random() * 1000) / 1000
            status = getattr(error, "status_code", None) or getattr(error, "status", None)
            reason = f"HTTP {status}" if status else (getattr(error, "code", None) or "network error")
            print_retry(attempt + 1, max_retries, reason)
            await asyncio.sleep(delay)

MODEL_CONTEXT = {
    "claude-opus-4-6": 200000,
    "claude-sonnet-4-6": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-haiku-4-5-20251001": 200000,
    "claude-opus-4-20250514": 200000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "deepseek-chat":200000,
    "deepseek-v4-flash": 1000000,
    "deepseek-v4-pro": 1000000
}

def _get_context_windows(model:str)->int:
    return MODEL_CONTEXT.get(model, 200000)


#多层级压缩常数
SNIP_THRESHOLD = 0.60
SNIP_PLACEHOLDER = "[Content snipped - re-read if needed]"
SNIPPABLE_TOOLS = {"read_file", "grep_search", "list_files", "run_shell"}
MICROCOMPACT_IDLE_S = 5 * 60  # 5 minutes

KEEP_RECENT_RESULTS = 3



def _get_max_output_tokens(model: str) -> int:
    m = model.lower()
    if "opus-4-6" in m:
        return 64000
    if "sonnet-4-6" in m:
        return 32000
    if any(x in m for x in ("opus-4", "sonnet-4", "haiku-4")):
        return 32000
    return 16384


# CJK 常用区间：中日韩统一表意文字、扩展 A、兼容表意文字、假名、谚文
_CJK_TOKEN_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")


def estimate_tokens(text: str) -> int:
    """近似估算文本 token 数：CJK 每字约 1.5 token，其余按约 4 字符/token。

    纯标准库实现的粗略度量，统一用于裁剪预算与发送前预估；空串返回 0，
    非空文本至少返回 1。
    """
    if not text:
        return 0
    cjk_count = len(_CJK_TOKEN_RE.findall(text))
    rest_len = max(0, len(text) - cjk_count)
    est = int(cjk_count * 1.5) + int(rest_len / 4)
    return max(1, est)


def _to_openai_tools(tools: list[ToolDef]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _filter_l1_rules(rules: list) -> list:
    """过滤出 L1 存在性规则（file_exists/glob_exists/dir_nonempty 等 level == 1 的规则）。

    子代理内部只跑 L1：L2/L3（file_contains / command_success 等）需要完整执行环境
    （编译、测试、业务状态断言），子代理上下文不适合运行，避免误报与副作用。
    """
    return [r for r in rules if getattr(r, "level", None) == 1]


class Agent:
    def __init__(self,
                 *,
                 permission_mode:str="default",
                 model:str="deepseek-chat",
                 api_base: str | None=None,
                 anthropic_base_url: str | None=None,
                 api_key: str | None=None,
                 thinking: bool=False,
                 max_cost_usd: float | None=None,
                 max_turns: int | None=None,
                 confirm_fn:Callable[[str], Awaitable[bool]] | None=None,
                 custom_system_prompt: str | None=None,
                 custom_tools: list[ToolDef] | None=None,
                 max_duration_s: float | None=None,
                 is_sub_agent: bool=False,):
        self.permission_mode = permission_mode
        self.thinking = thinking
        self.model = model
        self.use_openai = bool(api_base)
        self.is_sub_agent = is_sub_agent
        self.tools = custom_tools or tool_definitions
        self.max_cost_usd = max_cost_usd
        self.max_turns = max_turns
        self.confirm_fn = confirm_fn
        # wall-clock 超时控制（--max-duration）：记录起点，_check_timeout 判断是否超时。
        self.max_duration_s = max_duration_s
        self._start_time = time.monotonic()
        self._timed_out = False
        # 大结果临时文件目录（~/.otter-code/tool-results）的一次性过期清理标记。
        self._tool_results_cleaned = False
        self._custom_system_prompt = custom_system_prompt
        self.effective_window=_get_context_windows(model) - _get_max_output_tokens(model) - 4096
        self.session_id = uuid.uuid4().hex[:8]
        self.session_start_time= time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_token_count = 0
        self.current_turns = 0
        self.last_api_call_time = 0


        self._aborted = False
        #存储异步任务
        self._current_task:asyncio.Task | None = None
        #权限白名单
        self._confirmed_paths: set[str] = set()

        # 三层验证：本轮写过的产物文件 + 验证历史（用于自动验证与会话存档）
        self._written_files: set[str] = set()
        self._verification_log: list[dict] = []
        # 最近一次验证结果（run_once 的 verified 字段来源；None 表示本轮从未触发验证）。
        self._last_verification_passed: bool | None = None
        # 中途 L1 检查点的工具调用计数（仅主代理生效）。
        self._tool_call_count = 0


        # 计划模式”（Plan Mode）状态的变量
        self._pre_plan_mode: str | None=None
        self._plan_file_path: str | None=None
        self._plan_approval_fn : Callable[[str], Awaitable[bool]] | None=None
        self._context_cleared : bool=False

        #思考模式
        self._thinking_mode = self._resolve_thinking_mode()

        #子agent的输出缓存
        self._output_buffer: list[str] | None=None
        self._turn_output_buffer: list[str] | None = None

        # 编辑前读取
        self._read_file_state: dict[str, float] ={}

        #MCP集成
        self._mcp_manager = McpManager()
        self._mcp_initialized = False

        #记忆回溯
        #记忆agent已经回答过的信息
        self._already_surfaced_memories: set[str] = set()
        #当前会话占用的字节数
        self._session_memory_bytes = 0

        #区分message的历史消息
        self._anthropic_messages: list[str] = []
        self._openai_messages: list[str] = []
        self._last_retrieved_skill_reference: dict[str, Any] | None = None
        self._last_retrieved_skill_hits: list[dict[str, Any]] = []
        self._pending_skill_extraction_window: dict[str, Any] | None = None
        self._current_window_trace_id: str | None = None
        self._background_skill_tasks: set[asyncio.Task] = set()

        #构建系统提示词
        self._base_system_prompt = custom_system_prompt or build_system_prompt()

        if self.permission_mode == "plan":
            self._plan_file_path = self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
        else:
            self._system_prompt = self._base_system_prompt

        #初始化大模型客户端
        if self.use_openai:
            self._openai_client = openai.AsyncOpenAI(base_url=api_base, api_key=api_key)
            self._anthropic_client = None
            self._openai_messages.append({"role": "system", "content": self._system_prompt})
        else:
            kwargs : dict[str,Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if anthropic_base_url:
                kwargs["base_url"] = anthropic_base_url
            self._anthropic_client = anthropic.AsyncAnthropic(**kwargs)
            self._openai_client = None

    #判断返回模型的思考模式
    def _resolve_thinking_mode(self) -> str:
        if not self.thinking:
            return "disabled"
        if not self._model_supports_thinking():
            return "disabled"

        if self._mode_supports_adaptive_thinking(self.model):
            return "adaptive"
        return "enabled"

    def _model_supports_thinking(self) -> bool:
        m = self.model.lower()
        if "claude-3-" in m or "3-5-" in m or "3-7-" in m:
            return False
        if "claude" in m and any(x in m for x in ("opus", "sonnet", "haiku")):
            return True
        return False
    def _model_supports_adaptive_thinking(self) -> bool:
        m = self.model.lower()
        return "opus-4-6" in m or "sonnet-4-6" in m

    #生成一个用于保存 AI 计划（Plan）的 Markdown 文件的绝对路径。
    def _generate_plan_file_path(self) -> str:
        d = Path.home() / ".otter" / "plans"
        d.mkdir(parents=True, exist_ok=True)
        return str(d / f"plan-{self.session_id}.md")

    def _build_plan_mode_prompt(self) -> str:
        return f"""

    # Plan Mode Active

    Plan mode is active. You MUST NOT make any edits (except the plan file below), run non-readonly tools, or make any changes to the system.

    ## Plan File: {self._plan_file_path}
    Write your plan incrementally to this file using write_file or edit_file. This is the ONLY file you are allowed to edit.

    ## Workflow
    1. **Explore**: Read code to understand the task. Use read_file, list_files, grep_search.
    2. **Design**: Design your implementation approach. Use the agent tool with type="plan" if the task is complex.
    3. **Write Plan**: Write a structured plan to the plan file including:
       - **Context**: Why this change is needed
       - **Steps**: Implementation steps with critical file paths
       - **Verification**: How to test the changes
    4. **Exit**: Call exit_plan_mode when your plan is ready for user review.

    IMPORTANT: When your plan is complete, you MUST call exit_plan_mode. Do NOT ask the user to approve — exit_plan_mode handles that."""

    #判断当前的任务所有的任务是否完成
    @property
    def is_processing(self)->bool:
        return self._current_task is not None and not self._current_task.done()

    #大模型调用的工厂方法,构建一个用于记忆召回（memory recall）的 sideQuery 可调用对象，兼容anthropic, openai。
    # model 参数用于评测等场景覆盖主模型（如 flash 评），缺省时回落主模型（pro 写）。
    def _build_side_query(self, *, max_tokens: int = 256, model: str | None = None):
        if self._anthropic_client:
            client = self._anthropic_client
            model = model or self.model
            async def _sq(system:str, user_message:str)->str:

                resp = await client.messages.create(
                    model=model, max_tokens=max(1, int(max_tokens)), system=system,
                messages=[{"role": "user", "content": user_message}],
                )
                return "".join(b.text for b in resp.content if b.type == "text")
            return _sq
        if self._openai_client:
            client = self._openai_client
            model = model or self.model
            async def _sq_openai(system:str, user_message:str)->str:
                resp = await client.chat.completions.create(
                    model=model,
                    max_tokens=max(1, int(max_tokens)),
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],

                )
                return resp.choices[0].message.content or "" if resp.choices else ""
            return _sq_openai
        return None
    #异步任务取消（Abort）
    def abort(self) -> None:
        self._aborted = True
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    def set_confirm_fn(self, fn:Callable[[str], Awaitable[bool]]) -> None:
        self.confirm_fn = fn

    def set_plan_approval_fn(self, fn:Callable[[str], Awaitable[bool]]) -> None:
        self._plan_approval_fn = fn


    #计划模式开关（“状态切换与现场保护”机制）
    def toggle_plan_mode(self) -> str:
        """
               1. 退出计划模式（从 plan 切回原模式）
               当当前模式已经是 plan 时，执行 if 分支：
               恢复之前的状态：self.permission_mode = self._pre_plan_mode or "default"。
                   在进入计划模式时，程序会把原本的模式保存在 _pre_plan_mode 里。退出时，就把它重新拿出来赋值回去，恢复到切换前的状态。
               清理计划模式的痕迹：把 _pre_plan_mode 和 _plan_file_path（计划文件路径）清空，并将系统提示词 _system_prompt 恢复为最基础的 _base_system_prompt。
               同步 OpenAI 消息：如果底层使用的是 OpenAI 接口，它还会同步更新消息列表里的第一条系统提示词，确保 AI 的上下文也跟着切换回来。
               反馈返回：打印退出提示，并返回恢复后的模式名称。

               2. 进入计划模式（从其他模式切入 plan）
       当当前模式不是 plan 时，执行 else 分支：
       保护当前现场：self._pre_plan_mode = self.permission_mode。先把当前正在使用的模式（比如正常模式或自动接受模式）暂存起来，方便以后能原路返回。
       切换并初始化：将当前模式设为 "plan"，生成一个专属的计划文件路径，并扩展系统提示词。通过拼接 _build_plan_mode_prompt()，给 AI 注入“只动脑不动手、输出结构化计划”的专属指令。
       同步 OpenAI 消息：同样地，如果使用 OpenAI，也会实时更新上下文里的系统提示词。
       反馈与返回：打印进入提示（包含计划文件的路径），并返回 "plan"。
        """
        if self.permission_mode == "plan":
            self.permission_mode = self._pre_plan_mode or "default"
            self._pre_plan_mode = None
            self._plan_file_path = None
            self._system_prompt = self._base_system_prompt
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] =self._system_prompt
            print_info(f"Exited plan mode -> {self.permission_mode} mode")
            return self.permission_mode
        else:
            self._pre_plan_mode = self.permission_mode
            self.permission_mode = "plan"
            self._plan_file_path = self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
            print_info(f"Entered plan mode. Plan file: {self._plan_file_path}")
            return "plan"

    def get_token_usage(self) -> dict:
        return {"input":self.total_input_tokens, "output":self.total_output_tokens}

    #主入口

    async def  chat(self, user_message:str)->None:
        #懒加载MCP服务在第一次chat的时候
        if not self._mcp_initialized and not self.is_sub_agent:
            self._mcp_initialized = True
            try:
                await self._mcp_manager.load_and_connect()
                mcp_defs = self._mcp_manager.get_tool_definitions()
                if mcp_defs:
                    self.tools = self.tools + mcp_defs
            except Exception as e:
                print_error(f"MCP init failed: {e}")

        original_user_message = _safe_utf8_text(user_message)
        ready_skill_extraction_window: dict[str, Any] | None = None
        self._last_retrieved_skill_reference = None
        self._last_retrieved_skill_hits = []
        if not self.is_sub_agent:
            ready_skill_extraction_window = self._pop_pending_skill_extraction_window(original_user_message)
            user_message, self._last_retrieved_skill_reference = self._augment_user_message_with_skill_context(
                original_user_message
            )

        self._aborted = False
        self._turn_output_buffer = []
        coro = self._chat_openai(user_message) if self.use_openai else self._chat_anthropic(user_message)
        self._current_task = asyncio.create_task(coro)
        try:
            await self._current_task
        except asyncio.CancelledError:
            self._aborted = True

        finally:
            self._current_task = None
        if self._timed_out:
            note = "\n\n[max-duration reached] 已达最大运行时长限制，agent 已优雅中止。"
            if self._output_buffer is not None:
                self._output_buffer.append(note)
            if self._turn_output_buffer is not None:
                self._turn_output_buffer.append(note)
            if self._output_buffer is None:
                print_info(note.strip())
        assistant_text = "".join(self._turn_output_buffer or []).strip()
        # sleep-time compute：主代理会话结束时从本轮对话抽取候选记忆并入库，失败静默。
        if not self.is_sub_agent:
            await self._extract_memories_from_session()
        self._turn_output_buffer = None
        if not self.is_sub_agent and not self._aborted:
            self._schedule_background_skill_task(self._run_skill_usage_tracking(original_user_message, assistant_text))
            if ready_skill_extraction_window:
                self._schedule_background_skill_task(self._run_online_skill_evolution(ready_skill_extraction_window))
            self._set_pending_skill_extraction_window(
                original_user_message=original_user_message,
                assistant_text=assistant_text,
                retrieved_reference=self._last_retrieved_skill_reference,
            )
        if not self.is_sub_agent:
            print_divider()
            self._auto_save()



   #执行一次对话，收集本轮模型输出文本，并返回本轮消耗的 token 数
    async def run_once(self, prompt:str)->dict:
        self._output_buffer = []
        self._last_verification_passed = None
        prev_in = self.total_input_tokens
        prev_out = self.total_output_tokens
        await self.chat(prompt)
        text = "".join(self._output_buffer)
        self._output_buffer = None
        # verified：最近一次验证是否通过；本轮从未触发验证（无规则/未到验证点）时默认 True。
        verified = self._last_verification_passed if self._last_verification_passed is not None else True
        return {
            "text": text,
            "tokens":{
                "input":self.total_input_tokens-prev_in,
                "output":self.total_output_tokens-prev_out
            },
            "verified": bool(verified),
        }

    #输出工具：统一处理模型输出文本。根据当前是否处于“收集输出”的模式
    # 决定是把文本存进缓冲区，还是直接打印到终端。
    def _emit_text(self, text:str)->None:
        text = _safe_utf8_text(text)
        if self._turn_output_buffer is not None:
            self._turn_output_buffer.append(text)
        if self._output_buffer is not None:
            self._output_buffer.append(text)
        else:
            print_assistant_text(text)

    def _refresh_runtime_system_prompt(self) -> None:
        if self._custom_system_prompt is not None:
            return
        self._base_system_prompt = build_system_prompt()
        if self.permission_mode == "plan":
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
        else:
            self._system_prompt = self._base_system_prompt
        if self.use_openai and self._openai_messages:
            self._openai_messages[0]["content"] = self._system_prompt

    def _augment_user_message_with_skill_context(self, user_message: str) -> tuple[str, dict[str, Any] | None]:
        try:
            from .skills import format_retrieved_skill_context

            context, top_ref = format_retrieved_skill_context(user_message, limit=3)
        except Exception:
            return user_message, None
        if top_ref and isinstance(top_ref.get("all_hits"), list):
            self._last_retrieved_skill_hits = list(top_ref.get("all_hits") or [])
        if not context.strip():
            return user_message, top_ref
        return f"{user_message}\n\n{context}", top_ref

    def _strip_runtime_injections(self, text: str) -> str:
        return re.sub(r"\n*<retrieved_skills>.*?</retrieved_skills>\s*", "", str(text or ""), flags=re.DOTALL).strip()

    def _message_text(self, msg: dict[str, Any]) -> str:
        content = msg.get("content")
        if isinstance(content, str):
            return self._strip_runtime_injections(content)
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(str(block.get("text") or ""))
                    elif "content" in block and block.get("type") not in {"tool_result", "tool_use"}:
                        parts.append(str(block.get("content") or ""))
            return self._strip_runtime_injections("\n".join(parts))
        return ""

    def _recent_dialog_messages(self, *, max_messages: int = 8) -> list[dict[str, str]]:
        raw_messages = self._openai_messages if self.use_openai else self._anthropic_messages
        out: list[dict[str, str]] = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            text = self._message_text(msg)
            if text:
                out.append({"role": role, "content": text})
        return out[-max(2, int(max_messages)) :]

    async def _confirm_online_skill_write(self, summary: str) -> bool:
        if self.permission_mode in {"bypassPermissions", "acceptEdits"}:
            return True
        if self.permission_mode in {"plan", "dontAsk"}:
            return False
        if self.confirm_fn is None:
            return False
        print_confirmation(summary)
        try:
            return bool(await self.confirm_fn(summary))
        except Exception:
            return False

    async def _confirm_background_online_skill_write(self, summary: str) -> bool:
        return self.permission_mode in {"bypassPermissions", "acceptEdits"}

    def _online_evolution_enabled(self) -> bool:
        raw = os.environ.get("OTTER_AUTO_SKILL_EVOLUTION", "1").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def _schedule_background_skill_task(self, coro) -> None:
        if self.permission_mode == "plan":
            try:
                coro.close()
            except Exception:
                pass
            return
        task = asyncio.create_task(coro)
        self._background_skill_tasks.add(task)

        def _done(done_task: asyncio.Task) -> None:
            self._background_skill_tasks.discard(done_task)
            try:
                done_task.result()
            except Exception:
                pass

        task.add_done_callback(_done)

    async def drain_background_skill_tasks(self) -> None:
        tasks = [task for task in self._background_skill_tasks if not task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _pop_pending_skill_extraction_window(self, next_user_feedback: str) -> dict[str, Any] | None:
        pending = self._pending_skill_extraction_window
        self._pending_skill_extraction_window = None
        if not pending:
            return None
        messages = list(pending.get("messages") or [])
        feedback = _safe_utf8_text(next_user_feedback).strip()
        if feedback:
            messages.append({"role": "user", "content": feedback})
        pending["messages"] = messages[-10:]
        pending["next_user_feedback"] = feedback
        return pending

    def _set_pending_skill_extraction_window(
        self,
        *,
        original_user_message: str,
        assistant_text: str,
        retrieved_reference: dict[str, Any] | None,
    ) -> None:
        if not original_user_message.strip() or not assistant_text.strip():
            return
        messages = self._recent_dialog_messages(max_messages=8)
        # 窗口级 trace_id：基于内容哈希（skill + latest_user + messages），同一窗口回写共用同一条 trace。
        try:
            from .skill_evolution import _stable_hash, record_skill_trace

            top_hit = max(self._last_retrieved_skill_hits or [], key=lambda h: float(h.get("score", 0.0) or 0.0))
            top_skill = str(top_hit.get("name") or "") or (retrieved_reference or {}).get("name") or ""
            trace_id = _stable_hash(
                {"skill": top_skill, "messages": messages, "latest_user": original_user_message}
            )
            self._current_window_trace_id = trace_id
            record_skill_trace(
                skill_name=top_skill,
                trace_id=trace_id,
                trigger_query=original_user_message,
                hit_scores=list(self._last_retrieved_skill_hits or []),
                messages=messages,
                latest_user=original_user_message,
                latest_assistant=assistant_text,
                session_id=self.session_id,
            )
        except Exception:
            pass
        self._pending_skill_extraction_window = {
            "messages": messages,
            "latest_user": original_user_message,
            "latest_assistant": assistant_text,
            "retrieved_reference": self._compact_retrieved_reference(retrieved_reference),
            "session_id": self.session_id,
        }

    def _compact_retrieved_reference(self, ref: dict[str, Any] | None) -> dict[str, Any] | None:
        if not ref:
            return None
        return {k: v for k, v in ref.items() if k != "all_hits"}

    async def _run_online_skill_evolution(self, window: dict[str, Any], *, interactive_confirm: bool = False) -> None:
        if not self._online_evolution_enabled() or self.permission_mode == "plan":
            return
        messages = list(window.get("messages") or [])
        if not messages:
            return

        side_query = self._build_side_query(max_tokens=2200)
        if side_query is None:
            return

        try:
            from .online_skill_evolution import online_ingest
        except Exception:
            return

        result = await online_ingest(
            messages=messages,
            side_query=side_query,
            retrieved_reference=window.get("retrieved_reference") or None,
            hint=str(window.get("hint") or ""),
            confirm_write=self._confirm_online_skill_write if interactive_confirm else self._confirm_background_online_skill_write,
            target=os.environ.get("OTTER_AUTO_SKILL_TARGET", "project"),
            trace_id=self._current_window_trace_id or "",
        )
        # 演化结果回写同一窗口的 trace，串起“触发->执行->结果->演化”链条。
        if self._current_window_trace_id:
            try:
                from .skill_evolution import record_skill_trace

                record_skill_trace(
                    skill_name=str(result.get("skill") or ""),
                    trace_id=self._current_window_trace_id,
                    evolution_action=str(result.get("action") or "none"),
                    evolution_time=str(result.get("time") or ""),
                )
            except Exception:
                pass
        if result.get("ok"):
            if result.get("action") in {"add", "merge"}:
                self._refresh_runtime_system_prompt()
                print_info(f"Online skill {result.get('action')}: {result.get('skill')}")
                self._maybe_schedule_skill_eval(str(result.get("skill") or ""))
        elif result.get("action") not in {"add_denied", "merge_denied"}:
            print_error(f"Online skill evolution failed: {result.get('error') or result}")

    def _maybe_schedule_skill_eval(self, skill_name: str) -> None:
        # 事件驱动增量评测（VeRO 预算控制）：evolve/add 成功后，若 OTTER_EVAL_AUTO=1 则后台评测。
        # judge 用独立 flash 模型（OTTER_EVAL_JUDGE_MODEL），演化抽取仍走主模型（pro 写 + flash 评）。
        raw = os.environ.get("OTTER_EVAL_AUTO", "0").strip().lower()
        if raw not in {"1", "true", "yes", "on"} or not skill_name:
            return
        try:
            from .online_skill_eval import evaluate_online_skill_evolution_async

            judge_model = os.environ.get("OTTER_EVAL_JUDGE_MODEL") or "deepseek-v4-flash"
            side_query = self._build_side_query(max_tokens=500, model=judge_model)

            async def _eval_task() -> None:
                # 单 lineage 增量评测：只评测被改动的 skill，控制 token 成本。
                await evaluate_online_skill_evolution_async(side_query=side_query, skill_name=skill_name)

            self._schedule_background_skill_task(_eval_task())
        except Exception:
            pass

    async def _run_skill_usage_tracking(self, original_user_message: str, assistant_text: str) -> None:
        if not self._online_evolution_enabled() or self.permission_mode == "plan":
            return
        hits = list(self._last_retrieved_skill_hits or [])
        if not hits or not assistant_text.strip():
            return
        side_query = self._build_side_query(max_tokens=700)
        try:
            from .online_skill_evolution import judge_retrieved_skill_usage
            from .skills import record_usage_judgments

            judgments = await judge_retrieved_skill_usage(
                hits=hits,
                user_message=original_user_message,
                assistant_text=assistant_text,
                side_query=side_query,
            )
            result = record_usage_judgments(judgments)
            # usage 判断结果回写同一窗口 trace：记录每个命中 skill 的 retrieved/relevant/used。
            if self._current_window_trace_id:
                try:
                    from .skill_evolution import record_skill_trace

                    record_skill_trace(
                        skill_name=str((judgments[0] or {}).get("name") or "") if judgments else "",
                        trace_id=self._current_window_trace_id,
                        usage_judgment={"judgments": judgments[:10], "pruned": list(result.get("pruned") or [])},
                    )
                except Exception:
                    pass
            if result.get("pruned"):
                self._refresh_runtime_system_prompt()
        except Exception:
            return

    async def extract_now(self, hint: str = "") -> dict[str, Any]:
        pending = self._pending_skill_extraction_window
        if not pending:
            return {"ok": False, "error": "no pending online skill extraction window"}
        window = dict(pending)
        window["hint"] = hint
        await self._run_online_skill_evolution(window, interactive_confirm=True)
        self._pending_skill_extraction_window = None
        return {"ok": True}


    def clear_history(self)->None:
        self._anthropic_messages = []
        self._openai_messages = []
        self._pending_skill_extraction_window = None
        self._current_window_trace_id = None
        self._last_retrieved_skill_reference = None
        self._last_retrieved_skill_hits = []
        if self.use_openai:
            self._openai_messages.append({"role": "system", "content":self._system_prompt})
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_token_count = 0
        print_info("Conversation cleared.")

    def show_cost(self):
        total = self._get_current_cost_usd()
        budget_info = f" / ${self.max_cost_usd} budget" if self.max_cost_usd else ""
        turn_info = f" | Turns: {self.current_turns}/{self.max_turns}" if self.max_turns else ""
        print_info(
            f"Tokens: {self.total_input_tokens} in / {self.total_output_tokens} out\n  Estimated cost: ${total:.4f}{budget_info}{turn_info}")

    #获取当前的花费，
    def _get_current_cost_usd(self) -> float:
        return (self.total_input_tokens / 1_000_000) * 3 + (self.total_output_tokens / 1_000_000) * 15

    #检查预算
    def _check_budget(self) -> dict:
        if self.max_cost_usd is not None and self._get_current_cost_usd() >= self.max_cost_usd:
            return {"exceeded": True, "reason": f"Cost limit reached (${self._get_current_cost_usd():.4f} >= ${self.max_cost_usd})"}
        if self.max_turns is not None and self.current_turns >= self.max_turns:
            return {"exceeded": True, "reason": f"Turn limit reached ({self.current_turns} >= {self.max_turns})"}
        return {"exceeded": False}

    #压缩会话
    async def compact(self)->None:
        await self._compact_conversation()


    #恢复会话信息
    def restore_session(self, data:dict)->None:
        if data.get("anthropicMessages"):
            self._anthropic_messages = self._normalize_anthropic_messages(_sanitize_for_utf8(data["anthropicMessages"]))
        if data.get("openaiMessages"):
            self._openai_messages = _sanitize_for_utf8(data["openaiMessages"])
        # 恢复 read-before-edit 保护状态：edit_file 仍要求先 read_file。
        raw_state = data.get("readFileState")
        if isinstance(raw_state, dict):
            cleaned: dict[str, float] = {}
            for k, v in raw_state.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    cleaned[str(k)] = float(v)
            self._read_file_state = cleaned
        print_info(f"Session restored ({self._get_message_count()} messages).")



#整理 Anthropic 的历史消息，修正部分角色错误，并丢弃不合法的工具调用消息。
    def _normalize_anthropic_messages(self, messages: list[dict]) -> list[dict]:
        role_normalized = []
        for msg in messages:
            copied = dict(msg)
            content = copied.get("content")
            if copied.get("role") == "user" and isinstance(content, list):
                if any(isinstance(block, dict) and block.get("type") == "tool_use" for block in content):
                    copied["role"] = "assistant"
            role_normalized.append(copied)

        normalized = []
        i = 0
        while i < len(role_normalized):
            msg = role_normalized[i]
            tool_use_ids = self._anthropic_tool_use_ids(msg)
            if not tool_use_ids:
                normalized.append(msg)
                i += 1
                continue

            next_msg = role_normalized[i + 1] if i + 1 < len(role_normalized) else None
            result_ids = self._anthropic_tool_result_ids(next_msg) if next_msg else set()
            if tool_use_ids.issubset(result_ids):
                normalized.append(msg)
                normalized.append(next_msg)
                i += 2
                continue

            # 半截轮连带丢弃：assistant 因结果不完整被跳过时，若紧随其后的 user
            # 消息内容全为指向这些孤儿 tool_use 的 tool_result（无文本），一并跳过，
            # 避免恢复后的历史残留孤立 tool_result（后续发送触发 400）。
            if self._is_orphan_tool_result_follower(next_msg, tool_use_ids):
                i += 2
                continue

            i += 1
        return normalized

    @staticmethod
    def _is_orphan_tool_result_follower(next_msg: dict | None, skipped_tool_use_ids: set[str]) -> bool:
        """判断紧随被跳过 assistant 的消息是否为纯孤儿 tool_result 消息：
        role 为 user、内容全为 tool_result block（无文本等其他 block）、
        且每个 tool_use_id 都属于被跳过的 assistant。"""
        if not next_msg or not skipped_tool_use_ids:
            return False
        if next_msg.get("role") != "user" or not isinstance(next_msg.get("content"), list):
            return False
        blocks = next_msg["content"]
        if not blocks:
            return False
        for block in blocks:
            if not isinstance(block, dict):
                return False
            if block.get("type") != "tool_result":
                return False  # 含文本等其他 block → 保留（文本有价值）
            if block.get("tool_use_id") not in skipped_tool_use_ids:
                return False  # 指向其他 assistant 的结果 → 保留
        return True

    @staticmethod
    def _anthropic_tool_use_ids(msg: dict | None) -> set[str]:
        if not msg or msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            return set()
        return {
            block.get("id")
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id")
        }

    @staticmethod
    def _anthropic_tool_result_ids(msg: dict | None) -> set[str]:
        if not msg or msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            return set()
        return {
            block.get("tool_use_id")
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id")
        }

    def _get_message_count(self) -> int:
        return len(self._openai_messages) if self.use_openai else len(self._anthropic_messages)

    def _auto_save(self) -> None:
        try:
            save_session(self.session_id, {
                "metadata": {
                    "id": self.session_id,
                    "model": self.model,
                    "cwd": str(Path.cwd()),
                    "startTime": self.session_start_time,
                    "messageCount": self._get_message_count(),
                },
                "anthropicMessages": _sanitize_for_utf8(self._anthropic_messages) if not self.use_openai else None,
                "openaiMessages": _sanitize_for_utf8(self._openai_messages) if self.use_openai else None,
                "verification": self._verification_log or None,
                "readFileState": self._read_file_state or None,
            })
        except Exception:
            pass

    def _persist_compact_summary(self, summary_text: str) -> None:
        """将摘要压缩产生的会话摘要回写为 project 类型记忆。失败静默。

        name 带时间戳（秒 + 纳秒），保证每次压缩产生独立记忆文件，
        不再被 save_memory_structured 按固定 name 去重覆盖历史。
        """
        if not summary_text or self.is_sub_agent:
            return
        try:
            if save_memory_structured is not None:
                save_memory_structured(
                    name=f"conversation-compact-summary-{time.strftime('%Y%m%d%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}",
                    description="Auto-saved conversation summary from context compaction",
                    type="project",
                    content=summary_text,
                    session_id=self.session_id,
                )
        except Exception:
            pass

    def estimate_messages_tokens(self) -> int:
        """估算当前消息列表的 token 总量（含 system prompt 与 tools schema 粗估）。"""
        raw_messages = self._openai_messages if self.use_openai else self._anthropic_messages
        total = estimate_tokens(self._system_prompt)
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                total += estimate_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if isinstance(block.get("text"), str):
                        total += estimate_tokens(block["text"])
                    elif isinstance(block.get("content"), str):
                        total += estimate_tokens(block["content"])
        # 工具定义的固定开销粗估（每轮 system/tools 前缀），失败跳过。
        try:
            total += estimate_tokens(json.dumps(self.tools))
        except Exception:
            pass
        return total

    async def _extract_memories_from_session(self) -> None:
        """会话结束时用 side query 从本轮对话抽取候选记忆并入库。失败静默。"""
        if self.is_sub_agent or not self._turn_output_buffer:
            return
        try:
            side_query = self._build_side_query()
            if side_query is None or save_memory_structured is None:
                return
            system_prompt = (
                "You extract structured memories from a conversation turn. "
                'Return ONLY a JSON object: {"memories": [{"name": str, "description": str, "type": str, "content": str}]}. '
                "type must be one of: user, feedback, project, reference (use project if unsure). "
                "Extract task points, user preferences, and project decisions. Skip empty content."
            )
            user_message = "Conversation turn:\n\n" + "\n".join(self._turn_output_buffer or [])
            raw = await side_query(system_prompt, user_message)
            match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
            if not match:
                return
            data = json.loads(match.group(0))
            for mem in (data.get("memories") or []):
                name = str(mem.get("name") or "").strip()
                content = str(mem.get("content") or "").strip()
                if not name or not content:
                    continue
                description = str(mem.get("description") or "").strip()
                mtype = str(mem.get("type") or "project").strip()
                if mtype not in ("user", "feedback", "project", "reference"):
                    mtype = "project"
                save_memory_structured(
                    name=name,
                    description=description,
                    type=mtype,
                    content=content,
                    session_id=self.session_id,
                )
        except Exception as e:
            print_error(f"[memory] session extraction failed: {e}")

    #自动压缩
    async def _check_and_compact(self)->None:
        if self.last_input_token_count>self.effective_window*0.85:
            print_info("Context window filling up, compacting conversation...")
            await self._compact_conversation()

    async def _compact_conversation(self)->None:
        if self.use_openai:
            await self._compact_openai()
        else:
            await self._compact_anthropic()
        print_info("Conversation compacted.")

    async def _compact_anthropic(self)->None:
        if len (self._anthropic_messages)<4:
            return

        last_user_msg = self._anthropic_messages[-1]
        # 摘要请求清洗：移除孤立 tool_use/tool_result block、空消息与连续同角色，
        # 保证工具轮之后触发压缩不产生 400。
        cleaned = _strip_unpaired_tool_blocks(_sanitize_for_utf8(self._anthropic_messages[:-1]))
        request_messages = _append_user_text_merged(
            cleaned,
            "Summarize the conversation so far in a concise paragraph, preserving key decisions, file paths, and context needed to continue the work.",
        )
        summary_resp = await self._anthropic_client.messages.create(
            model=self.model,
            max_tokens=2048,
            system ="You are a conversation summarizer. Be concise but preserve important details.",
            messages=request_messages,
        )
        summary_text = summary_resp.content[0].text if summary_resp.content and  summary_resp.content[0].type == "text" else "No summary available."
        # 摘要回写 project 记忆（不修改消息结构，只在外部写文件）。
        self._persist_compact_summary(summary_text)
        self._anthropic_messages=[
            {"role":"user","content":f"[Previous conversation summary]\n{summary_text}"},
            {"role": "assistant", "content": "Understood. I have the context from our previous conversation. How can I continue helping?"},
        ]
        # 追加最后一条 user 消息前同样清洗（含 tool_result 的 user 只保留文本）。
        stripped_last = _strip_unpaired_tool_blocks([last_user_msg])
        if stripped_last and stripped_last[0].get("role") == "user":
            self._anthropic_messages.append(stripped_last[0])
        self.last_input_token_count=0

    async def _compact_openai(self)->None:
        if len (self._openai_messages)<4:
            return
        system_msg = self._openai_messages[0]
        last_user_msg = self._openai_messages[-1]
        # 摘要请求清洗：移除 assistant 的 tool_calls 字段、tool 角色消息与空消息，
        # 保证任何工具轮之后触发压缩均不产生 400。
        cleaned = _strip_unpaired_tool_blocks(_sanitize_for_utf8(self._openai_messages[1:-1]))
        request_messages = [
            {"role": "system", "content": "You are a conversation summarizer. Be concise but preserve important details."},
            *_append_user_text_merged(
                cleaned,
                "Summarize the conversation so far in a concise paragraph, preserving key decisions, file paths, and context needed to continue the work.",
            ),
        ]
        summary_resp = await self._openai_client.chat.completions.create(
            model=self.model,
            messages=request_messages,
        )
        summary_text = summary_resp.choices[0].message.content or "" if summary_resp.choices else ""
        # 摘要回写 project 记忆（不修改消息结构，只在外部写文件）。
        self._persist_compact_summary(summary_text)
        self._openai_messages=[
            system_msg,
            {"role": "user", "content": f"[Previous conversation summary]\n{summary_text}"},
            {"role": "assistant","content": "Understood. I have the context from our previous conversation. How can I continue helping?"},
        ]
        # 追加最后一条 user 消息前同样清洗（tool 角色消息被丢弃）。
        stripped_last = _strip_unpaired_tool_blocks([last_user_msg])
        if stripped_last and stripped_last[0].get("role") == "user":
            self._openai_messages.append(stripped_last[0])
        self.last_input_token_count=0

    #多层级压缩流水线
    def _run_compression_pipeline(self)->None:
        if self.use_openai:
            self._budget_tool_results_openai()
            self._snip_stale_results_openai()
            self._microcompact_openai()
        else:
            self._budget_tool_results_anthropic()
            self._snip_stale_results_anthropic()
            self._microcompact_anthropic()

    #第一层级压缩，预算压缩
    def _budget_tool_results_anthropic(self)->None:
        #计算利用率：utilization = 已用Token / 有效窗口大小。
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        #如果利用率低于 50%，说明空间还很充裕，直接返回，不做任何处理。
        if utilization < 0.5:
            return
        #动态预算（Budget，单位 token）：危急状态（>70%）：如果利用率很高，允许单个工具结果保留 6000 token。
        # 警戒状态（50%-70%）：如果利用率中等，只允许保留 12000 token（按 4 字符/token 反推约 48000 字符）。
        budget = 6000 if utilization > 0.7 else 12000

        for msg in self._anthropic_messages:

            #只处理 role 为 "user" 的消息。在工具调用流程中，工具的执行结果通常是以“用户”的身份反馈给模型的。

            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str) and estimate_tokens(block["content"]) > budget:
                    #计算保留长度 (keep)：keep = (budget * 4 - 80) // 2 这里预留了约 80 个字符的空间给中间的提示语，
                    # 按 4 字符/token 把 token 预算换算回字符数，剩下的长度平分给开头和结尾。
                    keep = (budget * 4 - 80) // 2
                    #重组新内容 = 开头部分 + 提示语 + 结尾部分
                    block["content"] = block["content"][:keep] + f"\n\n[... budgeted: {len(block['content']) - keep * 2} chars truncated ...]\n\n" + block["content"][-keep:]

    def _budget_tool_results_openai(self)->None:
        #计算利用率：utilization = 已用Token / 有效窗口大小。
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        #如果利用率低于 50%，说明空间还很充裕，直接返回，不做任何处理。
        if utilization < 0.5:
            return
        #动态预算（Budget，单位 token）：危急状态（>70%）：如果利用率很高，允许单个工具结果保留 6000 token。
        # 警戒状态（50%-70%）：如果利用率中等，只允许保留 12000 token（按 4 字符/token 反推约 48000 字符）。
        budget = 6000 if utilization > 0.7 else 12000

        for msg in self._openai_messages:
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and estimate_tokens(msg["content"]) > budget:
                keep = (budget * 4 - 80) // 2
                msg["content"] = msg["content"][:keep] + f"\n\n[... budgeted: {len(msg['content']) - keep * 2} chars truncated ...]\n\n" + msg["content"][-keep:]


    #第二级策略：修剪过期的工具执行结果
    def _snip_stale_results_anthropic(self) -> None:
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        if utilization < SNIP_THRESHOLD:
            return
        results = []
        for mindex,  msg in enumerate(self._anthropic_messages):
            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue

            for bindex, block in enumerate(msg["content"]):
                if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str) and block["content"] != SNIP_PLACEHOLDER:
                    tool_use_id = block.get("tool_use_id")
                    # 对每个 tool_result，通过 tool_use_id 反查它来自哪个工具
                    tool_info = self._find_tool_use_by_id(tool_use_id)
                    if tool_info and tool_info["name"] in SNIPPABLE_TOOLS:
                        results.append({"mindex": mindex, "bindex": bindex, "name": tool_info["name"], "file_path": tool_info.get("input", {}).get("file_path")})

        if len(results) <= KEEP_RECENT_RESULTS:
            return

        to_snip =  set()
        seen_files: dict[str, list[int]] = {}

        for i, r in enumerate(results):
            if r["name"] == "read_file" and r.get("file_path"):
                seen_files.setdefault(r["file_path"], []).append(i)
        #如果一个文件被读取了多次，只保留最后一次读取的结果，把前面几次读取的内容全部标记为“修剪”（Snip）。
        for indices in seen_files.values():
            if len (indices) >1 :
                for j in indices[:-1]:
                    to_snip.add (j)

        snip_before = len(results) - KEEP_RECENT_RESULTS
        for i in range (snip_before):
            to_snip.add(i)

        for idx in to_snip:
            r = results[idx]
            self._anthropic_messages[r["mindex"]]["content"][r["bindex"]]["content"] = SNIP_PLACEHOLDER

    def _snip_stale_results_openai(self) -> None:
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        if utilization < SNIP_THRESHOLD:
            return
        tool_msgs = []
        for i, msg in enumerate(self._openai_messages):
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and msg["content"] != SNIP_PLACEHOLDER:
                tool_msgs.append(i)
        if len(tool_msgs) <= KEEP_RECENT_RESULTS:
            return
        snip_count = len(tool_msgs) - KEEP_RECENT_RESULTS
        for i in range(snip_count):
            self._openai_messages[tool_msgs[i]]["content"] = SNIP_PLACEHOLDER

    #微压缩

    #基于“时间”的上下文瘦身策略，
    #如果已经很久没说话了，说明之前的工具执行结果你已经看完了，那就把它们清理掉，腾出空间

    def _microcompact_anthropic(self) -> None:
        if not self.last_api_call_time or (time.time() - self.last_api_call_time) < MICROCOMPACT_IDLE_S:
            return

        all_results = []
        for mindex, msg in enumerate(self._anthropic_messages):
            if msg.get("role")!="user" or not isinstance(msg.get("content"), list):
                continue
            for bindex, block in enumerate(msg["content"]):
                if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str) and block["content"] not in (SNIP_PLACEHOLDER, "[Old result cleared]"):
                    all_results.append((mindex, bindex))

        clear_count = len(all_results) - KEEP_RECENT_RESULTS
        for i in range(max(0, clear_count)):
            mi, bi = all_results[i]
            self._anthropic_messages[mi]["content"][bi]["content"] = "[Old result cleared]"

    def _microcompact_openai(self) -> None:
        if not self.last_api_call_time or (time.time() - self.last_api_call_time) < MICROCOMPACT_IDLE_S:
            return
        tool_msgs = []
        for i, msg in enumerate(self._openai_messages):
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and msg["content"] not in (SNIP_PLACEHOLDER, "[Old result cleared]"):
                tool_msgs.append(i)
        clear_count = len(tool_msgs) - KEEP_RECENT_RESULTS
        for i in range(max(0, clear_count)):
            self._openai_messages[tool_msgs[i]]["content"] = "[Old result cleared]"

    def _find_tool_use_by_id(self, tool_use_id: int) -> dict | None:
        for msg in self._anthropic_messages:
            if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
                continue

            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id") == tool_use_id:
                    return {"name": block["name"], "input": block.get("input", {})}

    #大结果持久化
    #如果工具返回的结果太大（超过 30KB），不要硬塞进上下文里，而是把它存成一个临时文件。
    # 然后在对话里只留一个‘文件路径’和‘内容预览’。如果模型后面还需要看完整内容，它可以再次调用工具去读取这个文件

    def _cleanup_stale_tool_results(self, directory: Path | str | None = None,
                                    max_age_s: float = 7 * 86400) -> int:
        """清理 tool-results 目录中超过 max_age_s 秒的临时文件。

        目录不存在/无权限/单文件删除失败时均静默跳过，不影响正常执行。
        返回删除的文件数。directory 参数便于测试注入临时目录。
        """
        d = Path(directory) if directory else Path.home() / ".otter-code" / "tool-results"
        if not d.is_dir():
            return 0
        cutoff = time.time() - max_age_s
        removed = 0
        try:
            for f in d.iterdir():
                try:
                    if f.is_file() and f.stat().st_mtime < cutoff:
                        f.unlink()
                        removed += 1
                except Exception:
                    continue
        except Exception:
            pass
        return removed

    def _persist_large_result(self, tool_name: str, result: str) -> str:
        THRESHOLD = 30 * 1024  # 30 KB
        #转换成字节
        if (len (result.encode())) <= THRESHOLD:
            return result

        # 首次写入前清理 7 天前的陈旧临时文件，避免目录无限膨胀。
        if not self._tool_results_cleaned:
            self._cleanup_stale_tool_results()
            self._tool_results_cleaned = True

        d = Path.home() / ".otter-code" / "tool-results"
        d.mkdir(parents=True, exist_ok=True)
        filename = f"{int(time.time() * 1000)}-{tool_name}.txt"
        filepath = d / filename
        filepath.write_text(result, encoding="utf-8")

        lines = result.split("\n")
        preview = "\n".join(lines[:200])
        size_kb = len(result.encode()) / 1024

        return (
            f"[Result too large ({size_kb:.1f} KB, {len(lines)} lines). "
            f"Full output saved to {filepath}. "
            f"You can use read_file to see the full result.]\n\n"
            f"Preview (first 200 lines):\n{preview}"
        )

    #执行工具入口

    # ─── 三层验证（L1 存在性 / L2 正确性 / L3 业务状态）─────────
    def _load_active_verification_rules(self) -> list:
        """组合验证规则：配置文件声明规则 + 本轮写过的产物自动 L1 规则。"""
        rules = load_verification_rules()
        rules += collect_written_file_rules(self._written_files, root=Path.cwd())
        return rules

    def _inject_user_feedback(self, text: str) -> None:
        """把验证失败反馈作为 user 消息注入对话，驱动模型继续修复。"""
        text = _safe_utf8_text(text)
        if self.use_openai:
            self._openai_messages.append({"role": "user", "content": text})
        else:
            self._anthropic_messages.append({"role": "user", "content": text})

    def _inject_midloop_feedback(self, text: str) -> None:
        """中途检查点的反馈注入：OpenAI 直接追加 user 消息；
        Anthropic 合并进最后一条 user 消息（tool_result 消息），
        避免连续两条 user 消息导致 Anthropic API 报错。"""
        text = _safe_utf8_text(text)
        if self.use_openai:
            self._openai_messages.append({"role": "user", "content": text})
            return
        if self._anthropic_messages and self._anthropic_messages[-1].get("role") == "user":
            last = self._anthropic_messages[-1]
            content = last.get("content")
            if isinstance(content, list):
                content.append({"type": "text", "text": text})
            else:
                last["content"] = (content or "") + "\n\n" + text
        else:
            self._anthropic_messages.append({"role": "user", "content": text})

    def _append_user_text(self, text: str) -> None:
        """向当前协议的历史追加 user 文本；最后一条已是 user 时合并
        （list 追加 text block / str 拼接）而非新增一条，避免连续两条
        user 消息违反角色交替（参照 _inject_midloop_feedback 写法）。"""
        text = _safe_utf8_text(text)
        if self.use_openai:
            _append_user_text_merged(self._openai_messages, text)
        else:
            _append_user_text_merged(self._anthropic_messages, text)

    def _checkpoint_interval(self) -> int:
        """中途 L1 检查点的工具调用间隔（OTTER_VERIFY_CHECKPOINT_EVERY，默认 5，最小 1）。"""
        try:
            n = int(os.environ.get("OTTER_VERIFY_CHECKPOINT_EVERY", "5"))
            return n if n >= 1 else 5
        except Exception:
            return 5

    async def _run_checkpoint_verification(self) -> None:
        """中途 L1 检查点：只对主代理生效，只跑本轮写产物的 L1 规则；
        失败时注入修复反馈，但不终止循环（轻量、容错）。"""
        if self.is_sub_agent or self.permission_mode == "plan":
            return
        rules = collect_written_file_rules(self._written_files, root=Path.cwd())
        if not rules:
            return
        report = run_verification(rules, cwd=Path.cwd())
        if report["passed"]:
            return
        print_verification(report)
        self._inject_midloop_feedback(
            format_verification_feedback(report, attempt=0, max_attempts=get_max_verification_attempts())
        )

    def _check_timeout(self) -> bool:
        """wall-clock 超时判断：max_duration_s 不为 None 且已运行超过该时长时返回 True。"""
        if self.max_duration_s is None:
            return False
        return time.monotonic() - self._start_time > self.max_duration_s

    async def _verify_before_done(self) -> bool:
        """模型声称完成（不再调用工具）时的验证检查点。

        返回 True 表示可以结束本轮：无规则 / 全部通过 / 重试耗尽放行；
        返回 False 表示验证失败，已注入修复反馈，循环应继续。
        plan 模式只读、无产物，直接放行；子代理不再短路，改为只跑 L1 存在性验证
        （L2/L3 需要完整执行环境，不适合在子代理内部运行）。
        """
        if self.permission_mode == "plan":
            self._last_verification_passed = True
            return True
        rules = self._load_active_verification_rules()
        if self.is_sub_agent:
            rules = _filter_l1_rules(rules)
        if not rules:
            self._last_verification_passed = True
            return True
        report = run_verification(rules, cwd=Path.cwd())
        print_verification(report)
        self._verification_log.append({
            "attempt": len(self._verification_log) + 1,
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "passed": report["passed"],
            "total": report["total"],
            "failures": report["failures"],
        })
        if report["passed"]:
            self._last_verification_passed = True
            return True
        attempt = len(self._verification_log)
        max_attempts = get_max_verification_attempts()
        if attempt >= max_attempts:
            print_error(f"Verification failed after {attempt} attempts; releasing turn (marked unverified).")
            self._last_verification_passed = False
            return True
        print_info(f"Verification failed (attempt {attempt}/{max_attempts}); feeding feedback back to model.")
        self._last_verification_passed = False
        self._inject_user_feedback(format_verification_feedback(report, attempt, max_attempts))
        return False

    async def _run_verification_tool(self, inp: dict) -> str:
        """显式验证工具：按配置规则（可过滤 rule_ids）运行验证并返回结构化报告。"""
        rules = self._load_active_verification_rules()
        rule_ids = inp.get("rule_ids")
        if rule_ids:
            ids = set(rule_ids)
            rules = [r for r in rules if r.id in ids]
        if not rules:
            return "No verification rules configured or matching the requested rule_ids."
        report = run_verification(rules, cwd=Path.cwd())
        print_verification(report)
        return json.dumps(report, ensure_ascii=False, indent=2)

    async def _execute_tool_call(self, name: str, inp: dict) -> str:
        if name in ("enter_plan_mode", "exit_plan_mode"):
            return await self._execute_plan_mode_tool(name)
        if name == "run_verification":
            return await self._run_verification_tool(inp)
        if name == "agent":
            return await self._execute_agent_tool(inp)
        if name == "skill":
            return await self._execute_skill_tool(inp)
            # Route MCP tool calls to the MCP manager
        if self._mcp_manager.is_mcp_tool(name):
            return await self._mcp_manager.call_tool(name, inp)
        result = await execute_tool(name, inp, self._read_file_state)
        # execute_tool 现返回 {"content": str, "error": str|None, "retryable": bool}；
        # 兼容旧契约：外部 mock / 旧调用方仍可能返回 str（如测试桩），直接透传。
        if name in {"skill_create", "skill_evolve"}:
            raw = result["content"] if isinstance(result, dict) else result
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and parsed.get("ok"):
                    self._refresh_runtime_system_prompt()
            except Exception:
                pass
        if name in {"write_file", "edit_file"}:
            # 记录本轮声明写入的产物路径，供自动 L1 存在性验证使用。
            path = inp.get("file_path")
            if path:
                self._written_files.add(path)
        if isinstance(result, str):
            return result
        return result["content"] or result["error"] or ""


    async def _execute_skill_tool(self, inp: dict) -> str:
        from .skills import execute_skill
        result = execute_skill(inp.get("skill_name", ""), inp.get("args", ""))

        if not result:
            return f"Unknown skill: {inp.get('skill_name', '')}"

        #fork 表示这个 skill 不直接把 prompt 塞回当前对话，而是要启动一个子 Agent 单独完成任务。
        if result["context"] == "fork":
            # result["allowed_tools"] - 直接访问
            tools = (
                [t for t in self.tools if t["name"] in  result["allowed_tools"] ]
                #result.get("allowed_tools") - 安全访问
                # 存在key：返回对应的值（可能是 None、[]、["tool1"] 等）
                # 不存在key：返回 None（不会抛异常）
                if result.get("allowed_tools")
                else  [t for t in self.tools if t["name"] != "agent"]
            )

            print_sub_agent_start("skill-fork", inp.get("skill_name", ""))
            sub_agent = Agent(
                model=self.model,
                api_base=str(self._openai_client.base_url) if self.use_openai and self._openai_client else None,
                custom_system_prompt=result["prompt"],
                custom_tools=tools,
                is_sub_agent=True,
                permission_mode=self._sub_agent_permission_mode(),
            )
            try:
                sub_result = await sub_agent.run_once(inp.get("args") or "Execute this skill task.")
                self.total_input_tokens += sub_result["tokens"]["input"]
                self.total_output_tokens += sub_result["tokens"]["output"]
                print_sub_agent_end("skill-fork", inp.get("skill_name", ""))
                text = sub_result["text"] or "(Skill produced no output)"
                return self._append_unverified_marker(text, sub_result.get("verified", True))
            except Exception as e:
                print_sub_agent_end("skill-fork", inp.get("skill_name", ""))
                return f"Skill fork error: {e}"

        return f'[Skill "{inp.get("skill_name", "")}" activated]\n\n{result["prompt"]}'

    async def _execute_plan_mode_tool(self, name):
        if name == "enter_plan_mode":
            if self.permission_mode == "plan":
                return "Already in plan mode."
            self._pre_plan_mode = self.permission_mode
            self.permission_mode = "plan"
            self._plan_file_path =  self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt
            print_info("Entered plan mode (read-only). Plan file: " + self._plan_file_path)
            return f"Entered plan mode. You are now in read-only mode.\n\nYour plan file: {self._plan_file_path}\nWrite your plan to this file. This is the only file you can edit.\n\nWhen your plan is complete, call exit_plan_mode."
        if name == "exit_plan_mode":
            if self.permission_mode != "plan":
                return "Not in plan mode."
            plan_content = "(No plan file found)"
            if self._plan_file_path and Path(self._plan_file_path).exists():
                plan_content = self._plan_file_path
            # 交互式审批流程（如果有审批函数）
            if self._plan_approval_fn:
                result = self._plan_approval_fn(plan_content)
                choice = result.get("choice", "manual-execute")

                if choice =="keep-planning":
                    feedback = result.get("feedback") or "Please revise the plan."
                    return (
                        f"User rejected the plan and wants to keep planning.\n\n"
                        f"User feedback: {feedback}\n\n"
                        f"Please revise your plan based on this feedback. When done, call exit_plan_mode again."
                    )

                if choice == "clear-and-execute":
                    target_mode = "acceptEdits"
                elif choice == "execute":
                    target_mode = "acceptEdits"
                else:  # manual-execute
                    target_mode = self._pre_plan_mode or "default"

                #离开计划模式：切换到目标权限，并清空进入前保存的模式
                self.permission_mode = target_mode
                self._pre_plan_mode = None
                saved_plan_path = self._plan_file_path
                self._plan_file_path = None
                self._system_prompt = self._base_system_prompt
                if self.use_openai and self._openai_messages:
                    self._openai_messages[0]["content"] = self._system_prompt

                if choice == "clear-and-execute":
                    self._clear_history_keep_system()
                    self._context_cleared = True
                    print_info(f"Plan approved. Context cleared, executing in {target_mode} mode.")
                    return (
                        f"User approved the plan. Context was cleared. Permission mode: {target_mode}\n\n"
                        f"Plan file: {saved_plan_path}\n\n"
                        f"## Approved Plan:\n{plan_content}\n\n"
                        f"Proceed with implementation."
                    )
                print_info(f"Plan approved. Executing in {target_mode} mode.")
                return (
                    f"User approved the plan. Permission mode: {target_mode}\n\n"
                    f"## Approved Plan:\n{plan_content}\n\n"
                    f"Proceed with implementation."
                )
            # 没有审批函数时的回退（例如子代理）
            self.permission_mode = self._pre_plan_mode or "default"
            self._pre_plan_mode = None
            self._plan_file_path = None
            self._system_prompt = self._base_system_prompt
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt

            print_info("Exited plan mode. Restored to " + self.permission_mode + " mode.")
            return f"Exited plan mode. Permission mode restored to: {self.permission_mode}\n\n## Your Plan:\n{plan_content}"

        return f"Unknown plan mode tool: {name}"

    def _clear_history_keep_system(self) -> None:
        """清空历史信息，但是保留系统prompt."""
        self._anthropic_messages = []
        self._openai_messages = []
        if self.use_openai:
            self._openai_messages.append({"role": "system", "content": self._system_prompt})
        self.last_input_token_count = 0

    def _sub_agent_permission_mode(self) -> str:
        """子代理权限模式：plan 主代理派生的子代理保持只读 plan；
        其余一律 acceptEdits —— 编辑类工具自动放行，危险 shell 仍走确认
        （子代理 confirm_fn 为 None 时会被 _confirm_dangerous 拒绝，符合预期）。
        """
        return "plan" if self.permission_mode == "plan" else "acceptEdits"

    @staticmethod
    def _append_unverified_marker(text: str, verified: bool) -> str:
        """子代理产物未通过 L1 验证时，在返回文本末尾附加 [unverified] 标记，
        让调用方（主代理）知道结果不可信，避免把未经验证的产物当作完成。"""
        if verified is False:
            return f"{text}\n\n[unverified] 子代理产物未通过 L1 验证"
        return text

    async def _execute_agent_tool(self, inp:dict) -> str:
        agent_type = inp.get("type", "general")
        description = inp.get("description", "sub-agent task")
        prompt = inp.get("prompt", "")
        print_sub_agent_start(agent_type, description)

        config = get_sub_agent_config(agent_type)

        sub_agent = Agent(
            model=self.model,
            api_base=str(self._openai_client.base_url) if self.use_openai and self._openai_client else None,
            custom_system_prompt=config["system_prompt"],
            custom_tools=config["tools"],
            is_sub_agent=True,
            permission_mode=self._sub_agent_permission_mode(),
        )
        try:
            result = await sub_agent.run_once(prompt)
            self.total_input_tokens += result["tokens"]["input"]
            self.total_output_tokens += result["tokens"]["output"]
            print_sub_agent_end(agent_type, description)
            text = result["text"] or "(Sub-agent produced no output)"
            return self._append_unverified_marker(text, result.get("verified", True))
        except Exception as e:
            print_sub_agent_end(agent_type, description)
            return f"Sub-agent error: {e}"

#--------------Anthropic 后端---------------
    async def  _chat_anthropic(self, user_message: str) -> None:
        self._anthropic_messages = self._normalize_anthropic_messages(_sanitize_for_utf8(self._anthropic_messages))
        user_message = _safe_utf8_text(user_message)
        # 先把本轮用户输入放入 Anthropic 消息历史，后续每轮模型调用都会带上这段上下文。
        # 最后一条已是 user（如 budget 分支的 tool_result 结尾）时合并而非新增，
        # 避免连续两条 user 消息导致 Anthropic API 报错。
        self._append_user_text(user_message)

        # 异步内存预取：主 agent 才需要查 memory，sub agent 不额外注入记忆。
        # 这里只启动后台任务，不阻塞当前模型调用流程。
        memory_prefetch:MemoryPrefetch | None = None
        if not self.is_sub_agent:
            sq = self._build_side_query()
            if sq:
                memory_prefetch = start_memory_prefetch(
                    user_message, sq,
                    self._already_surfaced_memories, self._session_memory_bytes,
                )
        while True:
            # 外部请求中止时，结束整个 agent loop。
            if self._aborted:
                break

            # wall-clock 超时（--max-duration）：优雅中止，不抛异常。
            if self._check_timeout():
                self._timed_out = True
                print_error("Max duration reached; stopping agent loop gracefully.")
                break

            # 每轮调用模型前尝试压缩上下文，避免消息历史过长。
            self._run_compression_pipeline()

            # 发送前 token 预估算：若预估总量超过窗口阈值，提前触发摘要压缩。
            # （_check_and_compact 的 85% 兜底逻辑保留，二者双保险。）
            if self.estimate_messages_tokens() > self.effective_window - _get_max_output_tokens(self.model) - 1024:
                await self._compact_conversation()

            # 如果记忆预取任务已经完成，就把取回来的 memory 内容追加到最后一条用户消息里。
            # consumed 用来保证同一批 memory 只注入一次。
            if memory_prefetch and memory_prefetch.settled and not memory_prefetch.consumed:
                memory_prefetch.consumed = True
                try:
                    memories = memory_prefetch.task.result()
                    if memories:
                        injection_text = format_memories_for_injection(memories)
                        injection_text = _safe_utf8_text(injection_text)
                        last = self._anthropic_messages[-1] if self._anthropic_messages else None
                        if last and last.get("role") == "user":
                            content = last.get("content", "")
                            if isinstance(content, str):
                                # 字符串不可变，需要重新赋值回 message。
                                last["content"] = content + "\n\n" + injection_text
                            elif isinstance(content, list):
                                # list 是可变对象，append 会直接修改 last["content"] 指向的列表。
                                content.append({"type": "text", "text": injection_text})
                        else:
                            # 如果最后一条不是 user message，就单独追加一条用户消息承载 memory。
                            self._anthropic_messages.append({"role": "user", "content": injection_text})

                        for m in memories:
                            # 记录本 session 已经注入过的 memory，后续检索时可避免重复 surfaced。
                            self._already_surfaced_memories.add(m.path)
                            self._session_memory_bytes += m.size
                except:
                    # memory 注入失败不应该中断主对话流程。
                    pass

            if not self.is_sub_agent:
                start_spinner()


            # 保存“提前执行”的工具任务。key 是 Anthropic 返回的 tool_use block id。
            early_executions: dict[str, asyncio.Task] = {}


            def _on_tool_block(block:dict):
                # 流式响应中一旦完整收到 tool_use block，如果工具是并发安全且权限允许，
                # 就可以提前开始执行，减少等待完整模型响应后的空档时间。
                # 同轮重复触发同一 id（网关重放 content_block_stop）直接忽略，
                # 避免创建第二个 early task 导致重复执行。
                if block["id"] in early_executions:
                    return
                if block["name"] in CONCURRENCY_SAFE_TOOLS:
                    perm = check_permission(block["name"], block["input"], self.permission_mode, self._plan_file_path)
                    if perm["action"]=="allow":
                        task =asyncio.create_task(self._execute_tool_call(block["name"], block["input"]))
                        early_executions[block["id"]] = task


            # 调用 Anthropic 流式接口；流式过程中完成 tool block 时会触发 _on_tool_block。
            response = await self._call_anthropic_stream(on_tool_block_complete=_on_tool_block)
            if not self.is_sub_agent:
                stop_spinner()

            # 记录本次模型调用的耗时点和 token 消耗，用于成本展示与预算控制。
            self.last_api_call_time = time.time()
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            self.last_input_token_count = response.usage.input_tokens

            # Anthropic 的响应内容里可能混有 text block 和 tool_use block，这里只挑出工具调用。
            tool_uses = [b for b in response.content if b.type == "tool_use"]

            # 把模型返回的所有 content block 写入消息历史，后续 tool_result 要与这些 tool_use 对应。
            # 网关重放重复 id 时按 id 去重（保留首个），保证 tool_result 回填一一配对。
            seen_tool_use_ids: set[str] = set()
            content_blocks: list[dict] = []
            for b in response.content:
                if b.type == "tool_use":
                    if b.id in seen_tool_use_ids:
                        continue
                    seen_tool_use_ids.add(b.id)
                content_blocks.append(self._block_to_dict(b))
            self._anthropic_messages.append({"role": "assistant", "content": content_blocks})

            # 没有工具调用，说明模型认为已经完成。进入三层验证检查点：
            # 全部通过才结束；失败则注入反馈让模型修复（fix loop，限次）。
            if not tool_uses:
                if not self.is_sub_agent:
                    print_cost(self.total_input_tokens, self.total_output_tokens)
                if await self._verify_before_done():
                    break
                continue

            # 有工具调用时，进入下一轮工具执行。这里同时检查 turn/budget 限制。
            self.current_turns += 1
            budget = self._check_budget()
            if budget["exceeded"]:
                print_info(f"Budget exceeded: {budget['reason']}")
                self._anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": f"Tool execution skipped: {budget['reason']}",
                        }
                        for tu in tool_uses
                    ],
                })
                break


            # 收集本轮所有工具结果，之后作为 tool_result 消息回传给模型。
            tool_results: list[dict] = []
            context_break = False
            executed_tool_use_ids: set[str] = set()

            for tu in tool_uses:
                # context_break 表示某个工具执行期间清理了上下文，需要停止继续处理本轮剩余工具。
                if context_break or self._aborted:
                    break

                # 执行层按 id 去重：同一轮重复 id 只执行一次、只回填一次（保留首个）。
                if tu.id in executed_tool_use_ids:
                    continue
                executed_tool_use_ids.add(tu.id)

                # 将工具入参转为普通 dict，便于权限检查、打印和实际执行。
                inp = dict(tu.input) if hasattr(tu, "items") else tu.input
                print_tool_call(tu.name, inp)

                # 如果这个工具已经在流式阶段提前开始执行，这里只需要等待它完成并收集结果。
                early_task = early_executions.get(tu.id)
                if early_task:
                    try:
                        raw = await early_task
                    except Exception as e:
                        raw = f"Error executing tool: {e}"
                    raw = _safe_utf8_text(raw)
                    res = self._persist_large_result(tu.name, raw)
                    print_tool_result(tu.name, res)
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": res})
                    continue

                # 如果不是提前执行的工具，就在真正执行前做权限检查。

                perm = check_permission(tu.name, inp, self.permission_mode, self._plan_file_path)
                if perm["action"] == "deny":
                    # 权限拒绝时，也要返回一个 tool_result，让模型知道该工具调用失败的原因。
                    print_info(f"Denied: {perm.get('message', '')}")
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id,
                                         "content": f"Action denied: {perm.get('message', '')}"})
                    continue

                if perm["action"] == "confirm" and perm.get("message") and perm["message"] not in self._confirmed_paths:
                    # 高风险操作需要用户确认；同一个 message 确认过后会缓存，避免重复询问。
                    confirmed = await self._confirm_dangerous(perm["message"])
                    if not confirmed:
                        tool_results.append(
                            {"type": "tool_result", "tool_use_id": tu.id, "content": "User denied this action."})
                        continue
                    self._confirmed_paths.add(perm["message"])

                # 权限通过后执行工具，并把大输出持久化为可回传的摘要或引用。
                try:
                    raw = await self._execute_tool_call(tu.name, inp)
                except Exception as e:
                    raw = f"Error executing tool: {e}"
                raw = _safe_utf8_text(raw)
                res = self._persist_large_result(tu.name, raw)
                print_tool_result(tu.name, res)

                if self._context_cleared:
                    # 工具执行过程中如果清理了上下文，就把结果作为新的用户消息写入，
                    # 并停止继续处理本轮剩余工具，避免旧上下文和新上下文混在一起。
                    self._context_cleared = False
                    self._anthropic_messages.append({"role": "user", "content": res})
                    context_break = True
                    break

                # Anthropic 要求 tool_result 使用 tool_use_id 对应到前面的 tool_use block。
                tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": res})

            if not context_break and tool_results:
                # Anthropic 要求 assistant/tool_use 后面紧跟一条 user/tool_result 消息，
                # 且这条消息必须包含本轮所有 tool_use 的对应结果。
                self._anthropic_messages.append({"role": "user", "content": tool_results})

            self._context_cleared = False

            # 中途 L1 检查点：每 N 次工具调用后快速校验写产物存在性（仅主代理生效）。
            interval = self._checkpoint_interval()
            if self._tool_call_count > 0 and self._tool_call_count % interval == 0:
                await self._run_checkpoint_verification()

            # 工具结果可能很长，每轮工具执行后检查是否需要压缩上下文。
            await self._check_and_compact()

        # 回收未消费的 memory 预取任务，避免 asyncio task 泄漏（已 consumed 的跳过）。
        if memory_prefetch and not memory_prefetch.consumed:
            try:
                memory_prefetch.task.cancel()
            except Exception:
                pass

    @staticmethod
    def _block_to_dict(block) -> dict:
        if block.type == "text":
            return {"type": "text", "text": _safe_utf8_text(block.text)}
        if block.type == "tool_use":
            raw_input = dict(block.input) if hasattr(block.input, 'items') else block.input
            return {"type": "tool_use", "id": _safe_utf8_text(block.id), "name": _safe_utf8_text(block.name), "input": _sanitize_for_utf8(raw_input)}
        # Fallback
        return {"type": _safe_utf8_text(block.type)}

    async def _call_anthropic_stream(self, on_tool_block_complete=None):

        async def _do():
            max_output =  _get_max_output_tokens(self.model)

            # Prompt caching：system 改为带 cache_control 的 block 列表；
            # tools 在深拷贝的最后一项上追加 cache_control，不修改共享的 self.tools 结构。
            tools_defs = _sanitize_for_utf8(get_active_tool_definitions(self.tools))
            if tools_defs:
                tools_with_cache = list(tools_defs)
                last_tool = copy.deepcopy(tools_with_cache[-1])
                last_tool["cache_control"] = {"type": "ephemeral"}
                tools_with_cache[-1] = last_tool
            else:
                tools_with_cache = tools_defs

            create_params: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_output if self._thinking_mode != "disabled" else 16384,
                "system": [{"type": "text", "text": _safe_utf8_text(self._system_prompt), "cache_control": {"type": "ephemeral"}}],
                "tools": tools_with_cache,
                "messages": _sanitize_for_utf8(self._anthropic_messages),
            }
            #如果开启了思考模式，就给 Anthropic 请求加上 thinking 参数。
            if self._thinking_mode  in ("adaptive", "enabled"):
                create_params["thinking"]={"type": "enabled", "budget_tokens": max_output - 1}

            first_text = True

            tool_blocks_by_index: dict[int, dict] = {}

            async with self._anthropic_client.messages.stream(**create_params)as stream:
                async for event in stream:
                    if not hasattr(event, 'type'):
                        continue
                    # 当事件是工具调用开始：
                    if event.type == "content_block_start":
                        cb = getattr(event, 'content_block', None)
                        #如果 block 类型是 tool_use，就记录这个工具调用：
                        if cb and getattr(cb, 'type', None) == "tool_use":
                            #因为工具参数 JSON 是流式分片返回的，所以先准备一个空字符串 input_json。
                            tool_blocks_by_index[event.index]= {
                                "id": cb.id, "name": cb.name, "input_json": "",
                            }
                    #当事件是内容增量，分三种情况。
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        # 第一种，普通文本：模型输出正文时，
                        # 调用 _emit_text()。如果是普通交互，就打印；
                        # 如果是 run_once()，就写入 _output_buffer。
                        if hasattr(delta, "text"):
                            if first_text:
                                stop_spinner()
                                self._emit_text("\n")
                                first_text = False
                            self._emit_text(delta.text)
                        #第二种，thinking 内容：
                        #如果模型返回思考内容，也输出出来，并在开头加：[thinking]
                        elif hasattr(delta, 'thinking'):
                            if first_text:
                                stop_spinner()
                                self._emit_text("\n  [thinking] ")
                                first_text = False
                            self._emit_text(delta.thinking)
                        #第三种，工具参数 JSON 片段：工具调用的参数不是一次性返回，
                        # 而是一段一段返回，所以这里不断拼接到 input_json。
                        elif hasattr(delta, 'partial_json'):
                            tb = tool_blocks_by_index.get(event.index)
                            if tb:
                                tb["input_json"] += _safe_utf8_text(delta.partial_json)
                    #当一个 content block 结束：
                    #如果结束的是之前记录的工具调用，就把拼好的 JSON 解析出来：
                    elif event.type == "content_block_stop":
                        tb = tool_blocks_by_index.pop(event.index, None)
                        if tb and on_tool_block_complete:
                            import json as _json
                            try:
                                parsed = _json.loads(tb["input_json"] or "{}")
                            except Exception:
                                parsed = {}
                            #然后调用回调：
                            #这个回调的作用通常是：工具调用一完整，
                            # 就可以提前开始执行工具，不必等整条 assistant 消息全部结束。
                            on_tool_block_complete({
                                "type": "tool_use", "id": _safe_utf8_text(tb["id"]),
                                "name": _safe_utf8_text(tb["name"]), "input": _sanitize_for_utf8(parsed),
                            })
                final_message = await stream.get_final_message()

            #过滤思考的message（因为 thinking 内容一般不应该进入历史消息，否则后续上下文会变大，也可能不符合 API 消息格式要求。）
            final_message.content = [b for b in final_message.content if b.type != "thinking"]
            return final_message
#调用 _do()，如果遇到可重试错误，就由 _with_retry() 负责重试。
        return await _with_retry(_do)

    #openAI后端

    async def _chat_openai(self, user_message:str) -> None:
        user_message = _safe_utf8_text(user_message)
        # 最后一条已是 user 时合并而非新增，避免连续两条 user 消息。
        self._append_user_text(user_message)

        #预取句柄 MemoryPrefetch
        memory_prefetch: MemoryPrefetch | None = None
        if not self.is_sub_agent:
            sq = self._build_side_query()
            if sq:
                memory_prefetch = start_memory_prefetch(
                    user_message, sq,
                    self._already_surfaced_memories, self._session_memory_bytes,
                )

        while True:
            if self._aborted:
                break

            # wall-clock 超时（--max-duration）：优雅中止，不抛异常。
            if self._check_timeout():
                self._timed_out = True
                print_error("Max duration reached; stopping agent loop gracefully.")
                break

            self._run_compression_pipeline()

            # 发送前 token 预估算：若预估总量超过窗口阈值，提前触发摘要压缩。
            # （_check_and_compact 的 85% 兜底逻辑保留，二者双保险。）
            if self.estimate_messages_tokens() > self.effective_window - _get_max_output_tokens(self.model) - 1024:
                await self._compact_conversation()

            if memory_prefetch and memory_prefetch.settled and not memory_prefetch.consumed:
                memory_prefetch.consumed = True
                try:
                    memories = memory_prefetch.task.result()
                    if memories:
                        injection_text = format_memories_for_injection(memories)
                        injection_text = _safe_utf8_text(injection_text)
                        last = self._openai_messages[-1] if self._openai_messages else None

                        if last and last.get("role") == "user":
                            last["content"] = (last.get("content") or "") + "\n\n" + injection_text
                        else:
                            self._openai_messages.append({"role": "user", "content": injection_text})

                        for m in memories:
                            self._already_surfaced_memories.add(m.path)
                            self._session_memory_bytes += len(m.content.encode())
                except Exception:
                    pass

            if not self.is_sub_agent:
                start_spinner()

            response = await self._call_openai_stream()

            if not self.is_sub_agent:
                stop_spinner()

            self.last_api_call_time = time.time()

            if response.get("usage"):
                self.total_input_tokens += response["usage"]["prompt_tokens"]
                self.total_output_tokens += response["usage"]["completion_tokens"]
                self.last_input_token_count = response["usage"]["prompt_tokens"]

            choice = response.get("choices", [{}])[0] if response.get("choices") else {}
            message = choice.get("message", {})

            self._openai_messages.append(message)

            tool_calls = message.get("tool_calls")

            if not tool_calls:
                if not self.is_sub_agent:
                    print_cost(self.total_input_tokens, self.total_output_tokens)
                if await self._verify_before_done():
                    break
                continue

            self.current_turns += 1
            budget = self._check_budget()
            if budget["exceeded"]:
                print_info(f"Budget exceeded: {budget['reason']}")
                # 预算超限：为所有 tool_calls 回填 skipped 结果，保证历史
                # 每个 tool_call 都有对应 tool 消息（无孤儿 tool_calls，不报 400）。
                for tc in tool_calls:
                    self._openai_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "content": f"Tool execution skipped: {budget['reason']}",
                    })
                break

            # 权限收集阶段：逐个确认/拒绝，allowed/result 在收集时写入 oai_checked。
            oai_checked: list[dict] = []
            seen_tc_ids: set[str] = set()
            for tc in tool_calls:
                if self._aborted:
                    break

                if tc.get("type") != "function":
                    continue

                # 执行层按 id 去重：同一轮重复 id 只执行一次、只回填一次（保留首个）。
                tc_id = tc.get("id")
                if tc_id is not None:
                    if tc_id in seen_tc_ids:
                        continue
                    seen_tc_ids.add(tc_id)

                fn_name = tc["function"]["name"]
                try:
                    inp = json.loads(tc["function"]["arguments"])
                except Exception:
                    inp = {}

                print_tool_call(fn_name, inp)

                perm = check_permission(fn_name, inp, self.permission_mode, self._plan_file_path)

                if perm["action"] == "deny":
                    print_info(f"Denied: {perm.get('message', '')}")
                    oai_checked.append({"tc": tc, "fn": fn_name, "inp": inp, "allowed": False,
                                        "result": f"Action denied: {perm.get('message', '')}"})
                    continue
                if perm["action"] == "confirm" and perm.get("message") and perm["message"] not in self._confirmed_paths:
                    confirmed = await self._confirm_dangerous(perm["message"])
                    if not confirmed:
                        oai_checked.append({"tc": tc, "fn": fn_name, "inp": inp, "allowed": False,
                                            "result": "User denied this action."})
                        continue
                    self._confirmed_paths.add(perm["message"])
                oai_checked.append({"tc": tc, "fn": fn_name, "inp": inp, "allowed": True})

            # 批次构建阶段：按 CONCURRENCY_SAFE_TOOLS 把连续并发安全的工具分组。
            oai_batches: list[dict] = []
            for ct in oai_checked:
                safe = ct["allowed"] and ct["fn"] in CONCURRENCY_SAFE_TOOLS
                if safe and oai_batches and oai_batches[-1]["concurrent"]:
                    oai_batches[-1]["items"].append(ct)
                else:
                    oai_batches.append({"concurrent": safe, "items": [ct]})

            # 批次执行阶段：每个工具只执行一次、每条 tool 消息只回写一次。
            oai_context_break = False
            for batch in oai_batches:
                if oai_context_break or self._aborted:
                    break

                if batch["concurrent"]:
                    async def _run_oai_safe(ct_item: dict) -> tuple[dict, str]:
                        try:
                            raw = await self._execute_tool_call(ct_item["fn"], ct_item["inp"])
                        except Exception as e:
                            # 单个工具异常不中断整轮：回填错误消息（与 Anthropic 路径对齐）。
                            raw = f"Error executing tool: {e}"
                        raw = _safe_utf8_text(raw)
                        res = self._persist_large_result(ct_item["fn"], raw)
                        print_tool_result(ct_item["fn"], res)
                        return ct_item, res

                    results = await asyncio.gather(
                        *[_run_oai_safe(ct) for ct in batch["items"]],
                        return_exceptions=True,
                    )
                    # gather 保序返回（每项即 _run_oai_safe 的 (ct_item, res) 元组），
                    # 逐个处理；异常项按索引回填错误，不中断整轮。
                    for index, result in enumerate(results):
                        if isinstance(result, Exception):
                            # 极端兜底：任务级异常（正常不会走到）同样回填错误。
                            self._openai_messages.append(
                                {"role": "tool", "tool_call_id": batch["items"][index]["tc"]["id"],
                                 "content": f"Error executing tool: {result}"})
                            continue
                        ct_item, res = result
                        self._openai_messages.append(
                            {"role": "tool", "tool_call_id": ct_item["tc"]["id"], "content": res})
                else:
                    for ct in batch["items"]:
                        if not ct["allowed"]:
                            self._openai_messages.append(
                                {"role": "tool", "tool_call_id": ct["tc"]["id"], "content": ct["result"]})
                            continue

                        try:
                            raw = await self._execute_tool_call(ct["fn"], ct["inp"])
                        except Exception as e:
                            raw = f"Error executing tool: {e}"
                        raw = _safe_utf8_text(raw)
                        res = self._persist_large_result(ct["fn"], raw)
                        print_tool_result(ct["fn"], res)

                        if self._context_cleared:
                            self._context_cleared = False
                            self._openai_messages.append({"role": "user", "content": res})
                            oai_context_break = True
                            break

                        self._openai_messages.append(
                            {"role": "tool", "tool_call_id": ct["tc"]["id"], "content": res})

            self._context_cleared = False

            # 中途 L1 检查点：每 N 次工具调用后快速校验写产物存在性（仅主代理生效）。
            interval = self._checkpoint_interval()
            if self._tool_call_count > 0 and self._tool_call_count % interval == 0:
                await self._run_checkpoint_verification()

            await self._check_and_compact()

        # 回收未消费的 memory 预取任务，避免 asyncio task 泄漏（已 consumed 的跳过）。
        if memory_prefetch and not memory_prefetch.consumed:
            try:
                memory_prefetch.task.cancel()
            except Exception:
                pass

    async def _call_openai_stream(self) -> dict:
        async def _do():
            stream = await self._openai_client.chat.completions.create(
                model=self.model,
                tools=_sanitize_for_utf8(_to_openai_tools(get_active_tool_definitions(self.tools))),
                messages=_sanitize_for_utf8(_dedupe_openai_messages(self._openai_messages)),
                stream=True,
                stream_options={"include_usage": True},
            )

            content = ""
            first_text = True
            tool_calls: dict[int, dict] = {}
            finish_reason = ""
            usage = None

            async for chunk in stream:
                if chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                    }

                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta and delta.content:
                    if first_text:
                        stop_spinner()
                        self._emit_text("\n")
                        first_text = False
                    self._emit_text(delta.content)
                    content += _safe_utf8_text(delta.content)

                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        existing = tool_calls.get(tc.index)
                        if existing:
                            if tc.function and tc.function.arguments:
                                existing["arguments"] += _safe_utf8_text(tc.function.arguments)
                        else:
                            tool_calls[tc.index] = {
                                "id": _safe_utf8_text(tc.id or ""),
                                "name": _safe_utf8_text((tc.function.name if tc.function else "") or ""),
                                "arguments": _safe_utf8_text((tc.function.arguments if tc.function else "") or ""),
                            }

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            assembled = None
            if tool_calls:
                assembled = [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for _, tc in sorted(tool_calls.items())
                ]

            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": assembled,
                    },
                    "finish_reason": finish_reason or "stop",
                }],
                "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0},
            }

        return await _with_retry(_do)

    async def _confirm_dangerous(self, command: str) -> bool:
        print_confirmation(command)
        if self.confirm_fn:
            return await self.confirm_fn(command)
        # 子代理没有交互式确认通道：confirm_fn 为 None 时直接拒绝，
        # 避免在后台/子代理流程中阻塞等待用户输入。
        if self.is_sub_agent:
            return False
        # Fallback: blocking input
        try:
            answer = input("  Allow? (y/n): ")
            return answer.lower().startswith("y")
        except EOFError:
            return False
