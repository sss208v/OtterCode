"""
文件型记忆系统。

核心思路：
1. 每个项目有独立的 memory 目录，目录名由当前工作目录 hash 得到。
2. 每条记忆都是一个 Markdown 文件，文件头部用 YAML frontmatter 保存元信息。
3. MEMORY.md 是自动生成的索引，给 system prompt 快速展示已有记忆。
4. 对话时先轻量扫描记忆文件头，再用 side query 让模型挑出相关记忆。
5. 召回到的记忆会以 <system-reminder> 形式注入当前对话。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .frontmatter import parse_frontmatter, format_frontmatter
from typing import Callable
# side query 是一个异步函数：输入 system prompt 和 user prompt，返回模型文本。
# 这里标成 Any 是为了避免在运行时引入复杂 Awaitable 类型约束。
SideQueryFn = Callable[[str, str], Any]  # actually Awaitable[str]


VALID_TYPES = {"user", "feedback", "project", "reference"}
# BM25 参数统一常量：与 skills.py 检索共用，保证两处算法参数一致（k1 取 BM25 经典范围 1.2-2.0）。
BM25_K1 = 1.5
BM25_B = 0.75
BM25_TOP_K = 15             # BM25 预筛候选数。
MAX_INDEX_LINES = 200       # MEMORY.md 注入 system prompt 前最多保留的行数。
MAX_INDEX_BYTES = 25000     # MEMORY.md 注入 system prompt 前最多保留的字节数。
MAX_SUMMARY_INDEX_ENTRIES = 5   # Task 17：conversation-compact-summary 摘要记忆进索引的最多条数。


class MemoryEntry:
    """完整 memory 条目，用于 /memory 列表和 CRUD 操作。"""

    __slots__ = ("name", "description", "type", "filename", "content")

    def __init__(self, name: str, description: str, type: str, filename: str, content: str):
        self.name = name
        self.description = description
        self.type = type
        self.filename = filename
        self.content = content




def _project_hash() -> str:
    """用当前工作目录生成稳定 hash，让不同项目的记忆互相隔离。"""
    return hashlib.sha256(str(Path.cwd()).encode()).hexdigest()[:16]


def get_memory_dir() -> Path:
    """返回当前项目的 memory 目录，不存在时自动创建。"""
    d = Path.home() / ".OtterCode" / "projects" / _project_hash() / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_index_path() -> Path:
    """MEMORY.md 是当前项目 memory 文件的索引文件。"""
    return get_memory_dir() / "MEMORY.md"




def _slugify(text: str) -> str:
    """把记忆名称转成适合文件名的短 slug。"""
    s = re.sub(r"[^a-z0-9]+", "_", text.lower())
    s = s.strip("_")
    return s[:40]




def list_memories() -> list[MemoryEntry]:
    """读取当前项目所有 memory 文件，并按修改时间倒序返回。"""
    d = get_memory_dir()
    entries: list[MemoryEntry] = []
    for f in sorted(d.glob("*.md")):
        # MEMORY.md 是索引，不是一条真实记忆。
        if f.name == "MEMORY.md":
            continue
        try:
            result = parse_frontmatter(f.read_text())
            meta = result.meta
            # 没有 name/type 的文件不算合法 memory。
            if not meta.get("name") or not meta.get("type"):
                continue
            # type 不合法时降级为 project，避免坏文件中断列表。
            t = meta["type"] if meta["type"] in VALID_TYPES else "project"
            entries.append(MemoryEntry(
                name=meta["name"],
                description=meta.get("description", ""),
                type=t,
                filename=f.name,
                content=result.body,
            ))
        except Exception:
            pass
    # 最近修改的记忆排在前面，方便 /memory 展示。
    entries.sort(key=lambda e: (d / e.filename).stat().st_mtime, reverse=True)
    return entries


def save_memory(name: str, description: str, type: str, content: str) -> str:
    """保存一条 memory，并刷新 MEMORY.md 索引。"""
    d = get_memory_dir()
    filename = f"{type}_{_slugify(name)}.md"
    text = format_frontmatter({"name": name, "description": description, "type": type}, content)
    (d / filename).write_text(text)
    update_memory_index()
    return filename


def save_memory_structured(
    name: str,
    description: str,
    type: str,
    content: str,
    session_id: str | None = None,
) -> str:
    """结构化保存记忆：校验类型与字段非空，同名去重更新，写入后刷新索引。

    与 save_memory（REPL /memory 命令使用）独立，作为 memory_save 工具的入库入口。
    """
    if type not in VALID_TYPES:
        raise ValueError(f"非法记忆类型: {type}，合法类型为 {sorted(VALID_TYPES)}")
    if not name.strip():
        raise ValueError("记忆名称不能为空")
    if not content.strip():
        raise ValueError("记忆内容不能为空")

    d = get_memory_dir()
    filename = f"{type}_{_slugify(name)}.md"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    filepath = d / filename
    importance = IMPORTANCE_BASE * TYPE_IMPORTANCE.get(type, IMPORTANCE_BASE)
    if filepath.exists():
        # 同名记忆：保留原 created_at 与访问统计（access_count/last_accessed），刷新其余元数据与正文。
        old_meta = parse_frontmatter(filepath.read_text()).meta
        text = format_frontmatter({
            "name": name,
            "description": description,
            "type": type,
            "created_at": old_meta.get("created_at") or now,
            "updated_at": now,
            "source_session": session_id or "",
            "importance": importance,
            "access_count": _safe_int(old_meta.get("access_count"), 0),
            "last_accessed": old_meta.get("last_accessed") or "",
        }, content)
        filepath.write_text(text)
        update_memory_index()
        return f"updated existing: {filename}"

    # 新记忆：写入完整元数据，含衰减字段（access_count/last_accessed 初始值）。
    text = format_frontmatter({
        "name": name,
        "description": description,
        "type": type,
        "created_at": now,
        "updated_at": now,
        "source_session": session_id or "",
        "importance": importance,
        "access_count": 0,
        "last_accessed": "",
    }, content)
    filepath.write_text(text)
    update_memory_index()
    return f"saved: {filename}"


def delete_memory(filename: str) -> bool:
    """按文件名删除 memory，删除成功后刷新索引。"""
    filepath = get_memory_dir() / filename
    if not filepath.exists():
        return False
    filepath.unlink()
    update_memory_index()
    return True




def update_memory_index() -> None:
    """根据当前 memory 文件重新生成 MEMORY.md（临时文件 + os.replace 原子写）。

    Task 6：文件数超过 MAX_MEMORY_FILES 时先淘汰 importance 最低的记忆，
    保证索引反映淘汰后的状态。

    Task 17：conversation-compact-summary 前缀的会话摘要记忆（name 由
    agent.py 写作 conversation-compact-summary-<时间戳>）进索引只保留最近
    MAX_SUMMARY_INDEX_ENTRIES 条（按 mtime 倒序取最新），避免索引随会话
    压缩无限增长；其余记忆条目全部保留。
    """
    _evict_low_importance_memories(MAX_MEMORY_FILES)
    memories = list_memories()
    # Task 17：摘要记忆单独收集，非摘要记忆不受影响。
    summaries = [m for m in memories if m.name.startswith("conversation-compact-summary")]
    others = [m for m in memories if not m.name.startswith("conversation-compact-summary")]
    if len(summaries) > MAX_SUMMARY_INDEX_ENTRIES:
        # list_memories 已按 mtime 倒序（最新在前）；这里以 (mtime, filename)
        # 显式排序保证同秒写入时次序也确定，取最新 N 条进索引。
        summaries = sorted(
            summaries,
            key=lambda e: ((get_memory_dir() / e.filename).stat().st_mtime, e.filename),
            reverse=True,
        )[:MAX_SUMMARY_INDEX_ENTRIES]
    lines = ["# Memory Index", ""]
    for m in others + summaries:
        lines.append(f"- **[{m.name}]({m.filename})** ({m.type}) — {m.description}")
    index_path = _get_index_path()
    # 先写同目录临时文件，再原子替换，避免并发写产生半截索引。
    tmp_path = index_path.with_name(f"{index_path.name}.tmp")
    tmp_path.write_text("\n".join(lines))
    os.replace(tmp_path, index_path)


def load_memory_index() -> str:
    """读取 MEMORY.md，并在注入 system prompt 前做长度保护。"""
    index_path = _get_index_path()
    if not index_path.exists():
        return ""
    content = index_path.read_text()
    lines = content.split("\n")
    if len(lines) > MAX_INDEX_LINES:
        content = "\n".join(lines[:MAX_INDEX_LINES]) + "\n\n[... truncated, too many memory entries ...]"
    if len(content.encode()) > MAX_INDEX_BYTES:
        content = content[:MAX_INDEX_BYTES] + "\n\n[... truncated, index too large ...]"
    return content


# ─── Memory Header (lightweight scan) ──────────────────────

class MemoryHeader:
    """轻量 memory 摘要，只包含召回筛选需要的元信息。

    Task 6/7 扩展字段：
    - access_count / last_accessed：召回命中统计，用于记忆衰减；
    - importance：记忆重要性（类型权重 × 基础值），用于低分淘汰；
    - body_tokens：正文前若干行 tokenize 结果，参与 BM25 匹配。
    """

    __slots__ = (
        "filename", "file_path", "mtime_ms", "description", "type",
        "access_count", "last_accessed", "importance", "body_tokens",
    )

    def __init__(self, filename: str, file_path: str, mtime_ms: float,
                 description: str | None = None, type: str | None = None,
                 access_count: int = 0, last_accessed: str = "",
                 importance: float = 1.0, body_tokens: list[str] | None = None):
        self.filename = filename
        self.file_path = file_path
        self.mtime_ms = mtime_ms
        self.description = description
        self.type = type
        self.access_count = access_count
        self.last_accessed = last_accessed
        self.importance = importance
        self.body_tokens = body_tokens or []


MAX_MEMORY_FILES = 200                    # 参与召回筛选的最多 memory 文件数。
MAX_MEMORY_BYTES_PER_FILE = 4096          # 单个 memory 注入前的最大字节数。
MAX_SESSION_MEMORY_BYTES = 60 * 1024      # 单个会话最多注入的 memory 总量。

# ─── 记忆衰减（Task 6）───
IMPORTANCE_BASE = 1.0                     # importance 基础值，与类型权重相乘。
TYPE_IMPORTANCE = {                       # 记忆类型 → importance 权重。
    "user": 1.0,
    "feedback": 0.9,
    "project": 0.8,
    "reference": 0.6,
}
DEFAULT_TTL_DAYS = {                      # 各类型默认 TTL；None 表示永不过期。
    "reference": 90,
    "project": 180,
    "user": None,
    "feedback": None,
}
EVICT_IMPORTANCE_FLOOR = 0.5              # importance 低于该值的记忆才参与淘汰。
BODY_TOKEN_LINES = 20                     # 正文参与 BM25 索引的最大行数。


def _safe_int(value: Any, default: int) -> int:
    """把 frontmatter 里的字符串安全转成 int，失败返回默认值。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    """把 frontmatter 里的字符串安全转成 float，失败返回默认值。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _tokenize(text: str) -> list[str]:
    """简单 tokenize：按非字母数字切分、小写化（与 bm25_topk 一致）。"""
    return re.findall(r"[a-z0-9]+", text.lower())


def _tokenize_body_preview(body: str) -> list[str]:
    """对记忆正文前若干行做 tokenize，并限制在单文件注入预算内。"""
    preview = "\n".join(body.split("\n")[:BODY_TOKEN_LINES])
    if len(preview.encode()) > MAX_MEMORY_BYTES_PER_FILE:
        preview = preview[:MAX_MEMORY_BYTES_PER_FILE]
    return _tokenize(preview)


def _parse_iso_time(ts: str) -> float | None:
    """把 ISO 时间戳转成 epoch 秒；解析失败返回 None（调用方用 mtime 兜底）。"""
    s = ts.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _is_protected_memory_file(filename: str) -> bool:
    """MEMORY.md 与 project_memory* 等关键文件永不参与淘汰。"""
    return filename == "MEMORY.md" or filename.startswith("project_memory")


def scan_memory_headers() -> list[MemoryHeader]:
    """快速扫描 memory 文件头，不读取完整正文，用于低成本召回筛选。"""
    d = get_memory_dir()
    headers: list[MemoryHeader] = []
    for f in d.glob("*.md"):
        if f.name == "MEMORY.md":
            continue
        try:
            stat = f.stat()
            raw = f.read_text()
            # 解析完整文件：frontmatter 元数据 + 正文（正文用于生成 body_tokens）。
            result = parse_frontmatter(raw)
            meta = result.meta
            t = meta.get("type")
            headers.append(MemoryHeader(
                filename=f.name,
                file_path=str(f),
                mtime_ms=stat.st_mtime * 1000,
                description=meta.get("description"),
                type=t if t in VALID_TYPES else None,
                access_count=_safe_int(meta.get("access_count"), 0),
                last_accessed=meta.get("last_accessed") or "",
                importance=_safe_float(meta.get("importance"), 1.0),
                body_tokens=_tokenize_body_preview(result.body),
            ))
        except Exception:
            pass
    headers.sort(key=lambda h: h.mtime_ms, reverse=True)
    return headers[:MAX_MEMORY_FILES]


# ─── BM25 关键词预筛 ───────────────────────────────────────

def bm25_topk(
    query: str,
    headers: list[MemoryHeader],
    k: int = 15,
    exclude: set[str] | None = None,
) -> list[MemoryHeader]:
    """用 BM25 对 query 与 header 的 name+description 打分，返回得分最高的 top-k。

    exclude 是已展示过的 file_path 集合，命中的文件直接跳过。
    query 无 token 或所有 header 得分为 0 时返回空列表，不做模糊降级。
    """
    query_tokens = re.findall(r"[a-z0-9]+", query.lower())
    if not query_tokens:
        return []

    excluded = exclude or set()
    docs = [h for h in headers if h.file_path not in excluded]
    if not docs:
        return []

    # header 没有独立的 name 字段，filename 即 type_slug(name)，与 description 一起构成文档文本；
    # Task 7：追加 body_tokens（正文前若干行 tokenize 结果），让正文参与 BM25 匹配。
    doc_texts = [
        (h, _tokenize(f"{h.filename.removesuffix('.md')} {h.description or ''}") + list(h.body_tokens or []))
        for h in docs
    ]

    # 统计每个 token 出现在几个文档里，用于 IDF。
    df: dict[str, int] = {}
    for _, tokens in doc_texts:
        for tok in set(tokens):
            df[tok] = df.get(tok, 0) + 1

    n = len(doc_texts)
    avgdl = sum(len(tokens) for _, tokens in doc_texts) / n if n else 0.0

    k1, b = BM25_K1, BM25_B
    scored: list[tuple[float, MemoryHeader]] = []
    for h, tokens in doc_texts:
        dl = len(tokens)
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        score = 0.0
        for tok in query_tokens:
            freq = tf.get(tok, 0)
            if not freq:
                continue
            idf = math.log(1 + (n - df[tok] + 0.5) / (df[tok] + 0.5))
            if avgdl > 0:
                denom = freq + k1 * (1 - b + b * dl / avgdl)
            else:
                denom = freq + k1 * (1 - b)
            score += idf * (freq * (k1 + 1)) / denom
        if score > 0:
            scored.append((score, h))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored[:k]]


# ─── Memory Headers 缓存 ────────────────────────────────────

_headers_cache: list[MemoryHeader] | None = None
_headers_cache_mtime: float = 0.0


def get_cached_headers() -> list[MemoryHeader]:
    """带目录 mtime 失效的 scan_memory_headers 缓存，避免每轮全量扫描。"""
    global _headers_cache, _headers_cache_mtime
    mtime = os.path.getmtime(get_memory_dir())
    if _headers_cache is not None and mtime == _headers_cache_mtime:
        return _headers_cache
    _headers_cache = scan_memory_headers()
    _headers_cache_mtime = mtime
    return _headers_cache


def touch_memory(name_or_path: str) -> bool:
    """命中召回时递增 access_count 并刷新 last_accessed，回写记忆文件 frontmatter。

    name_or_path 可以是记忆文件名或绝对路径；回写失败返回 False，不影响召回主流程。
    成功后同步更新内存中的 headers 缓存，保持索引一致性。
    """
    try:
        d = get_memory_dir()
        p = Path(name_or_path)
        if not p.is_absolute():
            p = d / name_or_path
        if not p.exists() or p.name == "MEMORY.md":
            return False
        result = parse_frontmatter(p.read_text())
        meta = result.meta
        new_count = _safe_int(meta.get("access_count"), 0) + 1
        new_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta["access_count"] = str(new_count)
        meta["last_accessed"] = new_ts
        p.write_text(format_frontmatter(meta, result.body))
        # 同步内存索引缓存中对应的 header。
        if _headers_cache is not None:
            for h in _headers_cache:
                if h.file_path == str(p):
                    h.access_count = new_count
                    h.last_accessed = new_ts
                    break
        return True
    except Exception:
        return False


def format_memory_manifest(headers: list[MemoryHeader]) -> str:
    """把 memory 摘要列表格式化成给 side query 阅读的 manifest。"""
    lines = []
    for h in headers:
        tag = f"[{h.type}] " if h.type else ""
        ts = datetime.fromtimestamp(h.mtime_ms / 1000, tz=timezone.utc).isoformat()
        if h.description:
            lines.append(f"- {tag}{h.filename} ({ts}): {h.description}")
        else:
            lines.append(f"- {tag}{h.filename} ({ts})")
    return "\n".join(lines)


# ─── Memory Age / Freshness ────────────────────────────────

def memory_age(mtime_ms: float) -> str:
    """把修改时间转换成适合展示的相对时间。"""
    days = max(0, int((time.time() * 1000 - mtime_ms) / 86_400_000))
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def memory_freshness_warning(mtime_ms: float) -> str:
    """旧记忆可能过期，注入时提醒模型先核对当前代码。"""
    days = max(0, int((time.time() * 1000 - mtime_ms) / 86_400_000))
    if days <= 1:
        return ""
    return (f"This memory is {days} days old. Memories are point-in-time observations, "
            "not live state — claims about code behavior may be outdated. "
            "Verify against current code before asserting as fact.")



SELECT_MEMORIES_PROMPT = """You are selecting memories that will be useful to an AI coding assistant as it processes a user's query. You will be given the user's query and a list of available memory files with their filenames and descriptions.

Return a JSON object with a "selected_memories" array of filenames for the memories that will clearly be useful (up to 5). Only include memories that you are certain will be helpful based on their name and description.
- If you are unsure if a memory will be useful, do not include it.
- If no memories would clearly be useful, return an empty array."""


class RelevantMemory:
    """被召回并准备注入对话的完整 memory。"""

    __slots__ = ("path", "content", "mtime_ms", "header")

    def __init__(self, path: str, content: str, mtime_ms: float, header: str):
        self.path = path
        self.content = content
        self.mtime_ms = mtime_ms
        self.header = header

    @property
    def size(self) -> int:
        """当前 memory 内容占用的字节数，供 Agent 统计本会话注入预算。"""
        return len(self.content.encode())


async def select_relevant_memories(
    query: str,
    side_query: SideQueryFn,
    already_surfaced: set[str],
) -> list[RelevantMemory]:
    """
    从 memory 目录中选择和当前 query 最相关的记忆（BM25 预筛 + LLM rerank 混合检索）。

    流程：
    1. 从缓存读取 memory 头信息，避免每轮全量扫描。
    2. 排除本 session 已经注入过的 memory，用 BM25 取 top-k 候选。
    3. 候选 ≤5 时确定性足够，跳过 LLM 直接读取返回。
    4. 候选 >5 时才把候选清单交给 side query，让模型 rerank 选最多 5 个。
    5. 读取被选中的 memory 正文，截断过大的文件。
    6. 包装成 RelevantMemory，交给后续注入逻辑。
    """

    # 用缓存读取 memory 文件头信息。
    headers = get_cached_headers()
    if not headers:
        return []

    # BM25 关键词预筛：排除已展示过的 memory，取相关性最高的候选。
    candidates = bm25_topk(query, headers, k=BM25_TOP_K, exclude=already_surfaced)
    if not candidates:
        return []

    try:
        # 候选不超过 5 个时跳过 LLM，直接读取并包装返回。
        if len(candidates) <= 5:
            return _wrap_memory_headers(candidates)

        # 候选多于 5 个时，才调用 side_query 让模型从候选清单里 rerank 选 ≤5。
        manifest = format_memory_manifest(candidates)
        text = await side_query(
            SELECT_MEMORIES_PROMPT,
            f"Query: {query}\n\nAvailable memories:\n{manifest}",
        )

        # side query 可能返回解释文本，这里只提取其中的 JSON 对象。
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return []

        # 解析 JSON，拿到被选中的 memory 文件名，最多取 5 个。
        parsed = json.loads(match.group(0))
        selected_filenames = set(parsed.get("selected_memories", []))
        selected = [h for h in candidates if h.filename in selected_filenames][:5]
        return _wrap_memory_headers(selected)
    except Exception as e:
        # 召回失败不应该影响主对话；取消类错误直接静默。
        if "cancel" in str(e).lower():
            return []
        print(f"[memory] semantic recall failed: {e}")
        return []


def _wrap_memory_headers(headers: list[MemoryHeader]) -> list[RelevantMemory]:
    """读取候选 memory 正文并包装成 RelevantMemory（含截断与 freshness 提示头）。"""
    result: list[RelevantMemory] = []
    for h in headers:
        # 读取每个候选的 memory 文件内容。
        content = Path(h.file_path).read_text()
        # 如果文件太大，就截断，避免单条记忆占用过多上下文。
        if len(content.encode()) > MAX_MEMORY_BYTES_PER_FILE:
            content = content[:MAX_MEMORY_BYTES_PER_FILE] + "\n\n[... truncated, memory file too large ...]"

        # 根据 memory 修改时间生成提示头；旧记忆会附带 freshness warning。
        freshness = memory_freshness_warning(h.mtime_ms)
        header_text = (
            f"{freshness}\n\nMemory: {h.file_path}:" if freshness
            else f"Memory (saved {memory_age(h.mtime_ms)}): {h.file_path}:"
        )
        # 返回 RelevantMemory 列表，后续会被格式化成 <system-reminder>。
        result.append(RelevantMemory(
            path=h.file_path, content=content,
            mtime_ms=h.mtime_ms, header=header_text,
        ))
        # Task 6：命中召回 → 递增 access_count 并刷新 last_accessed。
        # touch_memory 内部容错，回写失败不影响召回主流程。
        touch_memory(h.file_path)
    return result


# ─── 记忆遗忘与淘汰（Task 6）────────────────────────────────

def expire_stale_memories(ttl_days: dict | None = None,
                          memory_dir: Path | None = None) -> list[str]:
    """按类型 TTL 标记过期记忆，返回被标记的文件名列表（不物理删除）。

    ttl_days 覆盖默认 TTL（reference 90 天、project 180 天、user/feedback 永不过期）；
    value 为 None 或缺失表示该类型永不过期。
    last_accessed 为空时用文件 mtime 兜底；已标记过 expired 的文件跳过，避免重复回写。
    """
    ttl = {**DEFAULT_TTL_DAYS, **(ttl_days or {})}
    d = memory_dir or get_memory_dir()
    expired: list[str] = []
    now = time.time()
    for f in sorted(d.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        try:
            result = parse_frontmatter(f.read_text())
            meta = result.meta
            days = ttl.get(meta.get("type", ""))
            if days is None:
                continue
            if meta.get("expired") == "true":
                continue
            ts = _parse_iso_time(meta.get("last_accessed") or "")
            if ts is None:
                ts = f.stat().st_mtime
            if (now - ts) / 86400 > days:
                meta["expired"] = "true"
                f.write_text(format_frontmatter(meta, result.body))
                expired.append(f.name)
        except Exception:
            continue
    return expired


def _evict_low_importance_memories(limit: int,
                                   memory_dir: Path | None = None) -> list[str]:
    """当记忆文件数超过 limit 时，按 importance 升序淘汰低分记忆（物理删除）。

    保护 MEMORY.md 与 project_memory* 等关键文件（_is_protected_memory_file）；
    importance >= EVICT_IMPORTANCE_FLOOR 的记忆不淘汰。返回被删除的文件名列表。
    memory_dir 参数便于纯函数测试。
    """
    d = memory_dir or get_memory_dir()
    files = [f for f in d.glob("*.md") if not _is_protected_memory_file(f.name)]
    if len(files) <= limit:
        return []

    scored: list[tuple[float, str]] = []
    for f in files:
        try:
            meta = parse_frontmatter(f.read_text()).meta
            importance = _safe_float(meta.get("importance"), 1.0)
        except Exception:
            importance = 1.0  # 无法解析的记忆按高 importance 对待，避免误删。
        scored.append((importance, f.name))
    # importance 升序，文件名作稳定次序。
    scored.sort(key=lambda x: (x[0], x[1]))

    removed: list[str] = []
    for importance, name in scored:
        if len(files) - len(removed) <= limit:
            break
        if importance >= EVICT_IMPORTANCE_FLOOR:
            continue
        try:
            (d / name).unlink()
            removed.append(name)
        except OSError:
            pass
    return removed


class MemoryPrefetch:
    """封装 memory 召回异步任务，供 Agent 主循环轮询。"""

    def __init__(self, task: asyncio.Task):
        self.task = task
        # consumed 表示结果是否已经注入过，避免同一个任务结果重复使用。
        self.consumed = False

    @property
    def settled(self) -> bool:
        """任务是否已经完成。"""
        return self.task.done()


def start_memory_prefetch(
    query: str,
    side_query: SideQueryFn,
    already_surfaced: set[str],
    session_memory_bytes: int,
) -> MemoryPrefetch | None:
    """
    在主模型回复前，提前异步启动 memory 召回。

    返回值不是 memory 内容，而是 MemoryPrefetch 句柄。
    Agent 主循环后续会检查任务是否完成，完成后再把 memory 注入当前消息。
    """

    # 只有多词输入才触发 memory 预取，避免每个短命令都消耗一次 side query。
    if not re.search(r"\s", query.strip()):
        return None

    # 当前 session 的 memory 使用量不能超过预算。
    if session_memory_bytes >= MAX_SESSION_MEMORY_BYTES:
        return None

    # memory 目录里必须真的有 memory 文件。
    d = get_memory_dir()
    has_memories = any(f.suffix == ".md" and f.name != "MEMORY.md" for f in d.iterdir())
    if not has_memories:
        return None

    # 条件通过后创建异步任务，让召回和主模型请求并行推进。
    task = asyncio.create_task(
        select_relevant_memories(query, side_query, already_surfaced)
    )
    return MemoryPrefetch(task)


def format_memories_for_injection(memories: list[RelevantMemory]) -> str:
    """把召回的 memory 包成 system-reminder，便于注入到用户消息。"""
    parts = []
    for m in memories:
        parts.append(f"<system-reminder>\n{m.header}\n\n{m.content}\n</system-reminder>")
    return "\n\n".join(parts)


def build_memory_prompt_section() -> str:
    """
    生成注入 system prompt 的 Memory System 说明。

    这段说明告诉模型：
    - memory 文件存放在哪里；
    - 有哪些 memory 类型；
    - 如何通过 memory_save 工具保存 memory；
    - 哪些内容不应该保存；
    - 当前 MEMORY.md 索引里有哪些记忆。
    """
    index = load_memory_index()
    memory_dir = str(get_memory_dir())

    return f"""# Memory System

You have a persistent, file-based memory system at `{memory_dir}`.

## Memory Types
- **user**: User's role, preferences, knowledge level
- **feedback**: Corrections and guidance from the user (include Why + How to apply)
- **project**: Ongoing work, goals, deadlines, decisions
- **reference**: Pointers to external resources (URLs, tools, dashboards)

## How to Save Memories
Use the memory_save tool with these fields:
- name: short memory name (e.g. "user-prefers-chinese")
- description: one-line description
- type: user|feedback|project|reference
- content: memory body

The system validates the type, dedups by name (same name updates the existing entry), and auto-updates the MEMORY.md index.

## What NOT to Save
- Code patterns or architecture (read the code instead)
- Git history (use git log)
- Anything already in CLAUDE.md
- Ephemeral task details

## When to Recall
When the user asks you to remember or recall, or when prior context seems relevant.
{chr(10) + "## Current Memory Index" + chr(10) + index if index else chr(10) + "(No memories saved yet.)"}"""
