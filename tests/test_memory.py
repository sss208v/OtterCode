# tests/test_memory.py
# 针对 agents/memory.py 记忆系统与上下文管理增强的回归测试：
#   1. save_memory_structured 结构化保存：frontmatter 元数据齐全 / type 与字段非空校验 / 同名去重更新
#   2. update_memory_index 索引原子写（temp + os.replace，不残留 .tmp）
#   3. bm25_topk 关键词预筛：相关性排序 / 无命中返回空 / exclude 生效
#   4. select_relevant_memories 混合检索：候选 ≤5 跳过 LLM / 无命中返回空 / LLM 失败降级 / 预算封顶截断
#   5. estimate_tokens 近似 token 估算（CJK 每字 1.5，ASCII 4 字符/token）
# 仅使用标准库 unittest + unittest.mock + tempfile，运行方式：python -m unittest discover -s tests

import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agents import memory
from agents.agent import estimate_tokens
from agents.frontmatter import parse_frontmatter


class _MemoryDirTestCase(unittest.TestCase):
    """写盘用例基类：把 get_memory_dir 指向临时目录，隔离真实用户记忆。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._dir = Path(self._tmp.name)
        # 隔离真实用户记忆目录：patch 模块级 get_memory_dir，沿模块引用生效。
        self._patch_dir = patch("agents.memory.get_memory_dir", return_value=self._dir)
        self._patch_dir.start()
        self.addCleanup(self._patch_dir.stop)
        # 重置 headers 缓存，避免跨用例残留。
        self._saved_cache = (memory._headers_cache, memory._headers_cache_mtime)
        memory._headers_cache = None
        memory._headers_cache_mtime = 0.0

    def tearDown(self):
        memory._headers_cache, memory._headers_cache_mtime = self._saved_cache
        self._tmp.cleanup()

    def _write_memory(self, filename: str, name: str, description: str, body: str = "body"):
        """在临时记忆目录写入一条带 frontmatter 的真实记忆文件。"""
        (self._dir / filename).write_text(
            f"---\nname: {name}\ndescription: {description}\ntype: project\n---\n\n{body}"
        )

    def _write_typed_memory(self, filename: str, name: str, mtype: str,
                            body: str = "body", **extra_meta):
        """写入一条可自定义 type 与额外 frontmatter 字段（如 importance/last_accessed）的记忆。"""
        meta_lines = "\n".join(
            [f"name: {name}", "description: misc note", f"type: {mtype}"]
            + [f"{k}: {v}" for k, v in extra_meta.items()]
        )
        (self._dir / filename).write_text(f"---\n{meta_lines}\n---\n\n{body}")


class TestFrontmatterAndSave(_MemoryDirTestCase):
    """save_memory_structured 新建记忆：frontmatter 元数据齐全、字段校验。"""

    def test_new_memory_writes_full_metadata(self):
        ret = memory.save_memory_structured(
            name="user prefers chinese",
            description="用户偏好中文回复",
            type="user",
            content="回复时使用简体中文。",
            session_id="sess-123",
        )
        self.assertTrue(ret.startswith("saved: "))
        fpath = self._dir / ret[len("saved: "):]
        self.assertTrue(fpath.exists())
        meta = parse_frontmatter(fpath.read_text()).meta
        for key in ("name", "description", "type", "created_at", "updated_at", "source_session"):
            self.assertIn(key, meta)
        self.assertEqual(meta["name"], "user prefers chinese")
        self.assertEqual(meta["type"], "user")
        self.assertEqual(meta["source_session"], "sess-123")
        self.assertEqual(meta["created_at"], meta["updated_at"])

    def test_invalid_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            memory.save_memory_structured("n", "d", "bogus", "c")

    def test_empty_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            memory.save_memory_structured("  ", "d", "project", "c")

    def test_empty_content_raises_value_error(self):
        with self.assertRaises(ValueError):
            memory.save_memory_structured("n", "d", "project", "   ")


class TestDedupUpdate(_MemoryDirTestCase):
    """同名记忆去重更新：返回 updated existing，created_at 保留、updated_at 刷新。"""

    def test_same_name_updates_existing_entry(self):
        calls = {"n": 0}

        def fake_strftime(fmt, t=None):
            # 第一次保存返回 01-01，第二次返回 01-02，保证秒级时间戳变化可观测。
            calls["n"] += 1
            return "2024-01-01T00:00:00Z" if calls["n"] == 1 else "2024-01-02T00:00:00Z"

        with patch.object(memory.time, "strftime", side_effect=fake_strftime):
            ret1 = memory.save_memory_structured("db schema", "schema", "project", "content v1")
            fname = ret1[len("saved: "):]
            # 第一次保存后立即读取元数据，再触发第二次更新。
            meta1 = parse_frontmatter((self._dir / fname).read_text()).meta
            ret2 = memory.save_memory_structured("db schema", "schema", "project", "content v2")
            meta2 = parse_frontmatter((self._dir / fname).read_text()).meta

        self.assertTrue(ret1.startswith("saved: "))
        self.assertTrue(ret2.startswith("updated existing: "))
        self.assertEqual(ret2[len("updated existing: "):], fname)
        # created_at 保留原值，updated_at 刷新。
        self.assertEqual(meta2["created_at"], meta1["created_at"])
        self.assertNotEqual(meta2["updated_at"], meta1["updated_at"])
        self.assertEqual(meta2["updated_at"], "2024-01-02T00:00:00Z")
        # 正文被新内容覆盖，旧内容不残留。
        self.assertNotIn("content v1", (self._dir / fname).read_text())


class TestMemoryIndex(_MemoryDirTestCase):
    """update_memory_index 索引原子写：内容完整、无 .tmp 残留。"""

    def test_index_generated_without_tmp_leftover(self):
        memory.save_memory_structured("db schema", "数据库表结构说明", "project", "tables: users, orders")
        index_path = self._dir / "MEMORY.md"
        self.assertTrue(index_path.exists())
        content = index_path.read_text()
        self.assertTrue(content.startswith("# Memory Index"))
        self.assertIn("db schema", content)
        self.assertIn("数据库表结构说明", content)
        # 原子写：目录下不残留 .tmp 临时文件。
        self.assertEqual(list(self._dir.glob("*.tmp")), [])


class TestBM25TopK(_MemoryDirTestCase):
    """bm25_topk 关键词预筛：相关性排序、无命中、exclude。"""

    def _header(self, filename: str, description: str):
        return memory.MemoryHeader(
            filename=filename,
            file_path=str(self._dir / filename),
            mtime_ms=time.time() * 1000,
            description=description,
            type="project",
        )

    def test_relevant_ranks_first(self):
        db = self._header("project_1.md", "database schema notes")
        dep = self._header("project_2.md", "deployment pipeline guide")
        des = self._header("project_3.md", "dessert recipes")
        result = memory.bm25_topk("database", [db, dep, des])
        self.assertEqual(result[0].filename, "project_1.md")

    def test_no_query_token_returns_empty(self):
        db = self._header("project_1.md", "database schema notes")
        dep = self._header("project_2.md", "deployment pipeline guide")
        # query 无 token 或完全无命中时返回空，不做模糊降级。
        self.assertEqual(memory.bm25_topk("!!!", [db, dep]), [])
        self.assertEqual(memory.bm25_topk("zzzmissing", [db, dep]), [])

    def test_exclude_filters_files(self):
        db = self._header("project_1.md", "database schema notes")
        dep = self._header("project_2.md", "deployment pipeline guide")
        result = memory.bm25_topk("database", [db, dep], exclude={db.file_path})
        self.assertEqual(result, [])


class TestSelectRelevantMemories(_MemoryDirTestCase):
    """select_relevant_memories 混合检索：跳过 LLM / 无命中 / 失败降级 / 预算封顶。"""

    def _boom_side_query(self, message):
        async def _fail(system_prompt, user_prompt):
            raise RuntimeError(message)
        return _fail

    def test_candidates_leq5_skip_llm(self):
        for i in range(3):
            self._write_memory(f"project_mem_{i}.md", f"mem {i}", f"database deployment notes {i}")
        async def _must_not_call(system_prompt, user_prompt):
            raise AssertionError("side_query 不应在候选 ≤5 时被调用")
        result = asyncio.run(memory.select_relevant_memories("database", _must_not_call, set()))
        self.assertEqual(len(result), 3)
        self.assertEqual(
            {m.path for m in result},
            {str(self._dir / f"project_mem_{i}.md") for i in range(3)},
        )

    def test_no_bm25_hit_returns_empty(self):
        for i in range(3):
            self._write_memory(f"project_mem_{i}.md", f"mem {i}", f"apple banana notes {i}")
        async def _must_not_call(system_prompt, user_prompt):
            raise AssertionError("side_query 不应在 BM25 无命中时被调用")
        result = asyncio.run(memory.select_relevant_memories("zzzmissing", _must_not_call, set()))
        self.assertEqual(result, [])

    def test_llm_failure_falls_back_empty(self):
        # 候选 >5 才触发 side_query；side_query 抛异常时降级返回 []，不中断主流程。
        for i in range(6):
            self._write_memory(f"project_mem_{i}.md", f"mem {i}", f"database deployment notes {i}")
        result = asyncio.run(
            memory.select_relevant_memories("database", self._boom_side_query("llm down"), set())
        )
        self.assertEqual(result, [])

    def test_oversized_memory_truncated(self):
        big = "x" * (memory.MAX_MEMORY_BYTES_PER_FILE * 2)
        self._write_memory("project_big.md", "big", "database backup guide", body=big)
        async def _must_not_call(system_prompt, user_prompt):
            raise AssertionError("side_query 不应在候选 ≤5 时被调用")
        result = asyncio.run(memory.select_relevant_memories("database", _must_not_call, set()))
        self.assertEqual(len(result), 1)
        self.assertIn("[... truncated", result[0].content)


class TestMemoryDecay(_MemoryDirTestCase):
    """Task 6：TTL 过期标记、access_count 递增、importance 衰减字段回写。"""

    def test_reference_expires_after_ttl(self):
        old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 91 * 86400))
        self._write_typed_memory("reference_old.md", "old ref", "reference", last_accessed=old)
        expired = memory.expire_stale_memories(ttl_days={"reference": 90}, memory_dir=self._dir)
        self.assertIn("reference_old.md", expired)
        meta = parse_frontmatter((self._dir / "reference_old.md").read_text()).meta
        self.assertEqual(meta.get("expired"), "true")

    def test_user_memory_never_expires(self):
        old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 500 * 86400))
        self._write_typed_memory("user_old.md", "old pref", "user", last_accessed=old)
        expired = memory.expire_stale_memories(ttl_days={"reference": 90}, memory_dir=self._dir)
        self.assertEqual(expired, [])
        meta = parse_frontmatter((self._dir / "user_old.md").read_text()).meta
        self.assertNotIn("expired", meta)

    def test_project_uses_default_ttl(self):
        old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 181 * 86400))
        self._write_typed_memory("project_old.md", "old proj", "project", last_accessed=old)
        expired = memory.expire_stale_memories(memory_dir=self._dir)  # 用默认 TTL（project 180 天）
        self.assertIn("project_old.md", expired)

    def test_missing_last_accessed_falls_back_to_mtime(self):
        self._write_typed_memory("reference_stale.md", "stale", "reference")
        f = self._dir / "reference_stale.md"
        old_mtime = time.time() - 91 * 86400
        import os
        os.utime(f, (old_mtime, old_mtime))
        expired = memory.expire_stale_memories(ttl_days={"reference": 90}, memory_dir=self._dir)
        self.assertIn("reference_stale.md", expired)

    def test_save_memory_writes_decay_fields(self):
        memory.save_memory_structured("decay target", "d", "reference", "c")
        fpath = self._dir / "reference_decay_target.md"
        meta = parse_frontmatter(fpath.read_text()).meta
        self.assertEqual(meta["access_count"], "0")
        self.assertEqual(meta["last_accessed"], "")
        self.assertAlmostEqual(float(meta["importance"]), 0.6)  # reference 权重 0.6

    def test_touch_memory_increments_access_count(self):
        memory.save_memory_structured("touch target", "d", "project", "c")
        fpath = self._dir / "project_touch_target.md"
        self.assertTrue(memory.touch_memory(str(fpath)))
        meta = parse_frontmatter(fpath.read_text()).meta
        self.assertEqual(int(meta["access_count"]), 1)
        self.assertNotEqual(meta["last_accessed"], "")
        # 再次 touch 递增。
        memory.touch_memory(str(fpath))
        meta = parse_frontmatter(fpath.read_text()).meta
        self.assertEqual(int(meta["access_count"]), 2)

    def test_recall_touches_memory(self):
        memory.save_memory_structured("recall me", "database schema notes", "project", "body")
        async def _must_not_call(system_prompt, user_prompt):
            raise AssertionError("候选 ≤5 不应调用 side_query")
        asyncio.run(memory.select_relevant_memories("database", _must_not_call, set()))
        fpath = self._dir / "project_recall_me.md"
        meta = parse_frontmatter(fpath.read_text()).meta
        self.assertEqual(int(meta["access_count"]), 1)
        self.assertNotEqual(meta["last_accessed"], "")


class TestLowImportanceEviction(_MemoryDirTestCase):
    """Task 6：文件数超限时按 importance 升序淘汰低分记忆，关键文件受保护。"""

    def test_evict_helper_returns_removed_filenames(self):
        self._write_typed_memory("project_a.md", "a", "project", importance=0.1)
        self._write_typed_memory("project_b.md", "b", "project", importance=0.2)
        self._write_typed_memory("project_c.md", "c", "project", importance=0.9)
        removed = memory._evict_low_importance_memories(2, memory_dir=self._dir)
        self.assertEqual(removed, ["project_a.md"])  # 删 1 条即回到 limit
        self.assertFalse((self._dir / "project_a.md").exists())
        self.assertTrue((self._dir / "project_b.md").exists())
        self.assertTrue((self._dir / "project_c.md").exists())

    def test_update_memory_index_evicts_over_limit(self):
        with patch.object(memory, "MAX_MEMORY_FILES", 2):
            self._write_typed_memory("project_a.md", "a", "project", importance=0.1)
            self._write_typed_memory("project_b.md", "b", "project", importance=0.3)
            self._write_typed_memory("project_c.md", "c", "project", importance=0.9)
            memory.update_memory_index()
        self.assertFalse((self._dir / "project_a.md").exists())
        self.assertTrue((self._dir / "project_b.md").exists())
        self.assertTrue((self._dir / "project_c.md").exists())

    def test_protected_files_never_evicted(self):
        # project_memory 即使 importance 最低（0.0）也永不淘汰；淘汰池内的最低分先被删。
        with patch.object(memory, "MAX_MEMORY_FILES", 1):
            self._write_typed_memory("project_memory.md", "pm", "project", importance=0.0)
            self._write_typed_memory("project_a.md", "a", "project", importance=0.1)
            self._write_typed_memory("project_b.md", "b", "project", importance=0.2)
            memory.update_memory_index()
        self.assertTrue((self._dir / "project_memory.md").exists())
        self.assertFalse((self._dir / "project_a.md").exists())
        self.assertTrue((self._dir / "project_b.md").exists())

    def test_no_eviction_under_limit(self):
        self._write_typed_memory("project_a.md", "a", "project", importance=0.1)
        self._write_typed_memory("project_b.md", "b", "project", importance=0.2)
        removed = memory._evict_low_importance_memories(5, memory_dir=self._dir)
        self.assertEqual(removed, [])
        self.assertTrue((self._dir / "project_a.md").exists())
        self.assertTrue((self._dir / "project_b.md").exists())


class TestBM25BodyIndex(_MemoryDirTestCase):
    """Task 7：正文参与 BM25 匹配，description 模糊时正文命中可召回。"""

    def test_body_token_ranked_over_vague_description(self):
        self._write_memory("reference_db.md", "db usage", "misc note",
                           body="install and use sqlite for local storage")
        self._write_memory("reference_other.md", "other note", "misc note",
                           body="nothing about databases here")
        headers = memory.scan_memory_headers()
        # 正文被 tokenize 进 body_tokens。
        db = next(h for h in headers if h.filename == "reference_db.md")
        self.assertIn("sqlite", db.body_tokens)
        result = memory.bm25_topk("sqlite", headers)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].filename, "reference_db.md")

    def test_recall_finds_memory_by_body(self):
        self._write_memory("reference_db.md", "db usage", "misc note",
                           body="install and use sqlite for local storage")
        self._write_memory("reference_other.md", "other note", "misc note",
                           body="nothing about databases here")
        async def _must_not_call(system_prompt, user_prompt):
            raise AssertionError("候选 ≤5 不应调用 side_query")
        result = asyncio.run(memory.select_relevant_memories("sqlite", _must_not_call, set()))
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].path.endswith("reference_db.md"))


class TestSummaryIndexCap(_MemoryDirTestCase):
    """Task 17：MEMORY.md 索引对 conversation-compact-summary 摘要只保留最近 5 条。"""

    def _write_summaries(self, count: int):
        """写入 count 条 conversation-compact-summary-* 记忆，mtime 递增（i 越大越新）。"""
        import os
        base = time.time() - 10
        for i in range(count):
            self._write_typed_memory(
                f"project_summary_{i}.md",
                f"conversation-compact-summary-{i}",
                "project",
                body=f"summary content {i}",
            )
            os.utime(self._dir / f"project_summary_{i}.md", (base + i, base + i))

    def test_index_keeps_latest_5_summaries(self):
        self._write_summaries(7)
        memory.update_memory_index()
        content = (self._dir / "MEMORY.md").read_text()
        # 摘要条目最多 5 条。
        summary_lines = [ln for ln in content.splitlines() if "conversation-compact-summary" in ln]
        self.assertEqual(len(summary_lines), 5)
        # 保留的是最新 5 条（2..6），最旧的 0/1 被省略。
        self.assertIn("conversation-compact-summary-2", content)
        self.assertIn("conversation-compact-summary-6", content)
        self.assertNotIn("conversation-compact-summary-0", content)
        self.assertNotIn("conversation-compact-summary-1", content)

    def test_non_summary_memories_all_kept(self):
        for i in range(3):
            self._write_memory(f"project_norm_{i}.md", f"normal memory {i}", f"database deployment notes {i}")
        self._write_summaries(7)
        memory.update_memory_index()
        content = (self._dir / "MEMORY.md").read_text()
        # 非摘要记忆全部保留。
        for i in range(3):
            self.assertIn(f"normal memory {i}", content)
        # 摘要只留 5 条。
        summary_lines = [ln for ln in content.splitlines() if "conversation-compact-summary" in ln]
        self.assertEqual(len(summary_lines), 5)

    def test_few_summaries_keep_all(self):
        self._write_summaries(3)
        memory.update_memory_index()
        content = (self._dir / "MEMORY.md").read_text()
        summary_lines = [ln for ln in content.splitlines() if "conversation-compact-summary" in ln]
        self.assertEqual(len(summary_lines), 3)


class TestEstimateTokens(unittest.TestCase):
    """estimate_tokens 近似 token 估算冒烟。"""

    def test_estimate_tokens(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("x" * 100), 25)      # ASCII 约 4 字符/token
        self.assertEqual(estimate_tokens("中" * 100), 150)    # CJK 每字 1.5 token


if __name__ == "__main__":
    unittest.main()
