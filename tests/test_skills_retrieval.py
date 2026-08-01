# tests/test_skills_retrieval.py
# 针对 agents/skills.py retrieve_relevant_skills 的最小聚焦测试：
#   1. 命中排序：强命中排在弱命中之前，且分数非升序
#   2. min_score 阈值：返回项分数恒 >= 阈值，阈值抬高剔除弱命中，超过上限返回空
#   3. 空查询边界：空串/纯空白/纯停用词即使有 skill 也返回 []
# 通过注入 skills._cached_skills 隔离磁盘扫描，仅使用标准库 unittest。
# 运行方式：python -m unittest discover -s tests

import unittest

from agents import skills
from agents.skills import SkillDefinition, retrieve_relevant_skills

# 命中排序/阈值测试统一使用的查询：A 强命中(3 词)、B 弱命中(1 词)、C 不命中(0 词)。
QUERY = "database migration rollback"


def _skill(name, description, when_to_use="", body=""):
    return SkillDefinition(
        name=name,
        description=description,
        when_to_use=when_to_use,
        prompt_template=body,
        source="project",
        skill_dir=f"/fake/{name}",
    )


class TestRetrieveRelevantSkills(unittest.TestCase):
    def setUp(self):
        # 注入受控 skill 集合，绕过 ~/.otter 与项目 .otter 的真实磁盘扫描；teardown 还原。
        self._saved_cache = skills._cached_skills
        skills._cached_skills = [
            _skill(
                "database-migrations",
                "database migration schema rollback zero downtime",
                "use for database schema migration and rollback",
            ),
            _skill(
                "backend-patterns",
                "backend architecture patterns and database access layer",
                "general backend service design",
            ),
            _skill(
                "weather-forecast",
                "predict rain sunshine temperature",
                "daily weather outlook",
            ),
        ]

    def tearDown(self):
        skills._cached_skills = self._saved_cache

    # ---- 命中排序 ----

    def test_hits_ranked_strong_before_weak(self):
        hits = retrieve_relevant_skills(QUERY, limit=10, min_score=0.0)
        names = [h["name"] for h in hits]
        # 强命中与弱命中都应召回，完全不相关的 skill 不应出现。
        self.assertIn("database-migrations", names)
        self.assertIn("backend-patterns", names)
        self.assertNotIn("weather-forecast", names)
        # 强命中（3 词重叠）排在弱命中（1 词重叠）之前。
        self.assertEqual(hits[0]["name"], "database-migrations")
        self.assertLess(names.index("database-migrations"), names.index("backend-patterns"))
        # 结果按分数非升序排列。
        scores = [h["score"] for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_hit_shape_contains_expected_fields(self):
        top = retrieve_relevant_skills(QUERY, limit=1, min_score=0.0)[0]
        self.assertEqual(
            set(top),
            {
                "score",
                "name",
                "description",
                "when_to_use",
                "source",
                "context",
                "user_invocable",
                "skill_dir",
            },
        )
        self.assertIsInstance(top["score"], float)

    def test_limit_caps_and_clamps_to_at_least_one(self):
        # limit 截断到 top-N。
        self.assertEqual(len(retrieve_relevant_skills(QUERY, limit=1, min_score=0.0)), 1)
        # limit<=0 被 max(1, ...) 兜底为 1，而非返回空。
        self.assertEqual(len(retrieve_relevant_skills(QUERY, limit=0, min_score=0.0)), 1)

    # ---- min_score 阈值 ----

    def test_min_score_is_a_lower_bound(self):
        for ms in (0.0, 0.05, 0.1):
            for hit in retrieve_relevant_skills(QUERY, limit=10, min_score=ms):
                self.assertGreaterEqual(hit["score"], ms)

    def test_higher_threshold_drops_weak_hits(self):
        loose = retrieve_relevant_skills(QUERY, limit=10, min_score=0.0)
        self.assertGreaterEqual(len(loose), 2)  # 强、弱命中都在
        sc = sorted(h["score"] for h in loose)
        self.assertLess(sc[0], sc[-1])  # 前提：强弱命中分数确有差异
        # 取介于最弱与最强之间的阈值，弱命中应被剔除。
        strict = (sc[0] + sc[-1]) / 2.0
        filtered = retrieve_relevant_skills(QUERY, limit=10, min_score=strict)
        self.assertTrue(all(h["score"] >= strict for h in filtered))
        self.assertLess(len(filtered), len(loose))

    def test_threshold_above_max_score_returns_empty(self):
        # 分数上限被 min(1.0, ...) 钳制，阈值 > 1.0 时必然全部过滤。
        self.assertEqual(retrieve_relevant_skills(QUERY, min_score=1.01), [])

    # ---- 空查询边界 ----

    def test_empty_or_stopword_query_returns_empty_even_with_skills(self):
        # 即便注入了可命中的 skill，无有效 token 的查询也应返回 []。
        for q in ["", "   ", "\n\t ", "这个", "帮我"]:
            with self.subTest(query=repr(q)):
                self.assertEqual(retrieve_relevant_skills(q), [])


if __name__ == "__main__":
    unittest.main()
