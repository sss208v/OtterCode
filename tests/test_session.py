"""agents/session.py 存储层测试：save/load/list/get_latest/remove。

不触碰真实用户目录：setUp 把 SESSION_DIR 重定向到临时目录。
"""

import pathlib
import tempfile
import unittest

from agents import session


class TestSessionStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_dir = session.SESSION_DIR
        session.SESSION_DIR = pathlib.Path(self._tmp.name)

    def tearDown(self) -> None:
        session.SESSION_DIR = self._orig_dir
        self._tmp.cleanup()

    def _sample(self, session_id: str = "abc12345") -> dict:
        return {
            "metadata": {
                "id": session_id,
                "model": "test-model",
                "cwd": "/tmp",
                "startTime": "2026-08-06T00:00:00Z",
                "messageCount": 2,
            },
            "anthropicMessages": [{"role": "user", "content": "你好"}],
            "openaiMessages": None,
            "verification": None,
            "readFileState": None,
        }

    def test_save_load_roundtrip(self) -> None:
        session.save_session("abc12345", self._sample())
        data = session.load_session("abc12345")
        self.assertIsNotNone(data)
        self.assertEqual(data["metadata"]["id"], "abc12345")
        self.assertEqual(data["anthropicMessages"][0]["content"], "你好")

    def test_load_missing_returns_none(self) -> None:
        self.assertIsNone(session.load_session("nope1234"))

    def test_load_corrupt_returns_none(self) -> None:
        (session.SESSION_DIR / "corrupt1.json").write_text("{ not json")
        self.assertIsNone(session.load_session("corrupt1"))

    def test_list_sessions_returns_metadata_only(self) -> None:
        session.save_session("abc12345", self._sample())
        session.save_session("def56789", self._sample("def56789"))
        metas = session.list_sessions()
        self.assertEqual(len(metas), 2)
        for m in metas:
            self.assertIn("id", m)
            self.assertNotIn("anthropicMessages", m)

    def test_list_sessions_skips_corrupt(self) -> None:
        session.save_session("abc12345", self._sample())
        (session.SESSION_DIR / "corrupt1.json").write_text("{ not json")
        self.assertEqual(len(session.list_sessions()), 1)

    def test_get_latest_session_id_by_start_time(self) -> None:
        session.save_session("old1", self._sample("old1"))
        older = self._sample("new1")
        older["metadata"]["startTime"] = "2026-08-07T00:00:00Z"
        session.save_session("new1", older)
        self.assertEqual(session.get_latest_session_id(), "new1")

    def test_remove_session(self) -> None:
        session.save_session("abc12345", self._sample())
        session.remove_session("abc12345")
        self.assertIsNone(session.load_session("abc12345"))
        self.assertEqual(session.list_sessions(), [])

    def test_remove_missing_idempotent(self) -> None:
        session.remove_session("does_not_exist")  # 不应抛异常


if __name__ == "__main__":
    unittest.main()
