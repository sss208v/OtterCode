# tests/test_no_duplicate_definitions.py
# 轻量静态检查：用标准库 ast 扫描 agents/ 包内每个模块，禁止出现
# 重复的顶层函数/类定义（redefinition）。这类重复会让后一个定义静默遮蔽前一个，
# 属于典型的死代码/维护陷阱——正是 get_active_tool_definitions 曾出现的问题。
# 作为回归门禁，防止此类重复被再次引入。仅使用标准库 unittest。
# 运行方式：python -m unittest discover -s tests

import ast
import unittest
from collections import Counter
from pathlib import Path

import agents

_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _is_overload(node: ast.AST) -> bool:
    # @overload / @typing.overload 允许同名多次定义，属于合法场景，跳过。
    for dec in getattr(node, "decorator_list", []):
        if isinstance(dec, ast.Name) and dec.id == "overload":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "overload":
            return True
    return False


def _duplicate_top_level_defs(source: str) -> dict[str, int]:
    tree = ast.parse(source)
    counter: Counter[str] = Counter()
    for node in tree.body:  # 只看模块顶层，条件分支内的同名定义不误报
        if isinstance(node, _DEF_NODES):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_overload(node):
                continue
            counter[node.name] += 1
    return {name: n for name, n in counter.items() if n > 1}


class TestNoDuplicateTopLevelDefinitions(unittest.TestCase):
    def test_agents_package_has_no_redefined_symbols(self):
        # agents 是命名空间包(无 __init__.py)，__file__ 为 None，改用 __path__ 定位目录。
        py_files = sorted(f for p in agents.__path__ for f in Path(p).glob("*.py"))
        self.assertTrue(py_files, "未扫描到任何 agents/*.py，检查路径解析")

        offenders: dict[str, dict[str, int]] = {}
        for path in py_files:
            dup = _duplicate_top_level_defs(path.read_text(encoding="utf-8"))
            if dup:
                offenders[path.name] = dup

        self.assertEqual(
            offenders,
            {},
            f"发现重复的顶层定义（后者会遮蔽前者，应合并保留一处）：{offenders}",
        )


if __name__ == "__main__":
    unittest.main()
