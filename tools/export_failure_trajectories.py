#!/usr/bin/env python3
"""从会话存档导出失败轨迹（failure trajectories）为可回放 JSONL。

失败轨迹 = 任务描述 + 到失败为止的完整动作序列（user/assistant/tool，含工具错误），
供 eval / 训练 / 复盘分析。默认只导出 outcome=fail 的会话；--all 导出全部。

用法：
  python -m tools.export_failure_trajectories [--dir ~/.otter-code/sessions] [--out failures.jsonl]
      [--task 关键词] [--since 2026-08-01] [--all] [--exclude-tbench]

输出每行一个会话：
  {session_id, task, outcome, message_count, last_verification, trajectory: [...]}
trajectory 元素：{role: user|assistant|tool, text | name/input_summary/result}
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SESSION_DIR = Path.home() / ".otter-code" / "sessions"


def _iter_sessions(session_dir: Path):
    for f in sorted(session_dir.glob("*.json")):
        try:
            yield f, json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue


def _trajectory(data: dict) -> list[dict]:
    """从消息历史提取可回放的动作序列（tool_result 按 tool_use_id 回填）。"""
    msgs = data.get("anthropicMessages") or data.get("openaiMessages") or []
    trajectory: list[dict] = []
    pending: dict[str, dict] = {}  # tool_use_id -> tool 条目
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            content = m.get("content")
            if isinstance(content, str):
                trajectory.append({"role": "user", "text": content})
            elif isinstance(content, list):
                # tool_result 块：按 tool_use_id 匹配回填（含错误文本）
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "tool_result" and blk.get("tool_use_id"):
                        t = pending.pop(blk["tool_use_id"], None)
                        if t is not None:
                            res = blk.get("content")
                            if isinstance(res, list):
                                res = "\n".join(str(b.get("text", "")) for b in res if isinstance(b, dict))
                            t["result"] = str(res or "")[:2000]
                            trajectory.append(t)
            continue
        if role == "assistant":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                trajectory.append({"role": "assistant", "text": content})
            elif isinstance(content, list):
                for blk in content:
                    if not isinstance(blk, dict):
                        continue
                    if blk.get("type") == "text" and blk.get("text"):
                        trajectory.append({"role": "assistant", "text": blk["text"]})
                    elif blk.get("type") == "tool_use":
                        inp = blk.get("input") or {}
                        summary = str(inp.get("summary") or "") or json.dumps(inp, ensure_ascii=False)[:80]
                        pending[blk.get("id", "")] = {
                            "role": "tool",
                            "name": blk.get("name", ""),
                            "input_summary": summary,
                        }
    # 未匹配到结果的工具（中断/失败）补一个标记
    for t in pending.values():
        t["result"] = "[no result / interrupted]"
        trajectory.append(t)
    return trajectory


def _is_tbench_session(data: dict) -> bool:
    for m in data.get("anthropicMessages") or data.get("openaiMessages") or []:
        if isinstance(m, dict) and "T-Bench 容器环境" in str(m.get("content", "")):
            return True
    return False


def export(session_dir: Path, *, task: str = "", since: str = "", all_outcomes: bool = False,
           exclude_tbench: bool = True) -> list[dict]:
    rows = []
    for path, data in _iter_sessions(session_dir):
        meta = data.get("metadata") or {}
        outcome = str(meta.get("outcome") or "unknown")
        if not all_outcomes and outcome != "fail":
            continue
        if task and task not in str(meta.get("task") or ""):
            continue
        if since and str(meta.get("startTime") or "") < since:
            continue
        if exclude_tbench and _is_tbench_session(data):
            continue
        verification = data.get("verification") or []
        last_v = verification[-1] if verification else None
        rows.append({
            "session_id": str(meta.get("id") or path.stem),
            "task": str(meta.get("task") or ""),
            "outcome": outcome,
            "message_count": int(meta.get("messageCount") or 0),
            "last_verification": last_v,
            "trajectory": _trajectory(data),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="导出会话失败轨迹为可回放 JSONL")
    ap.add_argument("--dir", default=str(SESSION_DIR), help="会话存档目录")
    ap.add_argument("--out", default="failures.jsonl", help="输出文件")
    ap.add_argument("--task", default="", help="按任务关键词过滤")
    ap.add_argument("--since", default="", help="只导出 startTime >= 该日期（ISO）的会话")
    ap.add_argument("--all", action="store_true", help="导出全部会话（默认只导出 outcome=fail）")
    ap.add_argument("--exclude-tbench", action="store_true", default=True, help="排除 T-Bench 容器会话（默认开启）")
    args = ap.parse_args()

    rows = export(
        Path(args.dir),
        task=args.task,
        since=args.since,
        all_outcomes=args.all,
        exclude_tbench=args.exclude_tbench,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"已导出 {len(rows)} 个会话 → {args.out}")


if __name__ == "__main__":
    main()
