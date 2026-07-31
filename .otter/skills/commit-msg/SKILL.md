---
name: commit-msg
description: Generate a conventional commit message from current git changes.
user-invocable: true
when-to-use: When the user asks to write, generate, or improve a git commit message, e.g. "帮我写提交信息", "生成 commit message", "这次改动怎么写 commit".
---

# Commit Message Skill

Generate a concise, conventional commit message for the current staged/unstaged changes.

## Workflow

1. Run `git status --short` and `git diff --stat` to inspect what changed.
2. Classify the change type: feat / fix / docs / refactor / chore / test.
3. Determine the scope from the main files touched (e.g. tools, skills, docs).
4. Write the message in the Output Format below. Do NOT run `git commit`; only output the message.
5. If extra context is provided, treat it as the primary intent: $ARGUMENTS

## Output Format

- First line: `<type>(<scope>): <summary>` — summary in Chinese, under 50 characters, no trailing period.
- Add a short body (1-3 bullet lines) only when the change spans multiple concerns.
- Example: `feat(tools): 新增 file_stats 只读工具`
