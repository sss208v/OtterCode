# AGENTS.md

面向进驻本仓库的 coding agent 的机器可读指令。本文件只陈述仓库内可验证的事实与约束，与 README.md 保持一致；冲突时以源码为准。

## 核心边界：agents/ 目录

`agents/` 是本项目的全部核心源码，所有工程改动都发生在这里：

- `agents/main.py`：CLI 入口、REPL、参数解析
- `agents/agent.py`：Agent Runtime、模型调用、工具调度、上下文压缩
- `agents/tools.py`：内置工具、权限系统、危险命令拦截（见下方危险区）
- `agents/prompt.py`：system prompt 动态构建
- `agents/skills.py` / `agents/online_skill_evolution.py` / `agents/skill_evolution.py`：Skills 加载、检索与自进化
- `agents/memory.py`、`agents/mcp_client.py`、`agents/subagent.py`、`agents/session.py`、`agents/frontmatter.py`、`agents/ui.py`：记忆、MCP 客户端、子 Agent、会话、frontmatter 解析、终端 UI

评审和修改的边界就是 `agents/` 加上仓库根的配置文件（`.mcp.json`、`requirements.txt`、`Dockerfile`、`.env.example`）。

## .otter/ 是产品运行时资产，不是本仓库的评审门禁

`.otter/` 目录是 Otter Code 这个产品自身在运行时读写的资产，不是给进驻本仓库的 coding agent 用的规则或门禁：

- `.otter/skills/<skill_name>/SKILL.md`：项目级 Skills，由 Otter Code Runtime 加载和自进化写入
- `.otter/skill-evolution/`：Skills 自进化的审计产物（usage.jsonl、provenance、版本快照等）

编码约束：SKILL.md 一律 UTF-8 编码。Windows 下默认 GBK 解码会静默丢弃含中文的技能文件（见 `skills.py` `_parse_skill_file` 的 `read_text(encoding="utf-8")`）。

进驻 agent 不要把 `.otter/skills` 下的 SKILL.md 当作本仓库的工程规范来遵守，也不要在评审中把它们当作代码质量门禁；它们是产品功能的数据。修改 `.otter/` 内容属于产品行为验证范畴，不属于常规代码变更。

## 危险区：agents/tools.py 的权限模式语义

`tools.py` 中的权限系统是安全关键路径，改动前必须先读懂现有语义（`PermissionMode`、`check_permission`、`DANGEROUS_PATTERNS`、`is_dangerous`、`HARD_BLOCKLIST`、`is_hard_blocked`、`_is_within_workspace`）：

- 权限模式共五种：`default`、`plan`、`acceptEdits`、`bypassPermissions`、`dontAsk`（`tools.py` 中 `PermissionMode` 定义）
- `bypassPermissions`（对应 CLI `--yolo`）跳过用户确认，但仍执行硬黑名单检测；`HARD_BLOCKLIST` 命中的不可逆命令（`rm -rf /`、`mkfs`、`dd of=/dev/`、`> /dev/sd`、Windows `Format-Volume`/`Clear-Disk` 等）一律拒绝，`bypassPermissions` 也不例外
- `plan` 模式：编辑类工具只允许写计划文件，其余编辑与 `run_shell` 一律拒绝；`run_verification` 是只读操作，plan 模式允许
- `acceptEdits`（对应 CLI `--accept-edits`）：自动放行编辑类工具，但越界路径（workspace 外）与硬黑名单命令仍拒绝
- `default`：危险命令（匹配 `DANGEROUS_PATTERNS`，如 `rm`、`git push/reset/clean`、`curl|sh`、`pip/npm install`、`Stop-Process` 等）、新建文件、越界路径、`skill_evolve`、`skill_create` 需要用户确认
- `dontAsk`（对应 CLI `--dont-ask`）：所有本应确认的操作自动拒绝
- 用户/项目 `.otter/settings.json` 中的 `permissions.deny` 规则优先于 `allow`，两者都优先于模式默认行为（`bypassPermissions` 与 `HARD_BLOCKLIST` 除外）
- workspace 路径沙箱：`read_file`/`write_file`/`edit_file`/`file_stats`/`list_files`/`grep_search` 只能访问 cwd 子树内路径；越界时 default 需确认、dontAsk/plan/acceptEdits 拒绝、bypassPermissions 放行
- 权限决策审计：`check_permission` 每次决策（allow/deny/confirm）写入 `.otter/logs/permissions.log`，供事后追溯
- 子代理权限：`agent`/`skill` fork 子代理统一使用 `acceptEdits`（`_sub_agent_permission_mode`），不再继承 `bypassPermissions`；子代理 `confirm_fn` 为空时危险操作自动拒绝

对这些分支的任何改动都必须补充或更新 `tests/test_tools_permissions.py` 中的对应用例，不允许无回归信号地修改。`agent.py` 中涉及子代理权限、验证、重试、超时的改动必须补充 `tests/test_agent_harness_enhance.py` 对应用例。

## 变更后必须执行的验证命令

在仓库根依次运行：

```bash
python -m compileall agents
python -m unittest discover -s tests
```

第一条做编译级语法检查，第二条运行权限与危险命令拦截的回归测试。两条命令均只依赖标准库，全部通过后改动才算完成。

运行时冒烟验证（可选，需配置 `.env`）：

```bash
python -m agents.main "总结这个项目的目录结构和核心模块"
```
