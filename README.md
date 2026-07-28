# Otter Code

Otter Code 是一个基于 Python 实现的**自进化 Harness Agent**。它不是简单的命令行聊天工具，而是一个可运行、可阅读、可扩展的本地 Coding Agent Runtime：统一编排大模型推理、工具调用、文件编辑、Shell 执行、权限控制、长期记忆、Skills、自进化、MCP 外部工具、子 Agent 和会话恢复。

项目重点是 **Harness**：模型只负责推理和提出工具调用意图，真正的环境操作由 Otter Code Runtime 统一做权限判断、工具执行、结果回写、上下文压缩和经验沉淀。它适合学习 Claude Code 类工具的底层机制，也适合作为个人 Coding Agent、项目分析助手或领域 Agent 的二次开发基础。

## 核心亮点

- **自进化 Harness Agent**：从用户反馈中自动抽取可复用规则，新增或合并到 SKILL.md，让 Agent 能随着使用持续沉淀能力。
- **上下文工程实践**：动态分层组装 system prompt；deferred tools + tool_search 实现工具渐进式披露；工具使用规范只写在工具描述里，保证指令单一来源；技能正文经 BM25 检索按需注入，不进 system prompt。
- **完整 Agent Loop**：模型请求、tool call 解析、权限检查、工具执行、tool result 回写、继续推理、会话保存形成闭环。
- **OpenAI / Anthropic 双协议**：支持 OpenAI-compatible 和 Anthropic-compatible 接口，便于接入不同模型服务或代理网关。
- **工具系统与权限控制**：支持读写文件、精确编辑、代码搜索、Shell 命令、Skill 调用、子 Agent 和 MCP 工具；Plan Mode 下阻断写操作和 Shell。
- **Skills 体系**：通过项目级和用户级 SKILL.md 保存可复用任务方法，支持检索、调用、inline / fork 执行和版本化演化。
- **长期 Memory**：按项目路径 hash 隔离记忆，保存用户偏好、项目背景、历史决策和参考资料。
- **MCP 外部工具扩展**：自研 stdio JSON-RPC MCP Client，把外部 MCP Server 工具包装为 `mcp__server__tool`。
- **子 Agent**：支持 explore、plan、general 以及自定义子 Agent，用隔离上下文完成探索、规划或局部任务。
- **会话恢复和上下文压缩**：自动保存 session，支持 `--resume`、`/compact`，并对大工具结果做截断或持久化。

## 项目架构

核心运行链路：

```
用户输入
  -> agents/main.py
  -> Agent.chat()
  -> 构建 Prompt / 检索 Skills / 预取 Memory / 初始化 MCP
  -> 调用 OpenAI-compatible 或 Anthropic-compatible 模型
  -> 模型返回文本或 tool call
  -> Harness 做权限检查
  -> 执行工具 / Skill / MCP / 子 Agent
  -> tool result 回写模型
  -> 保存 Session
  -> 后台执行 Skill usage tracking 和 online skill evolution
```

system prompt 由 `prompt.py: build_system_prompt()` 每次启动时动态组装：

```
内嵌模板（角色/任务/安全/风格约束）
  + 环境信息（cwd / date / platform / shell / git 分支与状态）
  + CLAUDE.md 层级加载（cwd 向上逐级收集，支持 @include 递归解析）
  + .otter/rules/*.md 项目规则
  + 记忆索引（MEMORY.md，注入前裁剪）
  + 技能清单 + 子 Agent 清单 + deferred 工具名列表
```

## 目录结构

```
OtterCode/
├── agents/
│   ├── main.py                    # CLI 入口、REPL、参数解析
│   ├── agent.py                   # Agent Runtime、模型调用、工具调度、上下文压缩
│   ├── tools.py                   # 内置工具、权限系统、deferred 工具与 tool_search
│   ├── prompt.py                  # System prompt 动态构建
│   ├── skills.py                  # Skills 加载、BM25 检索、执行、创建和演化封装
│   ├── online_skill_evolution.py  # 在线 Skill 抽取和 add/merge/discard 决策
│   ├── skill_evolution.py         # Skill 落盘、版本快照、审计统计
│   ├── memory.py                  # 长期记忆系统
│   ├── mcp_client.py              # MCP stdio JSON-RPC 客户端
│   ├── subagent.py                # 子 Agent 配置
│   ├── session.py                 # 会话保存与恢复
│   ├── frontmatter.py             # SKILL.md frontmatter 解析
│   └── ui.py                      # 终端 UI 输出
├── .otter/
│   ├── skills/                    # 项目级 Skills
│   └── skill-evolution/           # Skills 自进化审计产物
├── .mcp.json                      # MCP Server 配置
├── Dockerfile
├── requirements.txt
└── README.md
```

## 快速启动

### 准备环境

- Python 3.11+
- Windows / macOS / Linux
- Git
- Node.js（可选，MCP 的 context7 / playwright 依赖 npx；没装不影响启动）
- 一个 OpenAI-compatible 或 Anthropic-compatible 模型接口

安装依赖：

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 配置 .env

项目会自动读取当前目录或父目录中的 `.env`，且**项目 .env 中的显式配置优先于系统环境变量**。

Anthropic-compatible 示例（DeepSeek 官方 Anthropic 端点）：

```
APIKEY=sk-your-api-key
API=https://api.deepseek.com/anthropic
MODEL=deepseek-v4-pro
```

OpenAI-compatible 示例：

```
OPENAI_API_KEY=sk-your-api-key
OPENAI_BASE_URL=https://your-host/v1
MODEL=your-model
```

协议判断规则：

- `API` 或 `--api-base` 路径包含 `/anthropic` 时，按 Anthropic-compatible 调用。
- 否则有 OpenAI base URL 时，按 OpenAI-compatible 调用。
- `--model` 会覆盖 `.env` 中的 `MODEL`；两者都未设置时默认 `deepseek-v4-pro`。

### 启动 REPL

```bash
python -m agents.main
```

启动后直接输入任务，例如：

```
阅读这个项目，告诉我 Agent Loop 是怎么跑起来的
```

### 执行一次性任务

```bash
python -m agents.main "总结这个项目的目录结构和核心模块"
```

### 使用 Plan Mode

Plan Mode 适合重构、复杂修复和多文件修改。它会先只读分析和写计划，用户审批后再执行。

```bash
python -m agents.main --plan "分析 Skills 检索逻辑应该如何优化"
```

REPL 中也可以输入 `/plan` 切换。

### 恢复最近会话

```bash
python -m agents.main --resume
```

## 如何让项目自动沉淀并进化 Skills

Otter Code 的核心特色是**自进化 Skills**：从用户明确反馈中抽取未来可复用的规则，并自动新增或合并到项目级或用户级 SKILL.md。

### 开启自动自进化

默认 `OTTER_AUTO_SKILL_EVOLUTION` 是开启的。为了明确配置，建议在 `.env` 中写：

```
OTTER_AUTO_SKILL_EVOLUTION=1
OTTER_AUTO_SKILL_TARGET=project
```

- `OTTER_AUTO_SKILL_EVOLUTION=1`：启用在线 Skill 自进化。
- `OTTER_AUTO_SKILL_TARGET=project`：自动新增的 Skill 写入当前项目 `.otter/skills/`；设为 `user` 则写入 `~/.otter/skills/`，所有项目共享。

### 用允许写入的权限模式启动

后台自动写入 Skill 仅在 `--accept-edits` 或 `--yolo` 模式下放行；default 模式需用 `/extract_now` 手动触发交互确认。日常推荐：

```bash
python -m agents.main --accept-edits
```

不建议长期使用 `--yolo`，它会跳过所有确认。

### 给出可复用反馈

自进化只沉淀稳定、明确、未来同类任务仍适用的规则。

不适合沉淀（一次性任务）：

```
帮我写一篇 500 字政府报告。
```

适合沉淀（可复用规则）：

```
以后写政府报告、工作汇报、调研材料时，默认先给可直接使用的初稿，不要连续追问；
结构按"标题、背景、主要情况、问题分析、工作举措、下一步计划"组织，语言要正式克制。
```

### 自进化链路如何工作

```
第 N 轮用户任务
  -> Agent 输出结果
  -> 保存 pending extraction window
第 N+1 轮用户反馈
  -> 合并进上一轮 window
  -> online_ingest()
  -> Extractor 抽取候选 Skill
  -> Maintainer 判断 add / merge / discard
  -> create_skill_file() 或 evolve_skill_file()
  -> 写入 SKILL.md
  -> 记录 provenance、usage stats 和版本快照
```

审计产物全部在 `.otter/skill-evolution/`：`usage.jsonl`、`online_provenance.jsonl`、`online_skill_provenance.json`、`skill_usage_stats.json`、`history/`、`pruned/`。

### 手动管理 Skills

```
/extract_now 这是一个可复用的写作规则      # 立即抽取当前对话窗口
/skill-create <name> | <description> | <when-to-use> | <instructions>
/skill-evolve <skill-name> <durable lesson>
/skill-feedback <skill-name> <rating> [note]
/skills                                    # 列出可用 Skills
/skill-stats                               # 查看使用和演化统计
```

## 常用命令

### CLI 参数

| 参数 | 功能 |
|---|---|
| `--model, -m` | 指定模型，覆盖 `.env` 中的 MODEL |
| `--api-base` | 覆盖 API base URL |
| `--plan` | 只读规划模式 |
| `--accept-edits` | 自动允许编辑类操作，推荐用于自动沉淀 Skills |
| `--yolo, -y` | 跳过确认 |
| `--dont-ask` | 自动拒绝需要确认的操作，适合 CI |
| `--thinking` | 启用扩展思考（仅 Anthropic 协议） |
| `--resume` | 恢复最近会话 |
| `--max-cost` | 费用上限（USD） |
| `--max-turns` | 最大 agentic turns |

### REPL 命令

| 命令 | 功能 |
|---|---|
| `/clear` | 清空对话历史 |
| `/plan` | 切换 Plan Mode |
| `/cost` | 显示 token 和费用估算 |
| `/compact` | 手动压缩上下文 |
| `/memory` | 列出长期记忆 |
| `/skills` | 列出可用 Skills |
| `/skill-stats` | 查看 Skill 使用和演化统计 |
| `/extract_now [hint]` | 抽取当前 pending window |
| `/skill-feedback <skill> <rating> [note]` | 记录 Skill 反馈 |
| `/skill-evolve <skill> <lesson>` | 手动演化 Skill |
| `/skill-create <name> \| <desc> \| <when-to-use> \| <instructions>` | 手动创建 Skill |
| `/<skill-name> [args]` | 直接调用 user-invocable Skill |
| `exit` / `quit` | 退出 |

## Skills 是什么

Skill 是一个可复用能力说明文件，即带 frontmatter 的 SKILL.md。它保存的不是某次任务的具体内容，而是未来同类任务可复用的方法、规范、偏好或流程。

```
用户级：~/.otter/skills/<skill_name>/SKILL.md    （优先级更高）
项目级：<project>/.otter/skills/<skill_name>/SKILL.md
```

示例：

```markdown
---
name: code_review
description: Review code changes with a bug-risk-first mindset.
when-to-use: When the user asks for code review.
user-invocable: true
context: inline
---

# Workflow

1. Read the relevant code first.
2. Lead with bugs, regressions, security risks, and missing tests.
3. Cite file paths and line numbers when possible.
4. Keep summary secondary to findings.
```

Skill 和 Memory 的区别：

| 类型 | 保存内容 |
|---|---|
| Memory | 用户偏好、项目事实、历史决策、参考资料（"知道什么"） |
| Skill | 可复用任务流程、输出规范、领域方法（"怎么做"） |

## MCP 支持

Otter Code 支持 MCP 外部工具扩展。MCP Server 通过 stdio JSON-RPC 暴露工具，Otter Code 将其包装为 Agent 可调用工具，命名规则为 `mcp__<serverName>__<toolName>`。

配置来源（后读取的覆盖同名 Server）：

```
~/.otter/settings.json
<project>/.otter/settings.json
<project>/.mcp.json
```

配置示例：

```json
{
  "mcpServers": {
    "example": {
      "command": "python",
      "args": ["server.py"],
      "env": {}
    }
  }
}
```

Windows 下 `npx` 等命令会自动经 PATH 解析完整路径，无需写 `npx.cmd`。

## Docker 运行

构建镜像：

```bash
docker build -t otter-code .
```

启动交互式会话：

```bash
docker run --rm -it \
  --env-file .env \
  -v "$PWD:/workspace" \
  -v otter-code-sessions:/root/.otter-code \
  -v otter-code-memory:/root/.OtterCode \
  otter-code
```

允许自动沉淀 Skills：

```bash
docker run --rm -it \
  --env-file .env \
  -e OTTER_AUTO_SKILL_EVOLUTION=1 \
  -e OTTER_AUTO_SKILL_TARGET=project \
  -v "$PWD:/workspace" \
  -v otter-code-sessions:/root/.otter-code \
  -v otter-code-memory:/root/.OtterCode \
  otter-code --accept-edits
```

镜像内置 Node.js 22 与 Playwright Chromium，MCP 的 context7 / playwright 开箱可用。

## 重要数据路径

| 数据 | 路径 |
|---|---|
| 项目级 Skills | `.otter/skills/<skill_name>/SKILL.md` |
| 用户级 Skills | `~/.otter/skills/<skill_name>/SKILL.md` |
| Skills 自进化审计 | `.otter/skill-evolution/` |
| 长期记忆 | `~/.OtterCode/projects/<project_hash>/memory/` |
| 会话历史 | `~/.otter-code/sessions/` |
| 大工具结果 | `~/.otter-code/tool-results/` |
| Plan Mode 计划 | `~/.otter/plans/` |

## 适合如何使用

Otter Code 适合：

- 学习 Claude Code 类工具的 Agent Loop 和工具调用机制。
- 学习上下文工程：动态 prompt 组装、渐进式披露、指令单一来源、按需检索注入。
- 学习如何做本地文件编辑型 Agent 的权限控制。
- 学习 Skills、Memory、MCP、子 Agent 如何接入同一个 Runtime。
- 扩展成个人 Coding Agent 或带长期记忆和可复用经验沉淀的领域 Agent。

如果只看一个核心点：Otter Code 是一个会把稳定用户反馈沉淀成 Skills 的**自进化 Harness Agent**。
