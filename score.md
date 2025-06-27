o3
（先给结论，再展开实现细节）

一句话回答  
→ 采用「数据源收集 → Redis Stream 统一事件总线 → 多 Agent（Ingest / Prioritize / Schedule / Review / Act） → 双向同步 Calendar / Reminder / Markdown」的流式架构，外围用 Typer-CLI + 插件系统暴露人机接口；内部用 Pydantic-based contract + pytest-asyncio 保障可测试性；流程全部容器化 + GitHub Actions CI；权限通过动态策略树渐进放权。下面给出端到端的拆解与落地路线。

────────────────────────
0. 你要解决的真实问题（复述需求）
────────────────────────
1) 多源任务/日程碎片化 → 统一建模与同步。  
2) 触发需人工，缺少“事件驱动、自动行动”链路。  
3) 脚本零散，无测试 / 无 DevOps / 难扩展。  
4) 想把 Claude 等 LLM 融入决策、计划、复盘，且要可插拔、多平台。

────────────────────────
1. 顶层架构总览
────────────────────────
```
┌────────┐      Redis Streams         ┌─────────────┐
│Sources │  ───────────►   ingest.*   │ Ingest Agent│
│(IMAP   │                           └───▲──────────┘
│ Google │                              │ normalized event
│  Cal   │                              │
│ mdNote │     priority.*             ┌─┴────────────┐
│ etc.)  │◄─────────────┐             │ Prioritizer  │◄─── Claude API
└────────┘              │             └─▲────────────┘
      ▲ capture.*       │schedule.*     │
      │                  ▼               │
┌─────┴──────┐      ┌───────────┐    ┌──┴────────────┐
│ Typer CLI  │◄────▶│ Event Bus │◄──┤ Scheduler     │───► Google/Apple Cal
└────────────┘      └───────────┘    └───▲───────────┘
      ▲                                   │review.*
      │                      ┌───────────┴─────────┐
      │                      │ Review / Report Agent│──► Markdown, Slack
      │                      └──────────────────────┘
      ▼
Plugins (Hammerspoon, Raycast, VSCode, Shortcuts...)
```
• 所有内部通信只走 Redis Streams（或 NATS JetStream），天然支持回溯 / 消费组 / 观测。  
• Agent 皆为独立容器；Agent-Claude 把 prompt 模版、retry、流式响应包装起来。  
• “渐进式放权”＝ Scheduler / Actuator 在写操作前检查 Policy Engine（OPA）是否允许；策略文件放 repo，可 PR 审核。  

────────────────────────
2. 代码库与目录布局（monorepo）
────────────────────────
```
galaxy/
├─ core/              # 领域模型 & 共用工具
│   ├─ models.py      # Pydantic BaseModel，事件 schema
│   └─ bus.py         # 读写 Redis Stream 的统一封装
├─ agents/
│   ├─ ingest/
│   │   ├─ email.py   # IMAP -> Event
│   │   └─ calendar.py
│   ├─ prioritize/
│   ├─ schedule/
│   ├─ review/
│   └─ act/
├─ cli/               # Typer apps，提供 capture / ask / debug
│   └─ __init__.py
├─ plugins/           # 可插拔外设适配层
│   ├─ hammerspoon/
│   └─ raycast/
├─ tests/
├─ infra/             # docker-compose, terraform, gha
└─ CHANGELOG.md
```
• 单一 `pyproject.toml`，依赖 pin 在 `uv.lock`; `make dev`=uv venv。  
• CLI 与 Agent 均 import `core`，确保 schema 复用。  
• 新数据源 → 丢到 `agents/ingest/$source.py`，注册到 `ENTRYPOINT := ingest.email:main`。

────────────────────────
3. 关键模块设计
────────────────────────
3.1 事件 schema（简化示例）
```python
class RawEvent(BaseModel):
    id: str
    source: Literal["email", "gcal", "markdown"]
    payload: dict
    ts: datetime

class Task(BaseModel):
    title: str
    due: datetime | None
    tags: list[str] = []
    priority: int = 3
    origin: RawEvent
```
• 每一步转换都产生新 Stream：`raw.* -> task.* -> schedule.*`，可观测、易 debug。  
• 使用 JSON-Lines 存储，保证跨语言消费。

3.2 Claude 调用包装
```python
async def llm_call(prompt: str, model="claude-3.5-sonnet") -> str:
    async with httpx.AsyncClient(timeout=60) as cli:
        r = await cli.post("https://api.anthropic.com/v1/messages",
                           json={"model": model, "stream": True, ...})
        return await parse_stream(r)
```
• Prompt 模版在 `prompts/ingest_email.jinja`; 测试时用 vcr.py 录制。

3.3 Prioritizer Agent
```python
async for ev in bus.subscribe("task.*"):
    annotated = await llm_call(PRIORITY_TMPL.render(task=ev))
    priority = extract_priority(annotated)  # 提示工程 or 正则
    await bus.publish("priority.*", ev.model_copy(update={"priority": priority}))
```

3.4 Scheduler Agent  
• 调 Google / Apple Calendar API；若权限不足发送 inbox “需要授权” 通知。  
• Policy = Rego 规则，示例：禁止在 22:00-07:00 自动排 meeting。

────────────────────────
4. 渐进式自动执行 & 权限
────────────────────────
Stage 0: 只读 → Claude 出计划，CLI 询问确认。  
Stage 1: 高置信度（LLM self-eval ≥ 0.9） + 白名单时间段 → 自动写入 Calendar。  
Stage 2: 自动发送邮件 / Slack DM 调整日程，需 OPA 标签 `auto_exec=true`。  
策略和阈值版本化在 `policy/`，每次变更走 PR + Code-Review + CI 测试 (`opa test .`）。

────────────────────────
5. DevOps / CI/CD
────────────────────────
• pre-commit：ruff, black, mypy, shellcheck, dot-env-linter。  
• GitHub Actions Matrix：  
  - `ci.yaml`:  macOS, Linux, Windows → `pytest -q && mypy && opa test`.  
  - `docker-publish.yaml`: tag push → build multi-arch image → ghcr.io。  
• Release：Typer CLI 打 Homebrew tap；Agents 镜像用 `docker-compose.yaml` 或 `k8s helm chart`。  
• Observability：Grafana + Prometheus exporter for Redis; Loki 收集 stdout logs。

────────────────────────
6. 测试策略
────────────────────────
1) 单元：Pydantic schema 验证、LLM prompt output 解析 (`pytest-snapshots`)。  
2) 集成：`pytest-asyncio` + `fakeredis` → 模拟 Event Bus。  
3) E2E：`docker-compose -f infra/ci.yml` 起所有 Agents → 送入邮件样例 → 断言 Calendar mock 收到。  
4) 回归：用 `git bisect run pytest` + GitHub Action 自动定位坏 commit。

────────────────────────
7. 渐进实施里程碑
────────────────────────
Day 0  ：搭目录 + uv + Typer CLI `galaxy capture/ask`.  
Day 3  ：Redis Stream + Ingest(Email, Markdown) Agent；fake data 流通。  
Day 7  ：Prioritizer + Scheduler 写 Google Cal（只读模式）。  
Day 14 ：Review Agent → 自动生成日报 Markdown + Slack 通知。  
Day 21 ：OPA Policy + Stage-1 自动排程；GitHub Actions 完整 CI。  
Day 30 ：插件化外设 (Hammerspoon / Raycast) + Homebrew 发布；开始阶段性放权到 Stage-2。

────────────────────────
8. 你可能没想到的扩展
────────────────────────
• “LLM 发条码”：用 self-eval + 语义缓存（SQLite pgvector）避免重复生成。  
• 本地 Llama3-8B 做鲁棒 fallback；离线时依旧可排程。  
• 任务 cost-aware：Scheduler 评估 token 预算，忙时自动降到本地模型。  
• 语义 Clipboard：检测剪贴板内容自动推送 `raw.clipboard.*` Stream。  
• iOS Shortcuts: 手机分享菜单一键发送到 `capture` Endpoint，真正全平台。

────────────────────────
9. 下一步可执行 Action
────────────────────────
1) `uv venv .venv && uv pip install typer[all] redis[async]==5.0.* pydantic[dotenv]`  
2) `galaxy init` 脚手架：自动生成上述目录 + `pyproject.toml` + pre-commit。  
3) 写第一条 `agents/ingest/email.py`，用 IMAP IDLE 拉新邮件；事件格式参照 `core/models.py`。  
4) push 到 GitHub，开 `actions/ci.yaml`。  
5) 30 分钟内即可看到第一个 Markdown inbox 被自动转事件、CLI 展示。

────────────────────────
参考 & 灵感（Sources）
────────────────────────
• Redis Streams 官方文档  
• Temporal / Prefect 任务编排对比  
• Anthropic Claude API Guide  
• Google Calendar API Best Practices  
• OPA Policy Design Patterns  

————————————  
以上即完整解决方案：用 Redis Stream 事件总线 + 多 Agent 架构，将 LLM 融入优先级判定与排程；Typer-CLI 做入口，OPA 策略把控自动执行权限；配套测试与 CI 保障质量。按里程碑滚动实施即可。

r1