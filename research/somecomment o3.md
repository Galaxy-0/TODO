以下内容是一份「Mac + Claude Code 协同」的生产力最大化方案，目标是在不更换硬件的前提下，让你的整套开发-思考-执行流程做到：

• 打开电脑即可进入“深度工作”环境  
• 任意窗口内容一键发送给 Claude 分析或改写  
• 所有脚本均跨平台、可测试、可持续集成  
• 桌面层面零鼠标——全键盘驱动

────────────────────────
一、系统级基础设施（15 分钟完成）
────────────────────────
1. Homebrew Brewfile（集中管理所有依赖）  
   建议内容：gnu-sed、ripgrep、fd、fzf、uv、python@3.11、gnu-time、reattach-to-user-namespace。  
   ```bash
   brew bundle --file=~/dotfiles/Brewfile
   ```

2. 终端与 Shell  
   • iTerm2 + zsh（或 fish），配合 `oh-my-zsh` + `fzf-tab` + `zsh-autosuggestions`。  
   • 在 `.zshrc` 加入：
     ```zsh
     eval "$(fzf --zsh)"          # ⌃T 文件/命令模糊查找
     export EZA_COLOR="..."
     alias ls="eza --icons"
     ```

3. Direnv + uv：目录切换自动加载虚拟环境  
   ```bash
   echo "layout python3 && uv venv .venv" > .envrc
   direnv allow
   ```

4. 键盘/窗口管理  
   • Yabai（平铺）+ Skhd（快捷键）或 Rectangle（一次性安装即用）。  
   • 常用键：`⌥⌘HJKL` → 左/下/上/右分屏；`⌥⌘F` → 全屏。

────────────────────────
二、Claude Code 深度集成（≈1 小时）
────────────────────────
1. Raycast Script Command（最快）  
   • `raycast/scripts/ask_claude.sh`：  
     ```bash
     #!/bin/zsh
     pbpaste | claude-cli --model claude-3.5-sonnet --stdin
     ```  
   • 选中文本 → `⌘⇧C` 复制 → `⌘⇧Space` 打开 Raycast → “Ask Claude” → 回车。  
   • Raycast AI Pro 可省略这步，但自建脚本无需订阅费且可换模型。

2. VS Code / Cursor 内一键上下文投递  
   • 安装「Run Commands」扩展，绑定快捷键  
   ```
   cmd+k cmd+a → echo "{file}:{line}" | claude-cli --stdin
   ```  
   • 选中文段自动作为 prompt，Claude Code 侧边栏输出补全/重写。

3. 系统级 Floating Prompt（Hammerspoon）  
   • `hyper`（⌃⌥⇧⌘Space）弹出 TextArea → 输入或粘贴 → `⌘↵` 发送至 Claude API → 原地显示结果。  
   • 配方：
     ```lua
     hs.hotkey.bind({'ctrl','alt','shift','cmd'}, 'space', function() ... end)
     ```

4. Apple Shortcuts + “朗读／语音输入”  
   • 镜像 iOS「听写」体验：按 Fn 2 次 → dictation → 自动发送到 `claude-cli` → 结果以通知弹回。  
   • 有利于走路、做饭时快速 capture 想法。

────────────────────────
三、脚本与 CLI 统一（替换现有 .sh）
────────────────────────
1. Typer-based `galaxy` CLI  
   ```
   galaxy/
     cli.py           # typer app
     commands/
       capture.py     # 替换 capture.sh
       analyze.py     # 替换 process.sh
   ```
   • 优势：同一代码库、单元测试、跨平台。  
   • 示例：
     ```python
     import typer, subprocess, json, os
     app = typer.Typer()

     @app.command()
     def capture(mode: str = typer.Option("auto")):
         ...

     @app.command()
     def ask(prompt: str = typer.Argument(None)):
         ctx = prompt or os.popen('pbpaste').read()
         out = subprocess.run(["claude-cli", "--stdin"], input=ctx, text=True,
                              capture_output=True)
         typer.echo(out.stdout)
     if __name__ == "__main__":
         app()
     ```

2. pre-commit 钩子  
   • `ruff`, `black`, `shellcheck`, `dotenv-linter`；提交前 5 秒完成质量检查。  
   ```yaml
   repos:
   - repo: https://github.com/astral-sh/ruff-pre-commit
     rev: v0.4.3
     hooks: [ { id: ruff } ]
   ```

3. GitHub Actions  
   • `ci.yml`：macOS-latest 触发 `pytest -q` + `ruff --fix`。  
   • `release.yml`：打 tag 自动发布 Homebrew tap（若 CLI 很多机器要装）。

────────────────────────
四、输入→事件→决策的全键盘流
────────────────────────
1. **Capture（⌘⇧S）**  
   • 自动检测活跃窗口：若是浏览器抓 URL 与选中；若是终端抓命令；否则抓选中文本。  
   • 写入 `inbox.md` + Raycast toast 成功提示。

2. **Process（⌘⇧P）**  
   • 打开 Spotlight-like palette（Typer CLI + fzf）列出未处理条目。  
   • 选择后自动调用 Claude 进行 quick / deep analysis 并输出至 `analysis/XX.md`。

3. **Inbox Zero（⌘⇧I）**  
   • Slack Bot `/inbox`: 读取 Redis Stream；`✅` 写状态回 Markdown，中间不用鼠标。  
   • 完成后自动归档 `analysis/*` 到 `docs/archives/`.

4. **每日复盘（22:30, launchd）**  
   ```
   <plist>
     <key>ProgramArguments</key>
     <array>
       <string>/Users/you/.venv/bin/galaxy</string>
       <string>review</string>
     </array>
     <key>StartCalendarInterval</key><dict><key>Hour</key><integer>22</integer><key>Minute</key><integer>30</integer></dict>
   </plist>
   ```
   • 复盘结果推送 Raycast Notification + 新建 Obsidian 日记。

────────────────────────
五、硬件 & macOS 原生功能补强
────────────────────────
1. Sidecar + iPad 作第二屏：Hammerspoon 脚本自动布局 Obsidian+终端。  
2. 快速屏幕录制 & OCR：CleanShot X ⌘⇧2 → 自动上传 S3 → 返回 URL 给 Claude 分析。  
3. Spotlight 替换：Raycast / Alfred + 自定义 Workflows（调用 `galaxy ask`）。  
4. Touch ID sudo：`sudo pam-auth-update --enable-touch-id`，少输入密码。  
5. 降温 & 延长续航：`brew install --cask stats` + 快速降频脚本（Apple Silicon 能耗管理）。

────────────────────────
六、潜在的新玩法（“你可能没想到”）
────────────────────────
• **语义剪贴板**：监控 clipboard，若检测到 code diff / URL / Meeting Link → Claude 自动生成摘要或 TODO 并入 inbox。  
• **Focus Mode 自动化**：进入日历标记 Deep-Work 时段，自动调用 `yabai -m config window_border off` + 关闭通知 + Claude 自动屏蔽高 token 消耗模型。  
• **USB-C KVM 一线切换**：当检测到 keyboard layout 变更（Karabiner-Elements event）时触发脚本，使 Claude 输出“当前项目状态”语音播报，降低上下文切换时间。  
• **小模型离线加速**：在 M-series 芯片本地跑 Ollama（Llama 3 8B-Instruct），常规改写无需云端，Claude 专攻深度分析。

────────────────────────
七、实施优先级 & 时间线
────────────────────────
1. **0-Day**：装 Brewfile、Yabai、Raycast；复制 `ask_claude.sh` 并绑定快捷键。  
2. **+1 Day**：Direnv+uv、Typer CLI 替代 capture / process 脚本；pre-commit + CI。  
3. **+3 Days**：Slack `/inbox` Bot、Redis Stream 事件流、launchd 复盘任务。  
4. **+7 Days**：Hammerspoon Floating Prompt、多模态 Capture（OCR、语音）。  
5. **+14 Days**：本地小模型、FocusMode 自动化、语义剪贴板、Dashboard。

────────────────────────
一句话总结
────────────────────────
通过「Homebrew +Brewfile 打底 → uv+Typer 统一 CLI → Raycast/Hammerspoon 全局快捷 → Redis Stream 事件驱动 → Slack Inbox & launchd 复盘」这一整链，你的 Mac 将从“跑脚本的工具箱”进化为“无缝协同 Claude Code 的智能生产力主控台”，从捕获-分析-执行-复盘全程几乎无需离开键盘，真正做到注意力最小切换、深度工作最大化。

o4-mini
您希望在 macOS 上构建一个“无须人工触发、完全事件驱动”的个人助理系统，核心依赖 Claude Code 等大模型，能够自动：

1. 多源（Calendar、邮件、Markdown 等）采集任务／日程  
2. 通过事件总线（Event Bus）触发多 Agent 流程：  
   - Ingestion（数据摄取）  
   - Prioritization（优先级判断）  
   - Scheduling（排期与提醒）  
   - Review（复盘）  
3. 双向同步日程与提醒（Calendar ↔ 本地/远程储存）  
4. 提供跨平台的 Typer CLI + 插件化扩展  
5. 支持渐进式“自动执行”权限管控  
6. 拥有完整的测试、版本管理与 CI/CD 流程  

下面是一个可落地的整体方案与分步实施计划。

---

## 一、技术选型与架构概览

• 语言与框架  
  - Python 3.11+，使用 Typer 打造 CLI（可跨平台）  
  - 异步 I/O：`asyncio` + `aioredis`（事件总线）  

• 事件总线  
  - Redis Streams 作为核心 Event Bus，保证可靠投递与持久化  
  - 封装成 `event_bus.py`，提供 `publish(channel, payload)`、`consume(channel)` 接口  

• Agent 模块  
  - 各 Agent 以命令或服务形式存在：  
    1. `ingest`：接入 Google Calendar API、IMAP/SMTP、Markdown 文件监听  
    2. `prioritize`：调用 Claude Code 或本地规则对事件打分  
    3. `schedule`：自动写入系统日历、发送通知、生成本地提醒  
    4. `review`：每日/每周复盘，推送分析报告  

• 插件化  
  - 利用 Python `entry_points` 机制，允许第三方插件在 `galaxy.plugins` 下注册新 Agent 或数据源  

• 数据存储  
  - 轻量级：SQLite（初期）→ 可扩展为 PostgreSQL  
  - 使用 `SQLModel` 或 `Tortoise ORM` 统一建模  

• 自动执行权限  
  - 用户按需授予特定命令（如删除邮件、发送通知）的 macOS 权限  
  - 抽象成“策略引擎”（Policy Engine），运行前判读并提示用户确认  

---

## 二、项目结构（建议）

```
.
├── .envrc                      # Direnv + uv 虚拟环境
├── pyproject.toml              # Poetry/Flit/Setuptools 配置
├── setup.cfg                   # Black、isort、flake8 等
├── .pre-commit-config.yaml     # ruff、black、shellcheck、dotenv-linter
├── CHANGELOG.md
├── README.md
├── galaxy/                     # 主应用包
│   ├── __init__.py
│   ├── cli.py                  # Typer CLI 入口
│   ├── event_bus.py            # 发布/订阅封装
│   ├── config.py               # 配置加载（dotenv / YAML）
│   ├── agents/
│   │   ├── ingestion.py
│   │   ├── prioritization.py
│   │   ├── scheduling.py
│   │   └── review.py
│   └── plugins/                # 第三方插件目录
│
├── tests/                      # Pytest 测试用例
│   ├── test_event_bus.py
│   ├── test_ingestion.py
│   └── …
└── .github/
    └── workflows/
        ├── ci.yml             # macOS + Linux CI: pytest, ruff, black
        └── release.yml        # 打 tag → 发布 PyPI / Homebrew Tap
```

---

## 三、关键实现要点

1. **Direnv + uv**  
   在项目根目录建立 `.envrc`：  
   ```bash
   layout python3
   uv venv .venv
   ```  
   这样 `cd` 进去自动激活 `.venv`。

2. **Typer CLI**  
   `galaxy/cli.py` 示例：  
   ```python
   import typer
   from .event_bus import publish
   from .agents import ingestion, prioritization, scheduling, review

   app = typer.Typer(help="GalaxyAI 事件驱动个人助理")

   @app.command()
   async def ingest():
       """摄取多源数据，发布{“source”:…, “payload”:…}事件"""
       await ingestion.run()

   @app.command()
   async def prioritize():
       """从事件总线拉取待处理事件，并打标签/打分"""
       await prioritization.run()
   ...
   if __name__ == "__main__":
       app()
   ```

3. **Event Bus**  
   ``galaxy/event_bus.py``  
   - 使用 `aioredis` 连接 Redis  
   - `async def publish(channel: str, data: dict)`  
   - `async def consume(channel: str)` 返回异步迭代器  

4. **多源采集 Agent**  
   - Google Calendar API（`google-api-python-client`）  
   - IMAP（`asyncio-imaplib`）监听未读邮件  
   - 本地 Markdown 文件 `watchdog` 目录监控  

5. **优先级判断**  
   - 调用本地 Claude Code：`subprocess.run(["claude-cli", "--stdin"], ...)`  
   - 或 OpenAI、Ollama 本地模型  

6. **排期与提醒**  
   - iCal 写入（`icalendar` + AppleScript）  
   - macOS 通知：`pync`  

7. **测试**  
   - 先写测试：`tests/test_event_bus.py` 验证 publish/consume  
   - Agent 的边界条件和模拟数据  

8. **DevOps 流程**  
   - **pre-commit**：ruff、black、shellcheck、dotenv-linter  
   - **CI (GitHub Actions)**：macOS + Linux 测试、lint → 自动修复 → 报告  
   - **Release**：打 Tag → 自动发布 PyPI、Homebrew Tap  

9. **Changelog**  
   - 每次功能提交后，更新 `CHANGELOG.md`，遵循 [Keep a Changelog](https://keepachangelog.com/) 格式  

---

## 四、下一步行动

1. **确认架构与技术选型**  
2. **生成项目骨架**（目录+配置文件+初始 CLI 模块）  
3. **编写首批测试用例**（Event Bus、Ingestion Agent）  
4. **搭建 pre-commit & CI 环境**  

请确认以上方案或指出调整意见，随后我将立即为您自动生成项目骨架及初始代码，并附上详细注释、测试用例和 DevOps 配置。

