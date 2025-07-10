# 2019 Intel MacBook Pro 终端工作流使用手册

> **适用范围**：iTerm2 + Oh My Zsh 环境，包含 ripgrep、eza、fd、bat、fzf、tldr、git、GitHub CLI (gh)、starship 等现代 CLI 工具。

---

## 目录

1. 环境概览
2. iTerm2 使用指南
3. Oh My Zsh 配置
4. 现代 CLI 工具详解
5. 常用 alias / 函数
6. starship Prompt 个性化
7. 版本更新与维护
8. 故障排查

---

## 1 环境概览

| 组件              | 版本                                            | 关键路径                      |
| --------------- | --------------------------------------------- | ------------------------- |
| **iTerm2**      | 3.x                                           | `/Applications/iTerm.app` |
| **zsh**         | macOS 默认（2.8+）                                | `/bin/zsh`                |
| **Oh My Zsh**   | 最新 master                                     | `~/.oh-my-zsh`            |
| **Homebrew 前缀** | `/usr/local`                                  | `brew --prefix`           |
| **主插件**         | zsh‑autosuggestions / zsh‑syntax‑highlighting | `~/.zshrc` 中 `source ...` |

---

## 2 iTerm2 使用指南

### 2.1 核心快捷键

| 操作         | 默认快捷键         | 说明                |
| ---------- | ------------- | ----------------- |
| 新建标签       | `⌘ + T`       | 每个标签对应一个 shell 会话 |
| 垂直分屏       | `⌘ + D`       | 当前 pane 左右拆分      |
| 水平分屏       | `⌘ + ⇧ + D`   | 当前 pane 上下拆分      |
| 在 pane 间切换 | `⌘ + ⌥ + 方向键` | 光标移动到相邻 pane      |
| 查找输出       | `⌘ + F`       | 实时高亮匹配            |
| 快速粘贴历史     | `⌘ + ⇧ + H`   | 显示剪贴板历史弹窗         |

### 2.2 外观与配色

1. **主题** → *Preferences › Profiles › Colors* 选择 `One Dark` 或导入 `Solarized Dark.itermcolors`。
2. **字体** → 推荐 `JetBrains Mono NL` 13 pt，宽字符友好。
3. **透明度** → Profiles › Window › *Window Appearance* 设置 10–15 %。

### 2.3 分屏布局 Tips

```text
⌘D  垂直分屏       ⌘⇧D 水平分屏
⌘⌥→/←/↑/↓  Pane 跳转
⌥⌘W  关闭当前 Pane/Tab
```

将长时间运行的进程放在单独 pane，避免滚动干扰。

### 2.4 触控栏 & 触发器

- *Preferences › Keys › Touch Bar*：添加常用 `git pull`、`npm run dev` 按钮。
- 触发器（Triggers）：监听关键字如 `ERROR`, 自动弹窗通知。

---

## 3 Oh My Zsh 配置

### 3.1 安装与更新

```bash
# 安装
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
# 更新
omz update
```

### 3.2 推荐插件加载顺序 `.zshrc`

```zsh
# --- Homebrew 路径 ---
eval "$(/usr/local/bin/brew shellenv)"

# --- Oh My Zsh Core ---
ZSH_THEME="robbyrussell"  # 或空，由 starship 控制
plugins=(git z extract)

source $ZSH/oh-my-zsh.sh

# --- 第三方插件 ---
source /usr/local/share/zsh-autosuggestions/zsh-autosuggestions.zsh
source /usr/local/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh

# --- fzf 键位 ---
[ -f "$(brew --prefix)/opt/fzf/shell/key-bindings.zsh" ] && \
  source "$(brew --prefix)/opt/fzf/shell/key-bindings.zsh"
```

### 3.3 常用 Zsh 键盘操作

| 功能        | 快捷键                         |
| --------- | --------------------------- |
| 历史搜索（fzf） | `Ctrl + R`                  |
| 正向/反向搜索   | `Ctrl + S / Ctrl + R` (若启用) |
| 光标按单词跳转   | `Esc + B / F`               |

---

## 4 现代 CLI 工具详解

### 4.1 ripgrep (`rg`)

- **核心命令**：`rg "pattern" path/` —— 递归搜索，默认忽略 .gitignore。
- **常用参数**：`-n` 显示行号，`-tpy` 只搜 Python，`-S` 不区分大小写。
- **示例**：
  ```bash
  rg -n "TODO" --glob "!tests/**"
  ```

### 4.2 eza（exa 替代）

- **基本用法**：`eza -la --git` 彩色列表 + git 状态。
- **树形视图**：`eza --tree --level=2`。
- **别名**：`alias ls='eza -la --git'`。

### 4.3 fd

- 现代化 `find`，语法更简单，支持正则。
- **示例**：`fd -e py "model" src/`。

### 4.4 bat

- 语法高亮的 `cat`。
- **查看差异**：`bat --diff file1.py file2.py`。

### 4.5 fzf

- **历史命令搜索**：`Ctrl + R` 打开模糊搜索界面。
- **文件搜索**：`fzf`（配合 `rg --files | fzf`）。
- **多选**：`fzf -m` 支持空格多选。

### 4.6 tldr

- 快速示例文档：`tldr tar`。

### 4.7 git & GitHub CLI (`gh`)

| 任务      | 命令                                    |
| ------- | ------------------------------------- |
| 创建 PR   | `gh pr create -t "feat:..." -b "..."` |
| 查看 PR   | `gh pr view --web`                    |
| 克隆 repo | `gh repo clone owner/name`            |

### 4.8 starship

- **安装**：`brew install starship`，在 `.zshrc` 末尾加：
  ```zsh
  eval "$(starship init zsh)"
  ```
- **配置**：创建 `~/.config/starship.toml`：
  ```toml
  add_newline = false
  [git_branch]
  format = "[ $branch](bold purple) "
  [python]
  format = "[🐍 $virtualenv](yellow) "
  ```

---

## 5 常用 alias / 函数

```zsh
# 快捷导航
alias ..='cd ..'
alias ...='cd ../..'
# 快速搜索并打开文件
ff () { rg --files | fzf | xargs -r $EDITOR; }
# Git one‑liner
alias gs='git status -sb'
```

---

## 6 starship Prompt 个性化

1. 主题文件 `~/.config/starship.toml` 支持 Lua 风格模板。
2. 可根据电池、电源状态显示图标：
   ```toml
   [battery]
   full_symbol = "🔋"
   charging_symbol = "⚡"
   discharging_symbol = "🔌"
   ```
3. 兼容 iTerm2 自定义色彩方案，无额外插件。

---

## 7 版本更新与维护

| 任务           | 命令                                                                     | 频率    |
| ------------ | ---------------------------------------------------------------------- | ----- |
| 更新所有软件       | `brew update && brew upgrade`                                          | 每周    |
| 清理旧版本        | `brew cleanup`                                                         | 每月    |
| 检查系统状态       | `brew doctor`                                                          | 出现异常时 |
| 更新 Oh My Zsh | `omz update`                                                           | 月度    |
| iTerm2 自动更新  | *Preferences › General › Updates* 勾选 *Automatically check for updates* | 周期性   |

---

## 8 故障排查

| 症状                  | 快速定位                                        | 解决方案                                    |
| ------------------- | ------------------------------------------- | --------------------------------------- |
| fzf 快捷键无响应          | `echo $FZF_DEFAULT_OPTS` 是否为空               | 重新运行 `$(brew --prefix)/opt/fzf/install` |
| zsh 插件报 “not found” | 查看 `.zshrc` 路径是否正确                          | 更新插件路径或 `brew reinstall <plugin>`       |
| Homebrew 更新超慢       | 查看 `brew config` 中 `HOMEBREW_BOTTLE_DOMAIN` | 临时切换清华/中科大镜像                            |
| iTerm2 输出乱码         | 确认字体支持 Powerline                            | 安装 `JetBrains Mono NL`、`Hack Nerd Font` |

---

### 结束语

完成本手册全部步骤后，你将拥有：

- ✨ 现代化、极简且功能齐备的终端外观；
- ⚡ 秒级全文搜索、模糊查找、自动补全；
- 🐙 Git/GitHub 全流程无需离开命令行；
- 🔋 精简高效的 prompt 与资源占用； 确保你的 2019 Intel MacBook Pro 在 2025 年依旧“战斗力 MAX”。

