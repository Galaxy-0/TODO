# GalaxyAI 个人助理系统使用指南

## 概述

GalaxyAI 是一个基于51/49协作模式的个人AI助理系统，旨在将您从"信息管理员"转变为"信息指挥官"。系统专注于高价值决策支持，而非简单的任务自动化。

### 核心理念
- **用户51%决策权**: 信息筛选、分析触发、最终决策、执行时机
- **AI 49%执行力**: 信息预处理、深度分析、方案草案生成、报告撰写

## 快速开始

### 1. 系统设置

```bash
# 1. 运行设置脚本
./setup.sh

# 2. 配置API密钥
# 编辑 config/config.env，填入您的OpenRouter API密钥
```

### 2. 基本工作流

```bash
# 捕获内容（三种方式）
./scripts/capture.sh           # 自动检测当前应用内容
./scripts/capture.sh -m        # 手动输入内容  
./scripts/capture.sh -s        # 截图模式

# 分析内容
./scripts/process.sh           # 交互式选择inbox中的内容
./scripts/process.sh "具体内容"  # 直接分析指定内容
./scripts/process.sh -q "内容"  # 快速分析模式

# 查看结果
open analysis/               # 查看所有分析报告
```

## 详细功能说明

### 内容捕获系统 (capture.sh)

#### 支持的应用程序
- **浏览器**: Safari, Chrome, Arc - 自动获取URL、标题、选中文本
- **文本编辑器**: 任意应用 - 获取选中文本和窗口标题
- **截图**: 可选OCR文字提取

#### 使用方法

```bash
# 基本捕获（检测当前活跃应用）
./scripts/capture.sh

# 手动输入模式
./scripts/capture.sh -m
# 然后输入内容，按Ctrl+D结束

# 截图模式
./scripts/capture.sh -s
# 系统会打开截图工具，选择区域后自动保存

# 查看帮助
./scripts/capture.sh --help
```

#### 全局快捷键设置

**推荐设置1: 使用macOS快捷键**
1. 打开「系统偏好设置」→「键盘」→「快捷键」→「服务」
2. 创建新的自动操作服务
3. 添加"运行Shell脚本"操作
4. 脚本内容: `/path/to/GalaxyAI/scripts/capture.sh`
5. 分配快捷键，如 `⌘⌥Space`

**推荐设置2: 使用Alfred/Raycast**
```bash
# Alfred Workflow
keyword: cap
script: /path/to/GalaxyAI/scripts/capture.sh

# Raycast Script Command  
#!/bin/bash
cd /path/to/GalaxyAI
./scripts/capture.sh
```

### 内容分析系统 (process.sh)

#### 分析模式

**深度分析模式（默认）**
- 生成完整的技术分析报告
- 包含价值评估、实施方案、风险分析
- 遵循CLAUDE.md中定义的分析格式
- 适用于重要决策和长期规划

**快速分析模式 (-q)**
- 简化版分析，重点关注核心价值
- 3-5个要点总结
- 适用于快速评估和初步筛选

#### 使用方法

```bash
# 交互式选择（推荐）
./scripts/process.sh
# 系统会显示inbox中所有待处理项目，选择编号即可

# 直接分析指定内容
./scripts/process.sh "git worktree工作流优化方案"

# 快速分析模式
./scripts/process.sh -q "新的AI编程工具"

# 从文件分析
./scripts/process.sh -f /path/to/content.txt

# 分析后不从inbox移除
./scripts/process.sh --no-remove "保留在inbox的内容"
```

#### 分析报告格式

生成的报告包含以下部分：
- **综合评级**: ⭐⭐⭐⭐⭐ (评分/5星)
- **核心问题识别**: 问题描述和机会分析
- **技术价值量化**: 直接效益和间接效益
- **深度技术解析**: 优势、限制、风险
- **集成方案设计**: 架构和工具建议
- **分阶段实施方案**: 具体的执行计划
- **最终建议**: 明确的行动建议

## 配置管理

### config/config.env 配置说明

```bash
# API配置
OPENROUTER_API_KEY=your-api-key           # 必需：OpenRouter API密钥
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet  # 可选：使用的AI模型

# 分析配置  
DEFAULT_ANALYSIS_MODE=deep                # deep | quick
MAX_CONTENT_LENGTH=5000                   # 内容长度限制
AUTO_REMOVE_FROM_INBOX=true              # 分析后自动清理

# 通知配置
ENABLE_NOTIFICATIONS=true                 # 启用macOS通知
NOTIFICATION_SOUND=true                   # 通知声音

# 截图配置
SCREENSHOT_FORMAT=png                     # png | jpg
ENABLE_OCR=false                         # 启用OCR（需要tesseract）
OCR_LANGUAGE=chi_sim+eng                 # OCR语言

# 成本控制
MONTHLY_API_BUDGET=50.00                 # 月度预算（美元）
COST_WARNING_THRESHOLD=0.80              # 预算警告阈值
```

### 推荐的AI模型选择

| 模型 | 速度 | 质量 | 成本 | 适用场景 |
|------|------|------|------|----------|
| `anthropic/claude-3-haiku` | 快 | 中 | 低 | 快速分析、日常任务 |
| `anthropic/claude-3.5-sonnet` | 中 | 高 | 中 | 平衡选择（推荐） |
| `anthropic/claude-3-opus` | 慢 | 最高 | 高 | 重要决策、复杂分析 |
| `openai/gpt-4` | 中 | 高 | 中 | 替代选择 |

## 最佳实践

### 日常工作流建议

**1. 信息捕获策略**
- 设置全局快捷键，养成随时捕获有价值信息的习惯
- 不要过度捕获，重质量不重数量
- 定期清理inbox，保持系统整洁

**2. 分析请求策略** 
- 优先使用快速分析模式进行初步筛选
- 只对真正重要的内容使用深度分析
- 批量处理相似内容，提高效率

**3. 成本控制策略**
- 设置合理的月度预算
- 多使用快速分析模式
- 利用缓存，避免重复分析相同内容

### 与现有工具集成

**git worktree工作流集成**
```bash
# 1. 使用GalaxyAI分析新功能想法
./scripts/process.sh "实现用户认证系统"

# 2. 查看生成的分析报告
open analysis/user-auth-analysis-2025-06-25.md

# 3. 基于分析结果创建工作树
git worktree add ../myproject-feature-auth feature/user-auth

# 4. 在新工作树中实施方案
cd ../myproject-feature-auth
# 开始编码...
```

**Claude Code工作流集成**
- 在Claude Code中打开GalaxyAI生成的分析报告
- 基于报告中的实施建议进行编码
- 将代码实现过程中的新想法捕获到GalaxyAI

### 隐私保护实践

**敏感信息处理**
- 微信/QQ等私人通讯内容谨慎捕获
- 工作相关敏感信息使用本地处理
- 定期检查和清理已保存的分析内容

**数据安全**
- API密钥安全存储，不要提交到版本控制
- 分析报告中避免包含密码、密钥等敏感信息
- 考虑使用本地AI模型处理敏感内容

## 故障排除

### 常见问题

**Q: capture.sh 无法获取浏览器内容**
A: 检查系统偏好设置中的「辅助功能」权限，确保已授权给Terminal或相关应用

**Q: process.sh 报告API密钥错误**  
A: 确认config/config.env中已正确设置OPENROUTER_API_KEY

**Q: 截图功能不工作**
A: 检查「屏幕录制」权限，确保已授权给Terminal

**Q: OCR功能不可用**
A: 安装tesseract: `brew install tesseract tesseract-lang`

### 调试模式

```bash
# 启用调试模式
echo "DEBUG_MODE=true" >> config/config.env

# 查看详细日志
./scripts/capture.sh   # 会显示详细的执行过程
./scripts/process.sh   # 会显示API调用详情
```

### 系统要求检查

```bash
# 运行系统健康检查
./setup.sh             # 重新运行设置脚本

# 手动检查依赖
which jq               # JSON处理工具
which osascript       # AppleScript支持
which tesseract        # OCR支持（可选）
```

## 高级用法

### 自定义分析提示词

编辑 `scripts/process.sh` 中的 `build_analysis_prompt()` 函数，可以自定义分析报告的格式和重点关注的方面。

### 批量处理

```bash
# 批量分析多个文件
for file in /path/to/files/*.txt; do
    ./scripts/process.sh -f "$file"
done

# 批量清理inbox
./scripts/process.sh -q $(cat inbox.md | grep "^\-" | head -5)
```

### API使用监控

```bash
# 查看API使用情况（需要自行实现日志记录）
tail -f logs/api_usage.log

# 估算月度成本
grep "API_COST" logs/api_usage.log | awk '{sum += $2} END {print sum}'
```

## 系统演进计划

### Phase 2 规划功能
- 本地AI集成（Ollama + Llama 3.1）
- 智能代办管理系统
- 个人偏好学习和优化
- 移动端同步支持

### Phase 3 长期目标
- Claude Code深度集成
- 自动化工作流编排
- 团队协作功能
- 高级分析和洞察

---

## 支持和反馈

如果您在使用过程中遇到问题或有改进建议，请：

1. 检查本指南的故障排除部分
2. 查看 `scripts/` 目录中的脚本注释
3. 在项目仓库中提交Issue

**记住**: GalaxyAI的目标是增强您的决策能力，而不是替代您的判断。保持51%的控制权，让AI成为您最得力的助手！