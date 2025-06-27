# GalaxyAI 个人画像系统使用指南

## 概述

这是一个基于Claude Code的个人AI助理定制系统，通过收集和分析您的工作偏好，生成完全个性化的AI协作指令。

## 系统架构

```
profile/
├── personal-questionnaire.md          # 21问个性化问卷
├── user-profile-schema.json          # 用户画像数据结构
├── user-profile-template.json        # 画像实例模板
├── behavior-feedback-system.md       # 行为观察和反馈机制
├── personalized-instruction-templates.md  # 指令模板系统
└── README.md                         # 使用指南

.claude/
└── instructions/
    └── personalized-core.md          # 生成的个性化指令
```

## 快速开始

### 步骤1: 填写个性化问卷
```bash
# 打开问卷文件
open profile/personal-questionnaire.md
```

按照问卷引导，填写您的工作偏好和协作习惯。

### 步骤2: 生成个人画像
基于问卷答案，创建您的个人画像文件：
```bash
# 复制模板
cp profile/user-profile-template.json profile/my-profile.json

# 根据问卷答案修改配置
```

### 步骤3: 生成个性化指令
```bash
# 基于画像生成指令（Python脚本待实现）
python scripts/generate_personalized_instruction.py --profile profile/my-profile.json --output .claude/instructions/core.md
```

### 步骤4: 激活个性化指令
```bash
# 在Claude Code中激活指令
claude config set instruction_file .claude/instructions/personalized-core.md
```

## 详细使用说明

### 1. 问卷系统
个性化问卷包含21个问题，分为6个维度：

**Part A: 核心工作模式** (Q1-Q3)
- 工作节奏偏好 (专注 vs 切换)
- 决策风格 (直觉 vs 分析)
- 信息详细程度偏好

**Part B: 技术生态** (Q4-Q6)
- 主要技术栈和工具
- 当前自动化水平
- 希望改进的工作流

**Part C: AI协作偏好** (Q7-Q9)
- AI主动性期望
- 错误和不确定性处理
- 学习和探索方式

**Part D: 具体场景需求** (Q10-Q12)
- 最希望AI帮助的场景
- 当前主要痛点
- 期望的能力提升

**Part E: 认知负荷管理** (Q13-Q15)
- 复杂问题处理偏好
- 风险承受度评估
- 情景判断测试

**Part F: Claude Code特定偏好** (Q16-Q21)
- 工具编排策略
- 代码审查重点
- 沟通风格和时间管理

### 2. 用户画像结构
用户画像包含三层数据：

**静态偏好** (`staticPreferences`)
- 从问卷收集的显式偏好
- 工作风格、沟通方式、技术偏好等

**行为指标** (`behavioralMetrics`)
- 从使用中观察到的真实模式
- 工具使用统计、交互模式、时间规律等

**反馈历史** (`feedbackHistory`)
- 用户主动提供的校正信息
- 满意度评分、改进建议、使用体验等

### 3. 个性化指令生成
系统会根据用户画像自动生成定制指令，包括：

**核心协作模式**
- 51/49权责分配的具体实现
- 决策流程和沟通标准

**行为参数配置**
- 详细程度 (1-5级)
- 主动性 (1-5级)
- 风险偏好 (1-5级)

**工具使用策略**
- 工具选择逻辑
- 工具组合模式
- 结果呈现方式

**场景化指令**
- 代码审查专用指令
- 技术分析专用指令
- 调试协助专用指令

## 高级功能

### 1. 动态优化
系统支持基于使用反馈的持续优化：

**实时调整**
```markdown
/feedback rating:4 comment:"回应太详细，希望更简洁"
/adjust verbosity 2  # 临时调整详细程度
```

**定期优化**
- 每周分析使用模式
- 检测偏好变化
- 自动更新指令配置

### 2. 多场景切换
```markdown
/load:core           # 加载通用协作指令
/load:code_review    # 切换到代码审查模式
/load:research       # 切换到技术调研模式  
/load:debug          # 切换到调试协助模式
```

### 3. 反馈收集
```markdown
## 快速反馈
/rate 5 "分析很有帮助"
/suggest "希望增加更多实例"

## 详细反馈
/feedback_form  # 打开详细反馈表单
```

## 隐私保护

### 数据存储策略
- **敏感数据**: 仅本地存储，不上传云端
- **匿名化**: 统计数据去除个人标识
- **加密保护**: 敏感配置文件加密存储
- **用户控制**: 完全的数据访问和删除权限

### 数据控制选项
```json
{
  "privacy_settings": {
    "collect_tool_usage": true,
    "collect_interaction_patterns": true,
    "collect_temporal_patterns": false,
    "store_feedback_history": true,
    "data_retention_days": 90
  }
}
```

## 故障排除

### 常见问题

**Q: 指令没有生效？**
A: 检查Claude Code配置：`claude config get instruction_file`

**Q: 个性化效果不明显？**
A: 确认问卷填写完整，并等待几次交互后提供反馈

**Q: 如何重置个性化设置？**
A: 删除 `profile/my-profile.json` 并重新填写问卷

**Q: 如何备份个人配置？**
A: 备份整个 `profile/` 目录

### 调试模式
```bash
# 启用详细日志
export GALAXYAI_DEBUG=true

# 查看配置状态
python scripts/check_profile_status.py
```

## 开发计划

### 已完成 ✅
- [x] 21问个性化问卷设计
- [x] 用户画像数据结构
- [x] 指令模板系统
- [x] 示例个性化指令

### 开发中 🚧
- [ ] Python指令生成脚本
- [ ] 行为数据收集工具
- [ ] 反馈分析系统

### 计划中 📋
- [ ] 可视化配置界面
- [ ] 智能建议引擎
- [ ] 跨设备同步功能

## 贡献和反馈

如果您在使用过程中发现问题或有改进建议：

1. 在项目仓库提交Issue
2. 使用 `/feedback` 指令提供使用反馈
3. 参与定期的用户体验调研

---

**记住**: 这个系统的目标是增强您的决策能力，而不是替代您的判断。始终保持51%的控制权，让AI成为您最得力的助手！

*最后更新: 2025-06-25*