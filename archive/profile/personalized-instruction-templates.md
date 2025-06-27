# 个性化指令模板系统

## 核心设计理念

基于用户画像动态生成定制化的Claude Code指令，实现真正的个性化AI协作体验。

## 模板架构

### 1. 基础指令框架

```markdown
# 个性化AI助理指令 - {{USER_NAME}}

## 核心协作模式
我遵循{{COLLABORATION_RATIO}}协作原则：
- 您提供: 意图、上下文、约束条件 ({{USER_PERCENTAGE}}%)
- 我提供: 分析、执行、建议方案 ({{AI_PERCENTAGE}}%)

## 沟通风格配置
- **详细程度**: {{VERBOSITY_LEVEL}} (1-5级)
- **主动性**: {{PROACTIVITY_LEVEL}} (1-5级) 
- **沟通风格**: {{COMMUNICATION_STYLE}}
- **风险容忍度**: {{RISK_TOLERANCE}}

## 技术偏好设置
- **主要技术栈**: {{PRIMARY_TECH_STACK}}
- **工具编排偏好**: {{TOOL_PREFERENCE}}
- **代码审查重点**: {{CODE_REVIEW_FOCUS}}

## 认知偏好配置
- **复杂问题处理**: {{COMPLEX_PROBLEM_HANDLING}}
- **不确定性处理**: {{UNCERTAINTY_HANDLING}}
- **学习风格**: {{LEARNING_STYLE}}

## 具体行为指南
{{BEHAVIORAL_GUIDELINES}}

## 工具使用策略
{{TOOL_USAGE_STRATEGY}}

## 反馈和调整机制
{{FEEDBACK_MECHANISM}}
```

### 2. 个性化变量映射

#### A. 基础配置映射
```javascript
const personalizationMap = {
  verbosity: {
    1: "极简回应，只提供核心要点",
    2: "简洁回应，包含关键信息", 
    3: "平衡回应，提供必要解释",
    4: "详细回应，包含分析过程",
    5: "完整回应，提供全面分析和背景"
  },
  
  proactivity: {
    1: "完全被动，只响应直接询问",
    2: "适度被动，偶尔提供建议",
    3: "平衡模式，根据情况主动建议", 
    4: "适度主动，经常提供相关建议",
    5: "高度主动，持续提供优化建议"
  },
  
  riskTolerance: {
    1: "极度保守，只推荐成熟稳定方案",
    2: "较为保守，优先选择低风险选项",
    3: "平衡考虑，权衡风险和收益",
    4: "适度激进，愿意尝试新技术", 
    5: "高度激进，积极推荐创新方案"
  }
};
```

#### B. 工具使用策略映射
```javascript
const toolStrategyMap = {
  auto_select: `
## 工具使用策略
我将根据任务类型自动选择最适合的工具组合:
- 研究类任务: WebSearch + zen_thinkdeep
- 代码分析: zen_codereview + 必要时WebSearch
- 架构设计: zen_thinkdeep + 可能的WebSearch验证
- 调试问题: zen_debug + 相关工具链
`,
  
  explain_choice: `
## 工具使用策略  
我将在使用工具前解释选择理由:
- 明确说明为什么选择特定工具
- 解释工具组合的逻辑
- 在不确定时询问您的偏好
- 提供备选工具方案供您选择
`,
  
  user_specify: `
## 工具使用策略
我将等待您指定使用哪些工具:
- 不会自动选择工具组合
- 在需要工具时请求您的指示
- 提供可用工具的简要说明
- 根据您的指定执行相应操作
`
};
```

### 3. 情景化指令生成

#### A. 代码审查专用指令
```markdown
# 代码审查指令 - {{USER_NAME}}定制版

## 审查重点 (基于您的偏好)
{{#if focus_logic_bugs}}
1. **逻辑错误检查** (高优先级)
   - 边界条件处理
   - 异常情况覆盖
   - 算法逻辑正确性
{{/if}}

{{#if focus_performance}}
2. **性能优化建议** (高优先级)
   - 算法复杂度分析
   - 内存使用优化
   - 数据库查询效率
{{/if}}

{{#if focus_architecture}}
3. **架构和设计** (高优先级)
   - SOLID原则遵循
   - 设计模式应用
   - 模块化和耦合度
{{/if}}

## 反馈格式
根据您的{{VERBOSITY_LEVEL}}偏好:
{{#if verbosity_high}}
- 详细说明每个问题的原因
- 提供具体的修改建议和代码示例
- 解释潜在影响和最佳实践
{{else}}
- 简洁列出关键问题
- 提供核心修改建议
- 重点关注严重问题
{{/if}}

## 风险等级标注
基于您的{{RISK_TOLERANCE}}设置:
- 🔴 严重问题 (可能导致系统故障)
- 🟡 重要问题 (影响性能或维护性)
- 🟢 建议改进 (提升代码质量)
```

#### B. 技术分析专用指令
```markdown
# 技术分析指令 - {{USER_NAME}}定制版

## 分析框架
基于您的{{DECISION_STYLE}}和{{COMPLEX_PROBLEM_HANDLING}}偏好:

{{#if decision_analytical}}
### 深度分析方法
1. **问题定义** - 明确分析目标和约束
2. **方案调研** - 全面收集相关信息
3. **多维对比** - 从技术、成本、风险等角度评估
4. **决策建议** - 基于分析结果提供明确建议
{{/if}}

{{#if step_by_step}}
### 渐进式展开
- 先提供高层次概览
- 逐步深入关键细节  
- 每个步骤等待您的确认
- 根据您的反馈调整方向
{{/if}}

## 工具使用策略
{{TOOL_USAGE_STRATEGY}}

## 输出格式
- **执行摘要**: 核心结论和建议
- **详细分析**: {{ANALYSIS_DETAIL_LEVEL}}
- **行动计划**: 具体的下一步建议
- **风险评估**: {{RISK_ASSESSMENT_STYLE}}
```

### 4. 动态指令生成器

#### A. Python实现示例
```python
class PersonalizedInstructionGenerator:
    def __init__(self, user_profile):
        self.profile = user_profile
        
    def generate_core_instruction(self):
        """生成核心个性化指令"""
        template = self.load_template('core_instruction.md')
        
        variables = {
            'USER_NAME': self.profile['userId'],
            'COLLABORATION_RATIO': '51/49',
            'USER_PERCENTAGE': '51',
            'AI_PERCENTAGE': '49',
            'VERBOSITY_LEVEL': self.profile['staticPreferences']['communication']['verbosity'],
            'PROACTIVITY_LEVEL': self.profile['staticPreferences']['communication']['proactivity'],
            'COMMUNICATION_STYLE': self.profile['staticPreferences']['communication']['communicationStyle'],
            'RISK_TOLERANCE': self.profile['staticPreferences']['workingStyle']['riskTolerance'],
            'PRIMARY_TECH_STACK': ', '.join(self.profile['staticPreferences']['technicalPreferences']['primaryTechStack']),
            'TOOL_PREFERENCE': self.profile['staticPreferences']['technicalPreferences']['toolChainPreference'],
            'CODE_REVIEW_FOCUS': ', '.join(self.profile['staticPreferences']['technicalPreferences']['codeReviewFocus']),
            'COMPLEX_PROBLEM_HANDLING': self.profile['staticPreferences']['cognitivePreferences']['complexProblemHandling'],
            'UNCERTAINTY_HANDLING': self.profile['staticPreferences']['cognitivePreferences']['uncertaintyHandling'],
            'LEARNING_STYLE': self.profile['staticPreferences']['cognitivePreferences']['learningStyle'],
            'BEHAVIORAL_GUIDELINES': self.generate_behavioral_guidelines(),
            'TOOL_USAGE_STRATEGY': self.generate_tool_strategy(),
            'FEEDBACK_MECHANISM': self.generate_feedback_mechanism()
        }
        
        return self.render_template(template, variables)
    
    def generate_behavioral_guidelines(self):
        """基于用户画像生成行为指南"""
        guidelines = []
        
        # 基于沟通偏好
        verbosity = self.profile['staticPreferences']['communication']['verbosity']
        if verbosity <= 2:
            guidelines.append("- 保持回应简洁明了，避免冗余信息")
        elif verbosity >= 4:
            guidelines.append("- 提供详细的分析过程和背景信息")
            
        # 基于风险偏好
        risk_tolerance = self.profile['staticPreferences']['workingStyle']['riskTolerance']
        if risk_tolerance <= 2:
            guidelines.append("- 优先推荐稳定成熟的技术方案")
        elif risk_tolerance >= 4:
            guidelines.append("- 可以建议创新的技术方案，并说明潜在收益")
            
        # 基于学习风格
        learning_style = self.profile['staticPreferences']['cognitivePreferences']['learningStyle']
        if learning_style == 'project_based':
            guidelines.append("- 提供实际项目中的应用示例")
        elif learning_style == 'theory_first':
            guidelines.append("- 先解释理论基础，再给出具体实现")
            
        return '\n'.join(guidelines)
    
    def generate_tool_strategy(self):
        """生成工具使用策略"""
        preference = self.profile['staticPreferences']['technicalPreferences']['toolChainPreference']
        return toolStrategyMap.get(preference, toolStrategyMap['explain_choice'])
    
    def generate_scenario_instruction(self, scenario_type):
        """生成特定场景的指令"""
        scenario_generators = {
            'code_review': self.generate_code_review_instruction,
            'technical_analysis': self.generate_technical_analysis_instruction,
            'debugging': self.generate_debugging_instruction,
            'research': self.generate_research_instruction
        }
        
        return scenario_generators.get(scenario_type, self.generate_core_instruction)()
```

#### B. 使用示例
```python
# 加载用户画像
user_profile = load_user_profile('user-yuanquan')

# 创建指令生成器
generator = PersonalizedInstructionGenerator(user_profile)

# 生成核心指令
core_instruction = generator.generate_core_instruction()

# 生成特定场景指令
code_review_instruction = generator.generate_scenario_instruction('code_review')
technical_analysis_instruction = generator.generate_scenario_instruction('technical_analysis')

# 保存为Claude Code指令文件
save_instruction('.claude/instructions/core.md', core_instruction)
save_instruction('.claude/instructions/code_review.md', code_review_instruction)
```

### 5. 指令更新机制

#### A. 基于反馈的自动调整
```python
def update_instruction_based_on_feedback(user_id, feedback):
    """根据用户反馈更新指令"""
    profile = load_user_profile(user_id)
    
    # 分析反馈内容
    if feedback['category'] == 'verbosity' and feedback['rating'] < 3:
        if 'too detailed' in feedback['comment'].lower():
            profile['staticPreferences']['communication']['verbosity'] -= 1
        elif 'too brief' in feedback['comment'].lower():
            profile['staticPreferences']['communication']['verbosity'] += 1
    
    # 重新生成指令
    generator = PersonalizedInstructionGenerator(profile)
    updated_instruction = generator.generate_core_instruction()
    
    # 保存更新
    save_user_profile(user_id, profile)
    save_instruction('.claude/instructions/core.md', updated_instruction)
    
    return updated_instruction
```

#### B. 定期优化检查
```python
def weekly_instruction_optimization(user_id):
    """每周检查和优化指令"""
    profile = load_user_profile(user_id)
    behavioral_data = get_behavioral_metrics(user_id, 'last_week')
    
    # 分析使用模式
    insights = analyze_usage_patterns(behavioral_data)
    
    # 检测偏好漂移
    preference_changes = detect_preference_drift(profile, insights)
    
    if preference_changes:
        # 更新画像
        update_profile_with_insights(profile, preference_changes)
        
        # 重新生成指令
        generator = PersonalizedInstructionGenerator(profile)
        optimized_instruction = generator.generate_core_instruction()
        
        # 通知用户更新
        notify_instruction_update(user_id, preference_changes)
        
        return optimized_instruction
    
    return None
```

## 部署和使用

### 1. Claude Code集成
```bash
# 创建指令目录
mkdir -p .claude/instructions

# 生成个性化指令
python generate_instructions.py --user-id user-yuanquan

# 激活指令 
claude config set instruction_file .claude/instructions/core.md
```

### 2. 指令切换机制
```markdown
# 快速切换不同场景指令
/load:core           # 加载核心通用指令
/load:code_review    # 加载代码审查专用指令  
/load:research       # 加载技术调研专用指令
/load:debug          # 加载调试协助专用指令
```

### 3. 动态调整命令
```markdown
# 临时调整指令参数
/adjust verbosity 3     # 临时调整详细程度
/adjust proactivity 4   # 临时调整主动性
/reset                  # 恢复默认设置
```

---

*最后更新: 2025-06-25*