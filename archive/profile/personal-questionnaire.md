# 个人AI助理定制问卷

**填写指南**: 请根据您的真实工作习惯和偏好回答以下问题。这些信息将用于为您定制专属的Claude Code指令集。

---

## Part A: 核心工作模式

### Q1: 工作节奏偏好
您更倾向于哪种工作模式？
- [ ] a) 长时间深度专注于单一任务
- [ ] b) 在多个任务间灵活切换
- [ ] c) 混合模式，根据任务类型调整

**您的回答**: 
c,看我的状态，如果感兴趣会持续投入，遇到可并行的任务或者是一般类型会多任务并行完成。
---

### Q2: 决策风格
面对技术选择时，您的决策流程通常是？
- [ ] a) 快速评估，依靠直觉和经验
- [ ] b) 深度研究，全面分析后决定
- [ ] c) 寻求多方意见，集体决策

**您的回答**: 
目前多方意见是不太可能的，一般是看任务需求，一般来说公司项目讲究最快mvp，而我个人的研究则偏向难度和深度。
---

### Q3: 信息处理偏好 (1-5评分)
您希望AI回应的详细程度？
```
1 (极简) ←→ 5 (详细完整)
[ 1 ] [ 2 ] [ 3 ] [ 4 ] [ 5 ]
```

**您的评分**: 
这个首先极简，必要时可以详细。
---

## Part B: 技术生态

### Q4: 主要技术栈
您当前主要使用的编程语言和技术栈？(可多选)

**您的回答**: 
python 用的多，有一些情况似乎会需要补充js相关的知识。但总的来说我编程经验只有六个月。
---

### Q5: 开发工具链
您的日常开发环境包括哪些核心工具？

**您的回答**: 
vscode，git，github，claudecode
---

### Q6: 自动化现状
- 您当前已经自动化了哪些工作流程？
- 还希望自动化哪些重复性工作？

**您的回答**: 
没有，网络数据定期收集分析这块，菜谱生成，
---

## Part C: AI协作偏好

### Q7: 主动性期望 (1-5评分)
您希望AI主动提供建议的频率？
```
1 (完全被动) ←→ 5 (高度主动)  
[ 1 ] [ 2 ] [ 3 ] [ 4 ] [ 5 ]
```

**您的评分**: 
5
---

### Q8: 错误容忍度
当AI的建议存在不确定性时，您希望？
- [ ] a) AI明确标注风险等级和置信度
- [ ] b) AI保守一些，只提供确定的建议  
- [ ] c) AI大胆尝试，由我来承担风险评估

**您的回答**: 
a，c
---

### Q9: 学习方式偏好
您倾向于如何学习新技术？
- [ ] a) 通过实际项目边做边学
- [ ] b) 先系统学习理论再实践
- [ ] c) 找到类似案例进行模仿改进

**您的回答**: 
a，b但是大概懂一些理论
---

## Part D: 具体场景需求

### Q10: 高频协作场景
您最希望AI帮助处理的TOP 3工作场景是？

**您的回答**: 
1.网络多源感兴趣领域信息的日常投喂 
2. 程序项目协作
3. 科研辅助

---

### Q11: 主要痛点识别
目前在工作中最耗时或最困扰您的重复性任务是？

**您的回答**: 
任务过多，安排和处理效率还是有点不足，希望更高效和自动化。
---

### Q12: 价值期望
通过AI协作，您最希望获得哪种能力提升？

**您的回答**: 
超强的前沿嗅觉和商业机会识别及创业能力
---

## Part E: 认知负荷管理

### Q13: 复杂问题处理偏好
当面对复杂技术问题时，您更喜欢？
- [ ] a) 一次性获得完整解决方案
- [ ] b) 分步骤渐进式指导
- [ ] c) 多个备选方案供我选择

**您的回答**: 
b
---

### Q14: 情景判断测试 - 技术风险处理
**情景**: 您正在重构一个复杂的认证系统，AI建议使用一个新的、功能强大但相对不成熟的开源库。

AI的最佳回应方式应该是：
- [ ] a) 详细分析新库的优缺点，并提供传统方案作为对比
- [ ] b) 直接实现新库方案，相信技术优势
- [ ] c) 建议分阶段迁移，先在非关键部分试用
- [ ] d) 询问项目的风险承受度后再给建议

**您的回答**: 
d
---

### Q15: 风险承受度 (1-5评分)
在技术决策中的风险偏好？
```
1 (保守稳健) ←→ 5 (激进创新)
[ 1 ] [ 2 ] [ 3 ] [ 4 ] [ 5 ]
```

**您的评分**: 
3
---

## Part F: Claude Code特定偏好

### Q16: 工具编排偏好
在Claude Code环境中，您希望AI如何使用不同工具？
- [ ] a) 自动选择最适合的工具组合
- [ ] b) 解释工具选择理由，让我确认
- [ ] c) 让我指定使用哪些工具

**您的回答**: 
a，b
---

### Q17: 情景判断测试 - 工具选择策略
**情景**: 您遇到一个复杂的架构设计问题，Claude Code可以使用WebSearch获取最新资料，也可以使用zen__thinkdeep进行深度分析。

您更希望AI：
- [ ] a) 先WebSearch了解业界做法，再用thinkdeep深度分析
- [ ] b) 直接用thinkdeep基于已有知识深度思考
- [ ] c) 告诉我两种方案的区别，让我选择
- [ ] d) 同时使用两种工具，综合给出建议

**您的回答**: 
d
---

### Q18: 代码审查深度偏好
您希望AI进行代码审查时的关注重点？(可多选)
- [ ] a) 逻辑错误和bug
- [ ] b) 性能优化建议
- [ ] c) 代码风格和可读性
- [ ] d) 安全漏洞检查
- [ ] e) 架构和设计模式

**您的回答**: 
e，a，d
---

## Part G: 沟通风格与时间管理

### Q19: 沟通风格偏好
您喜欢的沟通风格是？
- [ ] a) 直接了当，简洁明确
- [ ] b) 详细解释，论证充分
- [ ] c) 友好交流，适度互动

**您的回答**: 
b
---

### Q20: 高效时段和优先级管理
- 您的高效工作时段通常是？
- 您习惯如何安排工作优先级？

**您的回答**: 
看当天的状态，工作优先级安排主要看公司ddl，个人的话，看兴趣和个人分析的项目价值和创造性如何。
---

### Q21: 开放式反馈
请描述一次AI助手最帮助您的经历，以及一次最让您失望的经历。两者的关键差别是什么？

**您的回答**: 
能不能识别问题，或者说能否给我比较有价值的建议。
---

## 提交说明

完成问卷后，请将此文件保存。这些信息将用于：
1. 生成您的个人AI协作配置文件
2. 定制专属的Claude Code指令集
3. 建立个性化的AI协作模式

您的偏好将被记录在本地，并可随时修改和更新。

---
*最后更新: 2025-06-25*