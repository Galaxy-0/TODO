# GalaxyAI 产品需求文档（PRD）

## 1. 产品概述

### 1.1 产品定位
GalaxyAI 是一个深度集成 Claude Code SDK 的智能开发助手系统，通过统一的命令行界面（CLI）为开发者提供从信息捕获、任务管理到代码生成、商业洞察的全流程 AI 辅助能力。

### 1.2 核心价值主张
- **统一入口**：通过单一 CLI 命令访问所有 Claude Code 工具能力
- **智能编排**：自动选择和组合最适合的工具来完成复杂任务
- **上下文感知**：持续学习和适应用户的工作模式和项目特征
- **渐进式复杂度**：从简单命令到复杂工作流的平滑过渡

### 1.3 目标用户
- 主要用户：具有 6 个月以上编程经验的开发者
- 次要用户：希望提升效率的技术创业者
- 潜在用户：需要 AI 辅助的研究人员和学生

## 2. 功能架构

### 2.1 核心模块

#### 2.1.1 智能调度层（Intelligent Dispatcher）
**功能描述**：
作为系统的"大脑"，负责理解用户意图并调度合适的工具组合。

**关键特性**：
- 意图识别：通过关键词和上下文理解用户需求
- 策略选择：根据任务复杂度选择执行策略（反应式/计划式/交互式）
- 工具编排：智能组合多个工具完成复杂任务
- 成本优化：在多个可选方案中选择最优路径

**工具映射规则**：
- 调试类任务 → zen__debug + 文件操作工具
- 重构类任务 → zen__refactor + zen__codereview
- 实现类任务 → zen__planner + zen__testgen
- 分析类任务 → zen__analyze + WebSearch

#### 2.1.2 上下文引擎（Context Engine）
**功能描述**：
维护和管理项目、用户、任务的完整上下文信息。

**数据来源**：
- 用户画像：基于 21 问问卷的静态偏好
- 项目配置：CLAUDE.md 和项目特定设置
- 任务状态：通过 TodoWrite/TodoRead 管理的任务列表
- IDE 状态：通过 mcp__ide 工具获取的实时诊断信息
- Git 状态：当前分支、未提交更改、工作树信息

**上下文聚合流程**：
1. 读取全局用户配置（~/.config/galaxyai/profile.toml）
2. 扫描项目配置（./.galaxyai/project.toml）
3. 解析当前 Git 状态和分支语义
4. 获取活跃任务列表
5. 收集 IDE 诊断信息（如可用）
6. 生成综合上下文对象

#### 2.1.3 状态管理系统（State Management）
**功能描述**：
使用 Claude Code 的 TodoWrite/TodoRead 工具进行持久化状态管理。

**状态结构**：
- 主任务：用户的高层次意图
- 子任务：分解后的具体执行步骤
- 任务元数据：所需工具、预计时间、依赖关系
- 执行日志：每个步骤的输入输出和结果

**状态流转**：
待处理（pending）→ 进行中（in_progress）→ 已完成（completed）/失败（failed）

#### 2.1.4 错误恢复机制（Error Recovery）
**功能描述**：
当工具执行失败时，自动尝试恢复或提供修复建议。

**恢复策略**：
- 文件操作错误：自动搜索正确路径、检查权限
- 工具超时：切换到备用工具或简化任务
- API 限流：实施退避策略、使用本地缓存
- 执行失败：调用 zen__debug 分析原因并提供解决方案

### 2.2 工具集成策略

#### 2.2.1 基础工具层
**文件操作工具**：
- Read/Write/Edit：所有文件相关操作的基础
- Glob/Grep：快速定位和搜索内容
- 并发策略：支持批量读取，提升效率

**执行工具**：
- Bash：系统命令执行，需要严格的安全控制
- 超时控制：默认 30 秒，可配置
- 输出限制：防止大量输出阻塞系统

#### 2.2.2 认知工具层（Zen Tools）
**深度分析工具**：
- zen__thinkdeep：复杂问题的多角度分析
- 使用场景：架构设计、技术选型、问题探索
- 成本考虑：高成本工具，需要明确的使用场景

**代码质量工具**：
- zen__codereview：自动代码审查
- zen__refactor：智能重构建议
- zen__testgen：测试用例生成
- 协同使用：review → refactor → testgen 的完整质量提升链

**规划执行工具**：
- zen__planner：任务分解和执行计划
- zen__precommit：提交前的全面检查
- 集成点：与 Git hooks 深度集成

#### 2.2.3 IDE 集成层
**诊断集成**：
- mcp__ide__getDiagnostics：获取编辑器错误和警告
- 降级策略：IDE 不可用时回退到文件分析

**代码执行**：
- mcp__ide__executeCode：在 Jupyter 环境执行代码
- 安全限制：仅在明确的沙箱环境中使用

### 2.3 工作流设计

#### 2.3.1 信息处理工作流
**Capture（捕获）→ Process（处理）→ Archive（归档）**

1. **智能捕获**：
   - 来源：剪贴板、浏览器、终端输出、IDE 选中
   - 自动分类：代码片段、链接、文本、图片
   - 元数据标记：时间戳、来源、项目关联

2. **深度处理**：
   - 快速模式：分类归档，添加标签
   - 深度模式：调用 zen__analyze 进行内容分析
   - 关联分析：与现有知识库建立连接

3. **智能归档**：
   - 项目相关：存入项目文档
   - 知识积累：更新个人知识库
   - 任务生成：重要内容自动创建 TODO

#### 2.3.2 实验驱动工作流
**Hypothesis（假设）→ Experiment（实验）→ Validate（验证）→ Learn（学习）**

1. **实验初始化**：
   - 创建独立的 Git worktree
   - 生成实验清单（experiment.toml）
   - 配置专属的 CLAUDE.md 指令

2. **实验执行**：
   - 自动追踪所有更改
   - 定期生成进度报告
   - 智能提示偏离假设的操作

3. **结果验证**：
   - 对比假设与实际结果
   - 量化成功指标
   - 生成实验总结报告

4. **经验沉淀**：
   - 成功经验：合并到主分支
   - 失败教训：归档到知识库
   - 模式识别：更新个人最佳实践

#### 2.3.3 每日复盘工作流
**Collect（收集）→ Analyze（分析）→ Insight（洞察）→ Plan（计划）**

1. **数据收集**：
   - Git 提交历史
   - 完成的任务列表
   - 使用的工具统计
   - 遇到的问题记录

2. **智能分析**：
   - 效率评估：实际 vs 预期时间
   - 模式识别：高产时段、常见阻碍
   - 技能评估：使用的技术栈分析

3. **洞察生成**：
   - 调用 zen__thinkdeep 生成深度洞察
   - 识别改进机会
   - 发现潜在的技术债务

4. **明日计划**：
   - 基于优先级排序任务
   - 预估时间和资源需求
   - 设置具体可衡量的目标

### 2.4 个性化系统

#### 2.4.1 用户画像管理
**初始化流程**：
1. 21 问个性化问卷
2. 生成用户画像配置文件
3. 创建个性化 CLAUDE.md 指令
4. 配置工具使用偏好

**画像维度**：
- 技术背景：编程经验、主要语言、技术栈
- 工作风格：专注模式、多任务偏好、时间管理
- 沟通偏好：详细程度（1-5）、主动性（1-5）
- 目标导向：短期目标、长期愿景、学习计划

#### 2.4.2 动态适应机制
**行为学习**：
- 工具使用频率统计
- 任务完成时间分析
- 错误模式识别
- 偏好变化检测

**自动优化**：
- 调整默认工具选择
- 优化提示词生成
- 个性化错误恢复策略
- 定制化输出格式

### 2.5 安全与隐私

#### 2.5.1 数据安全
- 本地优先：所有个人数据仅存储在本地
- 加密存储：敏感配置使用加密保护
- 版本控制：配置文件支持 Git 管理
- 备份机制：自动定期备份关键数据

#### 2.5.2 执行安全
- 命令审核：危险命令需要二次确认
- 沙箱执行：高风险操作在隔离环境
- 权限控制：最小权限原则
- 审计日志：所有操作可追溯

## 3. 技术实现方案

### 3.1 技术栈选择
- **CLI 框架**：Python + Typer（类型安全、自动文档）
- **异步处理**：asyncio（支持并发工具调用）
- **配置管理**：TOML（人类可读、易于编辑）
- **状态存储**：JSON + 文件系统（简单可靠）
- **进程通信**：标准输入输出（与 Claude Code 交互）

### 3.2 部署架构
- **核心进程**：galaxy-ai CLI（无守护进程）
- **配置位置**：~/.config/galaxyai/（全局）+ .galaxyai/（项目）
- **日志位置**：~/.local/share/galaxyai/logs/
- **缓存位置**：~/.cache/galaxyai/

### 3.3 性能指标
- CLI 启动时间：< 100ms
- 命令响应延迟：< 500ms（不含 API 调用）
- 内存占用：< 50MB（空闲状态）
- 并发工具调用：最多 5 个并行

## 4. 实施计划

### 4.1 第一阶段（MVP - 第1周）
- 基础 CLI 框架搭建
- 用户配置系统实现
- 简单工具调用（Read/Write/Bash）
- 基础错误处理

### 4.2 第二阶段（核心功能 - 第2周）
- 智能调度层实现
- Zen 工具集成
- TODO 状态管理
- CLAUDE.md 动态生成

### 4.3 第三阶段（工作流 - 第3周）
- Capture-Process 工作流
- 实验管理功能
- Git 集成优化
- 并发执行优化

### 4.4 第四阶段（智能化 - 第4周）
- 个性化系统完善
- 自动复盘功能
- 高级错误恢复
- 性能优化

## 5. 成功指标

### 5.1 用户体验指标
- 学习曲线：新用户 30 分钟内完成首个任务
- 任务完成率：> 90% 的任务一次成功
- 用户满意度：NPS 分数 > 50

### 5.2 技术指标
- 系统可用性：> 99.9%
- 平均响应时间：< 2 秒
- 错误恢复率：> 80%

### 5.3 业务价值指标
- 开发效率提升：> 30%
- 代码质量改善：bug 减少 > 20%
- 知识沉淀率：> 70% 的实验有文档产出

## 6. 风险与缓解

### 6.1 技术风险
- **Claude API 依赖**：实现本地缓存和降级方案
- **工具版本兼容**：严格的版本锁定和测试
- **性能瓶颈**：渐进式加载和懒初始化

### 6.2 用户风险
- **学习成本**：详细文档和交互式教程
- **隐私担忧**：明确的数据使用说明
- **工作流中断**：所有功能可独立关闭

## 7. 未来展望

### 7.1 短期演进（3-6个月）
- 多语言支持（JavaScript、Go）
- 团队协作功能
- 插件生态系统
- Web 界面补充

### 7.2 长期愿景（1年+）
- 自主学习和进化
- 跨项目知识图谱
- 主动式开发辅助
- 开源社区建设