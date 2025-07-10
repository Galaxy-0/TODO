# 并行开发工作流指南 - 如何管理多个 Claude 会话

## 核心原则：清晰的边界和明确的目标

### 1. 前期规划模板

```markdown
## Feature A: 用户认证模块
### 目标
- 实现登录/注册功能
- JWT token 管理
- 密码加密存储

### 边界
- 只负责 /auth/* 路由
- 不涉及用户资料管理
- 不修改其他模块代码

### 接口约定
- 输出：JWT token 格式
- 依赖：数据库 User 表
- API：POST /auth/login, POST /auth/register

### 验收标准
- [ ] 单元测试覆盖率 > 80%
- [ ] API 文档完整
- [ ] 错误处理完善
```

### 2. CLAUDE.md 分支策略

为每个 worktree 创建独立的指令文件：

```bash
# 主仓库
/project/CLAUDE.md          # 通用规范

# 各个 worktree
/project-auth/CLAUDE_AUTH.md    # 认证模块专用指令
/project-api/CLAUDE_API.md      # API 模块专用指令
/project-ui/CLAUDE_UI.md        # UI 模块专用指令
```

### 3. 模块化 CLAUDE.md 示例

```markdown
# CLAUDE_AUTH.md

## 本分支任务
你正在开发用户认证模块（feature/auth）

## 具体目标
1. 实现用户注册 API
2. 实现用户登录 API  
3. 实现 JWT token 验证中间件

## 技术约定
- 使用 bcrypt 加密密码
- JWT 有效期 7 天
- 遵循 RESTful 规范

## 边界限制
- 不要修改 models/Product.js
- 不要修改 routes/order.js
- 只在 auth/ 目录下工作

## 接口规范
POST /auth/register
{
  "email": "string",
  "password": "string"
}

POST /auth/login
{
  "email": "string", 
  "password": "string"
}
返回: { "token": "jwt_token" }

## 依赖说明
- 数据库已有 users 表
- 可以使用 utils/db.js
- 环境变量 JWT_SECRET 已配置
```

### 4. 任务分解最佳实践

#### 垂直切分（推荐）
```
✅ 好的拆分：
- Feature A: 完整的用户认证模块
- Feature B: 完整的订单管理模块
- Feature C: 完整的支付集成

每个模块独立完整，接口清晰
```

#### 水平切分（谨慎使用）
```
⚠️ 需要更多协调：
- Feature A: 所有 API 接口
- Feature B: 所有前端页面
- Feature C: 所有数据库操作

容易产生依赖和冲突
```

### 5. 启动 Claude 会话的标准流程

```bash
# 1. 创建 worktree
git worktree add ../project-auth feature/auth

# 2. 准备专用指令
cd ../project-auth
cat > CLAUDE_AUTH.md << 'EOF'
# 认证模块开发任务

## 你的任务
开发用户认证模块...

## 验收标准
- [ ] 注册功能完成
- [ ] 登录功能完成
- [ ] 测试用例编写
EOF

# 3. 创建任务清单
cat > TODO_AUTH.md << 'EOF'
## 认证模块任务列表
- [ ] 设计数据库 schema
- [ ] 实现注册 API
- [ ] 实现登录 API
- [ ] 编写单元测试
- [ ] 编写 API 文档
EOF

# 4. 启动 Claude
claude --claude-md CLAUDE_AUTH.md "请阅读 TODO_AUTH.md 并开始开发认证模块"
```

### 6. 避免冲突的技巧

#### 目录隔离
```
/src
  /auth     # Feature A 专属
  /order    # Feature B 专属
  /payment  # Feature C 专属
  /common   # 共享代码（只读）
```

#### Git 配置
```bash
# 设置合并策略
git config merge.ours.driver true

# 在 .gitattributes 中标记
CLAUDE.md merge=ours
package-lock.json merge=ours
```

### 7. 进度同步机制

创建一个中央进度看板：

```markdown
# PROJECT_STATUS.md

## 并行开发进度

### Feature A: 认证模块
- 负责人：Claude Session 1
- 进度：60%
- 阻塞：无
- 预计完成：今天下午

### Feature B: 订单模块  
- 负责人：Claude Session 2
- 进度：30%
- 阻塞：等待认证模块的 JWT 格式
- 预计完成：明天

### 接口依赖关系
- 订单模块 → 需要认证模块的 JWT 验证
- 支付模块 → 需要订单模块的订单 ID 格式
```

### 8. 实际案例

```bash
# 场景：开发一个电商系统

# 1. 主分支创建整体规划
cat > DEVELOPMENT_PLAN.md << 'EOF'
## 模块划分
1. 用户系统（user-system）
   - 注册/登录
   - 个人资料
   - 地址管理

2. 商品系统（product-system）
   - 商品列表
   - 商品详情
   - 库存管理

3. 订单系统（order-system）
   - 创建订单
   - 订单查询
   - 订单状态

## 接口约定
[详细接口文档...]
EOF

# 2. 为每个模块创建 worktree
git worktree add ../shop-user feature/user
git worktree add ../shop-product feature/product  
git worktree add ../shop-order feature/order

# 3. 每个模块独立的 Claude 指令
# 用户系统
echo "专注于用户相关功能..." > ../shop-user/CLAUDE.md

# 商品系统
echo "专注于商品相关功能..." > ../shop-product/CLAUDE.md

# 订单系统
echo "专注于订单相关功能..." > ../shop-order/CLAUDE.md
```

### 9. 常见问题解决

#### Q: 如何处理共享代码？
A: 将共享代码放在 main 分支，其他分支只读引用

#### Q: 模块间有依赖怎么办？
A: 先定义接口契约，使用 mock 数据开发，最后集成

#### Q: 如何避免样式冲突？
A: 使用 CSS Modules 或命名空间

### 10. 检查清单

启动并行开发前，确保：

- [ ] 模块边界清晰定义
- [ ] 接口契约文档完成
- [ ] 每个模块有独立的 CLAUDE.md
- [ ] 依赖关系明确标注
- [ ] 冲突解决策略确定
- [ ] 集成测试计划制定

## 总结

成功的并行开发关键在于：
1. **清晰的模块边界**
2. **明确的接口契约**
3. **独立的任务指令**
4. **最小化的依赖关系**

记住：前期多花 30 分钟规划，能节省后期 3 小时的冲突处理！