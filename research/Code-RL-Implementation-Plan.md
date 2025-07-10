# 代码生成强化学习入手实施计划

## 项目概述

基于对SWE-Agent生态和代码生成RL领域的深度分析，制定一个3-6个月的渐进式实施计划，从最简单的Minimal Code Gym开始，逐步过渡到真实的软件工程环境。

### 核心目标
- 建立代码生成RL的技术能力和深度理解
- 验证结构化代码表示+密集奖励的创新思路
- 探索可能的商业化机会和技术贡献点

### 技术路径
```
简单环境(字符串) → 结构化代码环境(AST) → 真实工程环境(SWE-Bench)
     ↓                    ↓                     ↓
  验证RL循环         验证代码表示            验证完整pipeline
```

---

## Phase 1: 基础RL循环验证 (第1-3周)

### **目标：掌握RL环境设计的核心机制**

#### Week 1: 环境搭建与基础循环
**本周任务**：创建最简单的gymnasium环境

```python
# 核心代码框架
import gymnasium as gym
import numpy as np

class StringManipulationGym(gym.Env):
    """最简单的字符串操作环境 - 目标：将字符串逆序"""
    
    def __init__(self, target_string="hello"):
        super().__init__()
        self.target = target_string[::-1]  # "olleh"
        self.max_length = len(target_string)
        
        # 观察空间：当前字符串的ASCII编码
        self.observation_space = gym.spaces.Box(
            low=0, high=127, shape=(self.max_length,), dtype=np.int32
        )
        
        # 动作空间：交换位置i和j
        self.action_space = gym.spaces.MultiDiscrete([
            self.max_length,  # 位置i
            self.max_length   # 位置j
        ])
    
    def reset(self):
        self.current_string = list("hello")
        return self._get_observation()
    
    def step(self, action):
        # 交换字符
        i, j = action
        self.current_string[i], self.current_string[j] = \
            self.current_string[j], self.current_string[i]
        
        obs = self._get_observation()
        reward = self._compute_reward()
        done = "".join(self.current_string) == self.target
        
        return obs, reward, done, {}
    
    def _compute_reward(self):
        current = "".join(self.current_string)
        if current == self.target:
            return 10.0  # 成功奖励
        else:
            # 基于编辑距离的中间奖励
            return -0.1 * self._edit_distance(current, self.target)
```

**Week 1 交付物**：
- [ ] 完整的gym环境实现
- [ ] 环境可以稳定运行100个episode而不崩溃
- [ ] 基础的可视化和日志记录

**Week 1 验证标准**：
```python
# 验证代码
env = StringManipulationGym()
for episode in range(10):
    obs = env.reset()
    total_reward = 0
    for step in range(50):
        action = env.action_space.sample()  # 随机动作
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    print(f"Episode {episode}: Total Reward = {total_reward}")
```

#### Week 2: Agent集成与学习验证

**本周任务**：集成标准RL算法，验证学习效果

```python
# 使用stable-baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# 验证环境
env = StringManipulationGym()
check_env(env)

# 训练agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 测试学习效果
obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    if done:
        print(f"Solved in {i} steps!")
        break
```

**Week 2 交付物**：
- [ ] PPO agent能够学会解决问题
- [ ] 学习曲线显示明显的性能提升
- [ ] 成功率达到80%以上

#### Week 3: 文档化与反思

**本周任务**：深度理解和文档化

**Week 3 交付物**：
- [ ] 详细的技术文档，解释observation_space, action_space, reward_function的设计理念
- [ ] 实验报告：不同奖励函数设计的效果对比
- [ ] 下一阶段的技术方案文档

**关键决策点**：RL基础循环是否完全理解？如否，延长此阶段。

---

## Phase 2: 结构化代码环境 (第4-9周)

### **目标：处理真实代码作为状态，设计语义丰富的动作空间**

#### Week 4-5: 代码表示研究与实现

**本周任务**：基于搜索结果，实现多层次的代码表示

```python
import ast
import numpy as np
from typing import Dict, List

class CodeRepresentation:
    """多层次代码表示类"""
    
    def __init__(self):
        self.tokenizer = None  # 简单的token化
        
    def represent_code(self, code_string: str) -> Dict:
        """返回代码的多层次表示"""
        try:
            tree = ast.parse(code_string)
            return {
                'raw_text': code_string,
                'ast_tree': tree,
                'ast_nodes': self._extract_ast_nodes(tree),
                'ast_vector': self._ast_to_vector(tree),
                'syntax_valid': True
            }
        except SyntaxError:
            return {
                'raw_text': code_string,
                'ast_tree': None,
                'ast_nodes': [],
                'ast_vector': np.zeros(100),  # 默认向量
                'syntax_valid': False
            }
    
    def _extract_ast_nodes(self, tree) -> List[Dict]:
        """提取AST节点信息"""
        nodes = []
        for node in ast.walk(tree):
            nodes.append({
                'type': type(node).__name__,
                'lineno': getattr(node, 'lineno', 0),
                'fields': [f for f in node._fields if hasattr(node, f)]
            })
        return nodes
    
    def _ast_to_vector(self, tree) -> np.ndarray:
        """将AST转换为固定维度向量 - 基于节点类型统计"""
        node_types = {}
        for node in ast.walk(tree):
            node_type = type(node).__name__
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # 转换为固定维度向量 (简化版本)
        vector = np.zeros(100)
        for i, (node_type, count) in enumerate(sorted(node_types.items())):
            if i < 100:
                vector[i] = count
        return vector

class RefactoringGym(gym.Env):
    """自动重构环境 - 中等复杂度"""
    
    def __init__(self):
        super().__init__()
        self.repr_engine = CodeRepresentation()
        
        # 观察空间：AST向量 + 原始代码长度
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(200,), dtype=np.float32
        )
        
        # 动作空间：(操作类型, 行号, 内容ID)
        self.action_space = gym.spaces.MultiDiscrete([
            3,   # DELETE, INSERT_AFTER, REPLACE
            50,  # 最大50行
            10   # 预定义内容模板ID
        ])
        
        # 预定义的重构模板
        self.refactor_templates = [
            'print(f"{variable}")',  # 转换为f-string
            'if variable is None:',   # None检查
            'return variable',        # 简单返回
            # ... 更多模板
        ]
    
    def reset(self):
        # 加载一个需要重构的Python代码片段
        self.current_code = '''
def format_name(first, last):
    return "Hello, " + first + " " + last + "!"
        '''
        self.target_pattern = 'f"'  # 目标：转换为f-string
        return self._get_observation()
    
    def step(self, action):
        action_type, line_num, content_id = action
        
        # 执行代码修改
        success = self._apply_action(action_type, line_num, content_id)
        
        obs = self._get_observation()
        reward = self._compute_reward(success)
        done = self._check_completion()
        
        return obs, reward, done, {}
    
    def _apply_action(self, action_type, line_num, content_id):
        """应用代码修改动作"""
        lines = self.current_code.split('\n')
        
        if line_num >= len(lines):
            return False
            
        try:
            if action_type == 0:  # DELETE
                if line_num < len(lines):
                    lines.pop(line_num)
            elif action_type == 1:  # INSERT_AFTER
                template = self.refactor_templates[content_id % len(self.refactor_templates)]
                lines.insert(line_num + 1, template)
            elif action_type == 2:  # REPLACE
                template = self.refactor_templates[content_id % len(self.refactor_templates)]
                lines[line_num] = template
            
            self.current_code = '\n'.join(lines)
            return True
        except:
            return False
    
    def _compute_reward(self, action_success):
        """计算奖励 - 多层次反馈"""
        if not action_success:
            return -1.0
        
        repr_result = self.repr_engine.represent_code(self.current_code)
        
        reward = 0.0
        
        # 语法有效性奖励
        if repr_result['syntax_valid']:
            reward += 0.1
        else:
            reward -= 0.5
        
        # 重构目标进度奖励
        if self.target_pattern in self.current_code:
            reward += 1.0
        
        # 代码复杂度奖励 (简化版)
        ast_complexity = len(repr_result['ast_nodes'])
        if ast_complexity < 20:  # 鼓励简洁
            reward += 0.2
        
        return reward
```

**Week 4-5 交付物**：
- [ ] 完整的代码表示引擎，支持AST解析和向量化
- [ ] 重构环境的MVP版本
- [ ] AST表示效果的验证实验

#### Week 6-7: 动作空间优化

**本周任务**：设计更智能的动作空间

**改进的动作空间设计**：
```python
class SmartActionSpace:
    """基于AST的智能动作空间"""
    
    def __init__(self):
        self.common_refactors = {
            'string_format': self._convert_to_fstring,
            'none_check': self._add_none_check,
            'list_comprehension': self._convert_to_list_comp,
            'remove_unused': self._remove_unused_vars
        }
    
    def get_available_actions(self, ast_tree):
        """基于当前AST状态，返回可用的重构动作"""
        actions = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                # 发现字符串拼接，建议f-string
                actions.append(('string_format', node.lineno))
            
            if isinstance(node, ast.Name) and not self._has_none_check(node):
                # 发现变量使用，建议添加None检查
                actions.append(('none_check', node.lineno))
        
        return actions
```

**Week 6-7 交付物**：
- [ ] 基于AST的智能动作空间
- [ ] 动作有效性大幅提升（>50%的动作产生有意义的代码变更）

#### Week 8-9: 密集奖励设计

**本周任务**：实现多维度的代码质量评估

```python
class CodeQualityEvaluator:
    """代码质量多维度评估器"""
    
    def __init__(self):
        self.pylint_cmd = ["python", "-m", "pylint", "--score=yes"]
        
    def evaluate(self, code_string: str) -> Dict[str, float]:
        """返回多维度的代码质量分数"""
        results = {}
        
        # 1. 语法有效性
        results['syntax_valid'] = self._check_syntax(code_string)
        
        # 2. 静态分析评分
        results['pylint_score'] = self._run_pylint(code_string)
        
        # 3. 复杂度分析
        results['complexity'] = self._compute_complexity(code_string)
        
        # 4. 重构目标完成度
        results['refactor_progress'] = self._check_refactor_targets(code_string)
        
        return results
    
    def compute_dense_reward(self, before_scores: Dict, after_scores: Dict) -> float:
        """计算基于改进程度的密集奖励"""
        reward = 0.0
        
        # 语法有效性变化
        if after_scores['syntax_valid'] and not before_scores['syntax_valid']:
            reward += 2.0
        elif not after_scores['syntax_valid'] and before_scores['syntax_valid']:
            reward -= 5.0
        
        # Pylint分数改进
        pylint_improvement = after_scores['pylint_score'] - before_scores['pylint_score']
        reward += pylint_improvement * 0.1
        
        # 复杂度降低奖励
        complexity_reduction = before_scores['complexity'] - after_scores['complexity']
        reward += complexity_reduction * 0.05
        
        # 重构进度奖励
        progress_improvement = after_scores['refactor_progress'] - before_scores['refactor_progress']
        reward += progress_improvement * 1.0
        
        return reward
```

**Week 8-9 交付物**：
- [ ] 多维度代码质量评估系统
- [ ] 密集奖励函数，提供即时反馈
- [ ] Agent在重构任务上的明显学习效果

**Phase 2 决策点**：代码表示和动作空间是否足够有效？能否稳定处理真实代码？

---

## Phase 3: 真实工程环境集成 (第10-16周)

### **目标：集成SWE-Bench，处理复杂的软件工程任务**

#### Week 10-12: Docker化测试环境

**本周任务**：攻克SWE-Bench的环境复杂性

```python
import docker
import tempfile
import subprocess
from pathlib import Path

class SWEBenchRunner:
    """SWE-Bench任务执行器"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.base_images = {
            'python3.8': 'python:3.8-slim',
            'python3.9': 'python:3.9-slim',
        }
    
    def setup_problem(self, problem_id: str) -> Dict:
        """设置特定问题的环境"""
        # 从SWE-Bench数据集加载问题
        problem_data = self._load_problem(problem_id)
        
        # 创建Docker容器
        container = self._create_container(problem_data)
        
        # 安装依赖
        self._install_dependencies(container, problem_data)
        
        return {
            'container': container,
            'problem_data': problem_data,
            'test_commands': problem_data['test_commands']
        }
    
    def test_patch(self, setup_result: Dict, patch_content: str) -> Dict:
        """测试代码补丁"""
        container = setup_result['container']
        problem_data = setup_result['problem_data']
        
        try:
            # 应用补丁
            self._apply_patch(container, patch_content, problem_data['target_files'])
            
            # 运行测试
            test_results = self._run_tests(container, setup_result['test_commands'])
            
            return {
                'success': test_results['passed'],
                'test_output': test_results['output'],
                'error_log': test_results.get('errors', ''),
                'execution_time': test_results['execution_time']
            }
            
        except Exception as e:
            return {
                'success': False,
                'test_output': '',
                'error_log': str(e),
                'execution_time': 0
            }
    
    def _run_tests(self, container, test_commands: List[str]) -> Dict:
        """在容器中运行测试命令"""
        results = []
        total_time = 0
        
        for cmd in test_commands:
            start_time = time.time()
            try:
                result = container.exec_run(cmd, stdout=True, stderr=True)
                execution_time = time.time() - start_time
                total_time += execution_time
                
                results.append({
                    'command': cmd,
                    'return_code': result.exit_code,
                    'output': result.output.decode('utf-8'),
                    'passed': result.exit_code == 0,
                    'execution_time': execution_time
                })
            except Exception as e:
                results.append({
                    'command': cmd,
                    'return_code': -1,
                    'output': '',
                    'passed': False,
                    'error': str(e),
                    'execution_time': time.time() - start_time
                })
        
        overall_passed = all(r['passed'] for r in results)
        
        return {
            'passed': overall_passed,
            'output': '\n'.join(r['output'] for r in results),
            'execution_time': total_time,
            'individual_results': results
        }
```

**Week 10-12 交付物**：
- [ ] 单个SWE-Bench问题的完整测试pipeline
- [ ] 函数`run_test(problem_id, patch) -> (success, logs)`稳定工作
- [ ] 至少3个不同问题的成功运行记录

#### Week 13-14: SWE-Bench环境封装

**本周任务**：将Docker化的测试集成到gym环境

```python
class SWEBenchGym(gym.Env):
    """基于SWE-Bench的真实软件工程环境"""
    
    def __init__(self, problem_id: str):
        super().__init__()
        self.problem_id = problem_id
        self.runner = SWEBenchRunner()
        self.max_attempts = 10
        
        # 观察空间：文件内容 + 测试日志 (简化版)
        self.observation_space = gym.spaces.Text(50000)
        
        # 动作空间：代码补丁字符串
        self.action_space = gym.spaces.Text(10000)
        
    def reset(self):
        """重置环境，加载问题"""
        self.setup_result = self.runner.setup_problem(self.problem_id)
        self.attempts = 0
        
        # 构造观察：相关文件内容 + 失败的测试输出
        observation = self._build_observation()
        return observation
    
    def step(self, action: str):
        """执行一个补丁尝试"""
        self.attempts += 1
        
        # action是一个代码补丁字符串
        test_result = self.runner.test_patch(self.setup_result, action)
        
        # 计算奖励
        reward = 10.0 if test_result['success'] else -0.5
        
        # 检查是否完成
        done = test_result['success'] or self.attempts >= self.max_attempts
        
        # 更新观察（包含新的测试结果）
        observation = self._build_observation_with_result(test_result)
        
        return observation, reward, done, {
            'test_passed': test_result['success'],
            'test_output': test_result['test_output'],
            'attempts': self.attempts
        }
    
    def _build_observation(self) -> str:
        """构建当前状态的观察"""
        problem_data = self.setup_result['problem_data']
        
        observation_parts = []
        
        # 1. 问题描述
        observation_parts.append(f"PROBLEM: {problem_data['description']}")
        
        # 2. 相关文件内容
        for file_path, content in problem_data['relevant_files'].items():
            observation_parts.append(f"FILE: {file_path}\n{content}")
        
        # 3. 失败的测试输出
        if hasattr(self, 'last_test_result'):
            observation_parts.append(f"LAST_TEST_OUTPUT: {self.last_test_result['test_output']}")
        
        return '\n---\n'.join(observation_parts)
```

**Week 13-14 交付物**：
- [ ] 完整的SWE-Bench gym环境
- [ ] 环境可以加载和执行至少5个不同的问题
- [ ] 基础的错误处理和日志记录

#### Week 15-16: 本地LLM Agent集成

**本周任务**：使用本地LLM作为智能agent

```python
class LocalLLMSWEAgent:
    """基于本地LLM的软件工程Agent"""
    
    def __init__(self, model_type="transformers", model_name="Salesforce/codet5-small"):
        if model_type == "transformers":
            self.llm = LocalCodeLLM(model_name)
        elif model_type == "ollama":
            self.llm = OllamaCodeLLM(model_name)
        else:
            raise ValueError("model_type must be 'transformers' or 'ollama'")
        
        self.model_type = model_type
        
    def generate_patch(self, observation: str, previous_attempts: List[str] = None) -> str:
        """基于观察生成代码补丁"""
        
        # 构建prompt (本地模型通常需要更简洁的prompt)
        prompt = f"""
请修复以下Python代码中的bug：

{observation}

{self._format_previous_attempts(previous_attempts)}

生成一个代码补丁来修复这个问题：
"""
        
        try:
            if self.model_type == "transformers":
                return self.llm.generate_code(prompt, max_length=512)
            else:  # ollama
                return self.llm.generate_code(prompt)
            
        except Exception as e:
            print(f"本地LLM错误: {e}")
            return "# 无法生成补丁"
    
    def _format_previous_attempts(self, attempts: List[str]) -> str:
        """格式化之前的尝试"""
        if not attempts:
            return ""
        
        formatted = ["之前的尝试都失败了："]
        for i, attempt in enumerate(attempts[-3:]):  # 只显示最近3次
            formatted.append(f"尝试 {i+1}:")
            formatted.append(attempt)
            formatted.append("---")
        
        return '\n'.join(formatted)

# 完整的本地训练循环
def train_local_llm_agent():
    """本地LLM agent的完整训练循环"""
    
    # 选择几个相对简单的SWE-Bench问题
    test_problems = [
        "django__django-12453",
        "requests__requests-2317", 
        "flask__flask-1694"
    ]
    
    # 选择合适的本地模型 (根据GPU内存)
    import torch
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 14:
            agent = LocalLLMSWEAgent("ollama", "codellama:7b")
        elif vram_gb >= 8:
            agent = LocalLLMSWEAgent("transformers", "bigcode/starcoder")
        else:
            agent = LocalLLMSWEAgent("transformers", "Salesforce/codet5-small")
    else:
        print("⚠️  没有检测到GPU，使用CPU模式 (会很慢)")
        agent = LocalLLMSWEAgent("transformers", "Salesforce/codet5-small")
    
    results = []
    
    for problem_id in test_problems:
        print(f"\n处理问题: {problem_id}")
        
        env = SWEBenchGym(problem_id)
        observation = env.reset()
        
        attempts = []
        
        for attempt in range(5):  # 最多5次尝试
            print(f"  尝试 {attempt + 1}")
            
            # 生成补丁 (本地推理)
            patch = agent.generate_patch(observation, attempts)
            attempts.append(patch)
            
            # 执行
            observation, reward, done, info = env.step(patch)
            
            print(f"    奖励: {reward}")
            print(f"    测试通过: {info['test_passed']}")
            print(f"    GPU内存使用: {torch.cuda.memory_allocated()/1e9:.1f}GB" if torch.cuda.is_available() else "")
            
            if done:
                if info['test_passed']:
                    print(f"  ✅ 问题解决！")
                else:
                    print(f"  ❌ 达到最大尝试次数")
                break
        
        results.append({
            'problem_id': problem_id,
            'solved': info.get('test_passed', False),
            'attempts': len(attempts),
            'final_patch': attempts[-1] if attempts else None
        })
    
    # 总结结果
    solved_count = sum(1 for r in results if r['solved'])
    print(f"\n总结: {solved_count}/{len(results)} 问题解决")
    print(f"成功率: {solved_count/len(results)*100:.1f}%")
    print(f"平均尝试次数: {sum(r['attempts'] for r in results)/len(results):.1f}")
    
    return results

# 增加模型性能监控
class ModelPerformanceTracker:
    """监控本地模型的性能指标"""
    
    def __init__(self):
        self.inference_times = []
        self.memory_usage = []
        self.generation_quality = []
    
    def track_inference(self, start_time, end_time, memory_mb, quality_score):
        self.inference_times.append(end_time - start_time)
        self.memory_usage.append(memory_mb)
        self.generation_quality.append(quality_score)
    
    def get_stats(self):
        import statistics
        return {
            'avg_inference_time': statistics.mean(self.inference_times),
            'avg_memory_usage': statistics.mean(self.memory_usage),
            'avg_quality': statistics.mean(self.generation_quality),
            'total_inferences': len(self.inference_times)
        }
```

**Week 15-16 交付物**：
- [ ] 完整的本地LLM+环境自动化循环
- [ ] 在至少3个SWE-Bench问题上的尝试记录
- [ ] 本地模型性能分析报告 (推理时间、内存使用、质量评估)
- [ ] 不同模型大小的效果对比

**Phase 3 决策点**：是否成功建立了完整的代码生成RL pipeline？

---

## 资源需求与成本预算

### 计算资源

#### 硬件要求 (重要修正：使用本地开源模型)

**最低配置**：
- GPU: GTX 1660 (6GB VRAM) 
- RAM: 16GB
- 存储: 20GB (模型文件)
- 网络: 下载模型需要稳定连接

**推荐配置**：
- GPU: RTX 3060/4060 (12GB VRAM)
- RAM: 32GB  
- 存储: 50GB
- 备用方案: Google Colab Pro+ (~$200/月)

**开源模型选择策略**：
```python
def choose_model(vram_gb):
    if vram_gb >= 14:
        return "CodeLlama-7B-Instruct"  # 最佳质量
    elif vram_gb >= 8:
        return "StarCoder-1B"           # 平衡选择  
    elif vram_gb >= 4:
        return "CodeT5-small"           # 轻量级
    else:
        return "使用Google Colab"        # 本地资源不足
```

**渐进升级路径**：
1. **Week 1-3**: CodeT5-small (验证pipeline)
2. **Week 4-6**: StarCoder-1B (提升质量)  
3. **Week 7+**: CodeLlama-7B (最终性能)

#### 成本估算 (大幅降低)

- **电费**: ~$50/月 (24小时训练)
- **云GPU备选**: ~$200/月 (Google Colab Pro+)
- **API费用**: $0 ✅ (使用开源模型)
- **总成本**: $50-200/月 vs 原计划$500+/月

### 时间投入
- **全职投入**: 每周30-40小时
- **兼职投入**: 每周15-20小时 (时间线延长到6个月)

### 工具和依赖

```bash
# 基础环境
pip install gymnasium stable-baselines3 torch
pip install transformers accelerate bitsandbytes  # 本地LLM
pip install docker pylint ast
pip install pandas matplotlib  # 分析和可视化

# 可选: 简化的模型部署
pip install ollama  # 或者 curl https://ollama.ai/install.sh | sh

# SWE-Bench相关
git clone https://github.com/SWE-Gym/SWE-Gym.git

# 模型下载 (选择其中之一)
huggingface-cli download Salesforce/codet5-small
# 或者
ollama pull codellama:7b
```

### 本地LLM集成框架

```python
# 方案1: Transformers + 本地GPU
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LocalCodeLLM:
    def __init__(self, model_name="Salesforce/codet5-small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # 节省显存
            device_map="auto"
        )
    
    def generate_code(self, prompt, max_length=512):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# 方案2: Ollama (更简单的部署)
import ollama

class OllamaCodeLLM:
    def __init__(self, model="codellama:7b"):
        self.model = model
        # 需要先: ollama pull codellama:7b
    
    def generate_code(self, prompt):
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={'temperature': 0.7}
        )
        return response['response']
```

---

## 里程碑与决策节点

### 关键决策点

| 时间点 | 决策问题 | 继续条件 | 备选方案 |
|---------|----------|----------|----------|
| Week 3 | 是否理解RL基础循环？ | 环境稳定，agent学会解决玩具问题 | 延长Phase 1 |
| Week 9 | 代码表示是否有效？ | AST方法显著优于raw text | 探索其他表示方法 |
| Week 12 | Docker环境是否可控？ | 单个SWE-Bench问题可稳定运行 | 转向简化的代码任务 |
| Week 16 | 是否具备实用价值？ | 至少1个SWE-Bench问题被解决 | 重新定义成功标准 |

### 退出条件
- **技术退出**: 连续2周无明显进展
- **成本退出**: LLM API费用超过$1000/月
- **时间退出**: 超过预定时间线50%

### 成功指标

#### 最小成功 (Phase 1)
- [ ] 掌握RL环境设计的核心原理
- [ ] 有一个可工作的toy environment

#### 中等成功 (Phase 2)  
- [ ] 实现结构化的代码表示方法
- [ ] 设计有效的代码修改动作空间
- [ ] 验证密集奖励的效果

#### 最大成功 (Phase 3)
- [ ] 成功运行完整的SWE-Bench pipeline
- [ ] 解决至少1个真实的GitHub issue
- [ ] 产生可发表的技术洞察或开源贡献

---

## 风险缓解策略

### 技术风险

**风险1: AST表示效果不佳**
- *检测*: Week 6比较AST vs raw text的agent性能
- *缓解*: 准备基于token embedding的备选方案
- *应对*: 转向混合表示方法

**风险2: Docker环境过于复杂**
- *检测*: Week 10-11期间设置时间>预期2倍
- *缓解*: 提前准备SWE-Bench Lite的子集
- *应对*: 转向单文件的代码生成任务

**风险3: 本地模型性能不足**
- *检测*: 本地模型生成质量明显低于预期
- *缓解*: 优化prompt工程，使用更大的模型
- *应对*: 混合策略：本地模型+少量API调用验证

### 项目风险

**风险4: 时间管理失控**
- *缓解*: 每周五进行进度review和下周计划
- *应对*: 优先保证Phase 1-2的质量，Phase 3可选

**风险5: 失去动机**
- *缓解*: 每个Phase都有清晰的交付物和成就感
- *应对*: 调整为更实际的个人学习目标

---

## 预期产出与价值

### 短期产出 (3个月)
- **技术资产**: 完整的代码生成RL框架
- **知识资产**: 深度的领域理解和实战经验
- **网络资产**: 通过开源贡献建立技术声誉

### 中期产出 (6个月)
- **开源项目**: 完整的工具链开源
- **技术报告**: 详细的实验结果和技术洞察
- **商业机会**: 基于实验结果的商业化方向

### 长期价值 (12个月)
- **技术领先**: 在代码生成RL领域的专业能力
- **商业化**: 可能的产品化或服务化机会
- **学术价值**: 可发表的研究成果

---

## 下一步行动

### 立即行动 (本周)
1. [ ] 搭建本地LLM开发环境
2. [ ] 测试CodeT5-small的基础代码生成能力
3. [ ] 实现StringManipulationGym + 本地LLM的MVP
4. [ ] 验证GPU资源和模型加载

### 30天目标
- [ ] 完成Phase 1的所有里程碑
- [ ] 开始Phase 2的代码表示研究
- [ ] 建立实验跟踪和日志系统
- [ ] 建立本地模型性能baseline

### 模型准备检查清单
```bash
# GPU环境检查
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 模型下载验证
huggingface-cli download Salesforce/codet5-small
# 或者
ollama pull codellama:7b

# 基础推理测试
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-small')
model = AutoModelForCausalLM.from_pretrained('Salesforce/codet5-small')
print('模型加载成功!')
"
```

**重要修正总结：使用本地开源模型替代API调用，将成本从$500+/月降低到$50-200/月，同时获得更好的控制性和隐私保护。这个计划现在更加实际可行，每个阶段都有明确的技术验证和成本控制。**