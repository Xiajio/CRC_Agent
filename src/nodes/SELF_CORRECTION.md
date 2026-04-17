# Planner Self-Correction Loop (自我纠错循环)

## 概述

这是一个**自我纠错机制**，让 Agent 能够从错误中学习并自动调整策略，而不是直接报错给用户。

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                   Self-Correction Loop                       │
└─────────────────────────────────────────────────────────────┘

User Input
    ↓
INTENT → PLANNER (生成计划)
    ↓
ROUTER (根据计划分发)
    ↓
KNOWLEDGE (执行工具)
    ↓
    ├─ 成功 → 标记 completed → 检查是否还有步骤
    │                              ├─ 有 → 回到 PLANNER (继续执行)
    │                              └─ 无 → END
    │
    └─ 失败 → 标记 failed → 回到 PLANNER (自我纠错)
                              ↓
                        分析错误，重新规划
                              ↓
                        ROUTER (执行新计划)
```

## 核心组件

### 1. State 增强 (src/state.py)

```python
class PlanStep(BaseModel):
    """计划步骤"""
    id: str
    description: str
    tool_needed: str
    status: str  # pending | in_progress | completed | failed
    reasoning: str
    error_message: str = ""  # [新增] 错误信息
    retry_count: int = 0     # [新增] 重试次数

class CRCAgentState(BaseModel):
    # ...
    current_plan: List[PlanStep]  # 当前计划
    scratchpad: str               # 思考过程
    plan_iteration_count: int = 0 # [新增] 规划迭代次数
```

### 2. Planner 自我纠错逻辑 (src/nodes/planner.py)

```python
def _should_skip_planning(state):
    """判断是否跳过规划"""
    # 检查是否有失败的步骤
    has_failed = any(s.status == 'failed' for s in state.current_plan)
    if has_failed:
        # 有失败，不跳过，需要重新规划
        return False
    
    # 如果有未完成的步骤，跳过（保持专注）
    if any(s.status == 'pending' for s in state.current_plan):
        return True
    
    return False

def _planner_node(state):
    """规划节点"""
    # 检测失败的步骤
    failed_steps = [s for s in state.current_plan if s.status == 'failed']
    
    if failed_steps:
        # 自我纠错模式
        print(f"[Planner] 检测到 {len(failed_steps)} 个失败步骤")
        for step in failed_steps:
            print(f"  ❌ {step.id}: {step.error_message}")
        
        # 提供错误上下文给 LLM
        error_context = "上一步执行失败：\n"
        for step in failed_steps:
            error_context += f"- {step.description}\n"
            error_context += f"  错误: {step.error_message}\n"
        
        # LLM 生成新计划
        new_plan = llm.invoke(f"{error_context}\n请分析错误并重新规划")
    else:
        # 正常规划模式
        new_plan = llm.invoke("请生成执行计划")
    
    return {
        "current_plan": new_plan,
        "plan_iteration_count": state.plan_iteration_count + 1,
    }
```

### 3. Knowledge 节点错误处理

**需要在 `src/nodes/clinical_nodes.py` 的 `node_knowledge_retrieval` 中添加：**

```python
def node_knowledge_retrieval(tools, model, **kwargs):
    def _run(state: CRCAgentState):
        from src.nodes.planner import get_current_pending_step, mark_step_completed, mark_step_failed
        
        # 获取当前待执行的步骤
        current_step = get_current_pending_step(state)
        
        if not current_step:
            # 没有计划驱动，走原有逻辑
            # ... 原有代码 ...
            return {}
        
        # 根据 tool_needed 调用对应工具
        tool_type = current_step.tool_needed.lower()
        
        try:
            if "toc" in tool_type:
                # 调用 list_guideline_toc
                result = list_guideline_toc(...)
                
                # 检查是否有错误
                if "未找到" in result or "错误" in result:
                    # 标记为失败
                    new_plan = mark_step_failed(
                        state, 
                        current_step.id, 
                        f"目录查询失败: {result[:100]}"
                    )
                    return {
                        "current_plan": new_plan,
                        "scratchpad": state.scratchpad + f"\n[步骤失败] {current_step.id}\n"
                    }
                
                # 成功，标记为完成
                new_plan = mark_step_completed(state, current_step.id)
                return {
                    "current_plan": new_plan,
                    "messages": [AIMessage(content=result)],
                }
            
            # ... 其他工具类型 ...
            
        except Exception as e:
            # 捕获异常，标记为失败
            new_plan = mark_step_failed(state, current_step.id, str(e))
            return {
                "current_plan": new_plan,
                "scratchpad": state.scratchpad + f"\n[异常] {current_step.id}: {e}\n"
            }
    
    return _run
```

### 4. Graph 路由逻辑 (src/graph_builder.py)

```python
def route_after_knowledge(state: CRCAgentState) -> str:
    """Knowledge 执行后的路由"""
    plan = state.current_plan or []
    
    # 检查是否有失败的步骤
    if any(s.status == 'failed' for s in plan):
        return "planner"  # 回到 Planner 自我纠错
    
    # 检查是否还有待执行的步骤
    if any(s.status == 'pending' for s in plan):
        return "planner"  # 回到 Planner 继续执行
    
    # 所有步骤都完成
    return "end"

# 在 build_graph 中
builder.add_conditional_edges(
    NodeName.KNOWLEDGE,
    route_after_knowledge,
    {
        "planner": NodeName.PLANNER,
        "end": END,
    }
)
```

## 使用示例

### 示例 1：章节名称错误

**用户输入**：
```
查看 NCCN 指南的 "Treatment Guidelines" 章节
```

**执行流程**：

1. **Planner 生成计划**：
   ```
   Step 1: 读取 NCCN 指南的 "Treatment Guidelines" 章节
   Tool: chapter
   ```

2. **Knowledge 执行失败**：
   ```
   错误：未找到章节 "Treatment Guidelines"
   ```

3. **标记步骤为失败**：
   ```python
   step_1.status = "failed"
   step_1.error_message = "未找到章节 'Treatment Guidelines'"
   step_1.retry_count = 1
   ```

4. **回到 Planner 自我纠错**：
   ```
   [Planner] 检测到失败步骤
   ❌ step_1: 未找到章节 'Treatment Guidelines'
   
   分析：章节名称可能不正确，需要先查看完整目录
   ```

5. **Planner 生成新计划**：
   ```
   Step retry_1: 查看 NCCN 指南的完整目录
   Tool: toc
   Reasoning: 先列出所有章节，找到正确的章节名
   
   Step retry_2: 根据目录中的实际章节名读取内容
   Tool: chapter
   Reasoning: 使用从目录中获取的准确章节名
   ```

6. **执行新计划 → 成功**

### 示例 2：检索无结果

**用户输入**：
```
查询 T4b 期结肠癌的治疗方案
```

**执行流程**：

1. **Planner 生成计划**：
   ```
   Step 1: 检索 "T4b stage colon cancer treatment"
   Tool: search
   ```

2. **Knowledge 执行返回空结果**：
   ```
   检索结果：0 条
   ```

3. **标记步骤为失败**：
   ```python
   step_1.status = "failed"
   step_1.error_message = "检索关键词过于具体，无结果"
   ```

4. **Planner 自我纠错**：
   ```
   分析：关键词可能过于具体，尝试更宽泛的查询
   
   新计划：
   Step retry_1: 检索 "结肠癌 III期 高危 治疗"
   Tool: search
   Reasoning: 使用更宽泛的关键词，T4b 通常属于 III 期高危
   ```

5. **执行新计划 → 成功**

## 熔断保护

为避免无限循环，实现了多重保护：

1. **步骤重试次数限制**：
   ```python
   if step.retry_count >= 3:
       # 停止重试该步骤
   ```

2. **规划迭代次数限制**：
   ```python
   if state.plan_iteration_count >= 5:
       # 停止重新规划
   ```

3. **降级策略**：
   ```python
   if has_too_many_retries(state):
       # 使用常识给出兜底建议
       return fallback_response()
   ```

## 监控与调试

### Scratchpad 记录

所有思考过程都记录在 `scratchpad` 中：

```
[规划 - treatment_decision]
档案状态: cT4bN1cM0, pMMR
生成计划: 2 个步骤

[步骤失败] step_1
错误: 未找到章节 'Treatment Guidelines'

[自我纠错 - 迭代 2]
失败步骤: ['step_1']
新计划: 2 个步骤

[步骤完成] retry_1
成功获取目录: 15 个章节

[步骤完成] retry_2
成功读取章节: Stage III Treatment
```

### 日志输出

```
[Planner] 规划节点启动
[Planner] 检测到 1 个失败步骤
  ❌ [step_1] 读取章节 'Treatment Guidelines'
     错误: 未找到章节
     重试次数: 1

[Planner] 🔄 自我纠错模式
分析失败原因，调整策略...

[Planner] 生成计划 (2 步骤):
  • [retry_1] 查看NCCN指南完整目录
    工具: toc | 推理: 先列出所有章节

[Router] [P0-Plan] 执行计划步骤: 查看NCCN指南完整目录
[Router] 路由决策: -> knowledge (list_guideline_toc)

[Knowledge] 执行成功，标记步骤完成

[Graph Router] 还有 1 个待执行步骤，继续执行计划
```

## 最佳实践

1. **错误信息要具体**：
   ```python
   # ❌ 不好
   error_message = "失败"
   
   # ✅ 好
   error_message = "未找到章节 'Treatment Guidelines'，可用章节: Stage I, Stage II, ..."
   ```

2. **推理过程要清晰**：
   ```python
   reasoning = "上次章节名错误，这次先查看目录获取准确名称"
   ```

3. **工具选择要合理**：
   ```python
   # 如果 chapter 失败，不要再次尝试 chapter
   # 而是先用 toc 确认章节存在
   ```

4. **提供降级方案**：
   ```python
   if retry_count >= 3:
       # 不要继续重试，给出基于常识的兜底建议
       return "根据临床指南的一般原则，建议..."
   ```

## 扩展方向

1. **学习机制**：
   - 记录常见错误模式
   - 自动生成纠错规则

2. **并行尝试**：
   - 同时尝试多种策略
   - 选择最快成功的结果

3. **人类反馈**：
   - 当重试3次仍失败时，询问用户
   - 记录用户的纠正建议

4. **统计分析**：
   - 哪些工具最容易失败
   - 哪些纠错策略最有效
