"""
Sub-Agent Context Isolation Framework

核心概念：
- 子智能体在独立的 messages 列表中工作（沙箱）
- 主智能体只看到最终的蒸馏结果
- 子智能体的"脏工作"（多次搜索、重试、阅读无关内容）不污染主上下文

使用方式：
    async with SubAgentContext(model, task_prompt) as agent:
        result = await agent.execute_with_tools(tools)
    # agent 的整个对话历史已被销毁，只剩 result
"""

import json
import re
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate


# ==============================================================================
# 1. 子智能体上下文（沙箱）
# ==============================================================================

@dataclass
class SubAgentResult:
    """子智能体执行结果（蒸馏后的精华）"""
    success: bool
    report: str  # 提取的 <report> 内容或最终摘要
    references: List[Dict[str, Any]] = field(default_factory=list)  # 结构化引用
    raw_token_count: int = 0  # 子智能体消耗的 token 数（用于监控）
    iterations: int = 0  # 执行了多少轮
    error: Optional[str] = None


class SubAgentContext:
    """
    子智能体隔离上下文管理器
    
    核心特性：
    1. 独立的 messages 列表（不污染主智能体）
    2. 任务结束后自动销毁历史
    3. 强制结果蒸馏（只返回 <report> 内容）
    4. 内置重试和熔断机制
    
    使用示例：
        agent = SubAgentContext(
            model=model,
            system_prompt="你是一个专业的医学文献检索员...",
            task_description="查找 T3N1 直肠癌的新辅助治疗方案"
        )
        result = agent.execute_with_tools(rag_tools, max_iterations=5)
        # result.report 是蒸馏后的结果
        # agent 的内部历史已被丢弃
    """
    
    # 蒸馏提取的标签
    REPORT_TAG_PATTERN = re.compile(r'<report>(.*?)</report>', re.DOTALL | re.IGNORECASE)
    SUMMARY_TAG_PATTERN = re.compile(r'<summary>(.*?)</summary>', re.DOTALL | re.IGNORECASE)
    
    def __init__(
        self,
        model,
        system_prompt: str,
        task_description: str,
        max_iterations: int = 4,  # [优化] 默认从 5 降为 4
        show_thinking: bool = True
    ):
        """
        初始化子智能体上下文
        
        Args:
            model: LLM 模型实例
            system_prompt: 子智能体的系统提示（定义其角色和行为）
            task_description: 主智能体委派的具体任务
            max_iterations: 最大迭代次数（防止无限循环）【优化：默认 4 次】
            show_thinking: 是否显示子智能体的思考过程（调试用）
        """
        self.model = model
        self.system_prompt = system_prompt
        self.task_description = task_description
        self.max_iterations = max_iterations
        self.show_thinking = show_thinking
        
        # 独立的消息历史（这是隔离的核心）
        self._sandbox_messages: List[BaseMessage] = []
        
        # 收集的引用
        self._collected_references: List[Dict[str, Any]] = []
        
        # Token 计数（用于监控）
        self._token_count = 0
        self._iteration_count = 0
        
        # [优化] 工具调用统计，用于失败后切换工具
        self._tool_call_count: Dict[str, int] = {}
        self._tool_failure_count: Dict[str, int] = {}
    
    async def execute_with_tools(
        self,
        tools: List[BaseTool],
        context_data: Dict[str, Any] = None
    ) -> SubAgentResult:
        """
        在隔离环境中执行任务（异步版本）

        Args:
            tools: 子智能体可用的工具列表
            context_data: 额外的上下文数据（如患者信息）

        Returns:
            SubAgentResult: 蒸馏后的结果
        """
        try:
            # 1. 初始化沙箱消息
            self._init_sandbox(context_data)

            # 2. 执行循环（工具调用 -> 结果 -> 继续/结束）
            final_response = await self._run_agent_loop(tools)

            # 3. 蒸馏结果（提取 <report> 标签内容）
            report = self._distill_report(final_response)
            
            if self.show_thinking:
                print(f"📦 [SubAgent] 任务完成，蒸馏报告长度: {len(report)} 字符")
                print(f"   迭代次数: {self._iteration_count}, 沙箱消息数: {len(self._sandbox_messages)}")
            
            return SubAgentResult(
                success=True,
                report=report,
                references=self._collected_references,
                raw_token_count=self._token_count,
                iterations=self._iteration_count
            )
            
        except Exception as e:
            if self.show_thinking:
                print(f"❌ [SubAgent] 执行失败: {e}")
            return SubAgentResult(
                success=False,
                report=f"子智能体执行失败: {str(e)}",
                error=str(e),
                iterations=self._iteration_count
            )
        finally:
            # 4. 销毁沙箱历史（这是隔离的关键）
            self._destroy_sandbox()
    
    def _init_sandbox(self, context_data: Dict[str, Any] = None):
        """初始化沙箱消息列表"""
        # 系统消息
        system_msg = SystemMessage(content=self.system_prompt)
        
        # 任务消息（包含上下文）
        task_content = f"## 任务\n{self.task_description}"
        if context_data:
            task_content += f"\n\n## 上下文\n{json.dumps(context_data, ensure_ascii=False, indent=2)}"
        task_content += "\n\n请开始执行任务。完成后，请将最终结果包裹在 <report></report> 标签中。"
        
        task_msg = HumanMessage(content=task_content)
        
        # 初始化沙箱
        self._sandbox_messages = [system_msg, task_msg]
        self._collected_references = []
        self._token_count = 0
        self._iteration_count = 0
        
        if self.show_thinking:
            print(f"🔒 [SubAgent] 沙箱已初始化，任务: {self.task_description[:50]}...")
    
    async def _run_agent_loop(self, tools: List[BaseTool]) -> str:
        """
        运行智能体循环（异步版本）

        【v5.1 优化】
        - 相同工具连续失败 2 次后建议切换其他工具
        - 减少无效迭代
        """
        tool_map = {t.name: t for t in tools}
        model_with_tools = self.model.bind_tools(tools) if tools else self.model

        while self._iteration_count < self.max_iterations:
            self._iteration_count += 1

            if self.show_thinking:
                print(f"   🔄 [SubAgent] 迭代 {self._iteration_count}/{self.max_iterations}")

            # 调用模型（异步）
            response = await model_with_tools.ainvoke(self._sandbox_messages)
            self._sandbox_messages.append(response)

            # 估算 token 消耗
            self._token_count += self._estimate_tokens(response.content or "")

            # 检查是否有工具调用
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # 执行工具调用
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]

                    # [优化] 统计工具调用次数
                    self._tool_call_count[tool_name] = self._tool_call_count.get(tool_name, 0) + 1

                    # [优化] 检查是否重复调用同一工具（超过 2 次）
                    if self._tool_call_count[tool_name] > 2:
                        if self.show_thinking:
                            print(f"      ⚠️ 工具 {tool_name} 已调用 {self._tool_call_count[tool_name]} 次")
                            print(f"         建议：使用其他工具或生成 <report>")

                    if self.show_thinking:
                        print(f"      🛠️ 调用工具: {tool_name}")

                    if tool_name in tool_map:
                        try:
                            result = await tool_map[tool_name].ainvoke(tool_args)
                            result_str = str(result)
                            
                            # 提取引用（如果有）
                            self._extract_references(result_str)
                            
                            # 估算 token
                            self._token_count += self._estimate_tokens(result_str)
                            
                            # [优化] 检查结果是否有效（非空且有实质内容）
                            is_valid_result = len(result_str) > 50 and "No relevant" not in result_str
                            if not is_valid_result:
                                self._tool_failure_count[tool_name] = self._tool_failure_count.get(tool_name, 0) + 1
                                if self._tool_failure_count[tool_name] >= 2:
                                    result_str += "\n\n⚠️ 此工具已连续失败2次，建议切换其他工具或直接生成报告。"
                            
                        except Exception as e:
                            result_str = f"工具执行错误: {e}"
                            self._tool_failure_count[tool_name] = self._tool_failure_count.get(tool_name, 0) + 1
                    else:
                        result_str = f"未知工具: {tool_name}"
                    
                    # 添加工具结果到沙箱
                    from langchain_core.messages import ToolMessage
                    self._sandbox_messages.append(
                        ToolMessage(content=result_str, tool_call_id=tool_id, name=tool_name)
                    )
            else:
                # 没有工具调用，检查是否包含 <report> 标签
                content = response.content or ""
                if self.REPORT_TAG_PATTERN.search(content) or self.SUMMARY_TAG_PATTERN.search(content):
                    # 任务完成
                    return content
                
                # 如果没有工具调用也没有 report，可能是模型认为任务完成
                # 返回最后的内容
                return content
        
        # 达到最大迭代次数
        if self.show_thinking:
            print(f"   ⚠️ [SubAgent] 达到最大迭代次数 {self.max_iterations}")
        
        # 返回最后一条消息的内容
        last_msg = self._sandbox_messages[-1]
        return last_msg.content if hasattr(last_msg, 'content') else ""
    
    def _distill_report(self, final_response: str) -> str:
        """
        蒸馏最终报告（提取 <report> 或 <summary> 标签内容）
        
        这是上下文隔离的关键步骤：
        - 只有这部分内容会返回给主智能体
        - 其他所有中间过程都被丢弃
        """
        # 尝试提取 <report> 标签
        match = self.REPORT_TAG_PATTERN.search(final_response)
        if match:
            return match.group(1).strip()
        
        # 尝试提取 <summary> 标签
        match = self.SUMMARY_TAG_PATTERN.search(final_response)
        if match:
            return match.group(1).strip()
        
        # 如果没有标签，尝试提取最后一段有意义的内容
        # 移除思考过程（<think> 标签）
        cleaned = re.sub(r'<think>.*?</think>', '', final_response, flags=re.DOTALL | re.IGNORECASE)
        
        # 取最后 1000 字符作为摘要
        if len(cleaned) > 1000:
            return "..." + cleaned[-1000:].strip()
        
        return cleaned.strip() if cleaned.strip() else final_response[:500]
    
    def _extract_references(self, tool_result: str):
        """从工具结果中提取引用"""
        pattern = r"<retrieved_metadata>(.*?)</retrieved_metadata>"
        match = re.search(pattern, tool_result, re.DOTALL)
        if match:
            try:
                refs = json.loads(match.group(1))
                if isinstance(refs, list):
                    self._collected_references.extend(refs)
            except json.JSONDecodeError:
                pass
    
    def _estimate_tokens(self, text: str) -> int:
        """估算 token 数量"""
        if not text:
            return 0
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)
    
    def _destroy_sandbox(self):
        """销毁沙箱历史（释放内存，确保隔离）"""
        if self.show_thinking and self._sandbox_messages:
            print(f"🗑️ [SubAgent] 销毁沙箱历史 ({len(self._sandbox_messages)} 条消息, ~{self._token_count} tokens)")

        # 清空消息列表
        self._sandbox_messages.clear()
        self._sandbox_messages = []

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        self._destroy_sandbox()
        return False


# ==============================================================================
# 2. 预定义的子智能体模板
# ==============================================================================

RAG_RESEARCHER_SYSTEM_PROMPT = """你是一个专业的医学文献检索研究员。

## 你的职责
1. 使用提供的检索工具查找相关医学文献
2. 评估检索结果的相关性和权威性
3. 综合多个来源的信息
4. 生成简洁、准确的研究报告

## 【重要】工作流程优化
1. **批量检索优先**：将多个关键词合并为一个查询，一次性检索 top_k=12
2. **避免重复检索**：不要对相似内容多次检索，已检索过的内容不要重复
3. 阅读和筛选结果，选出最相关的 6-8 条
4. 综合信息，撰写报告

## 检索策略
- ✅ 合并查询："T3N1 直肠癌 新辅助 化疗 指南"（一次检索）
- ❌ 分开查询：先查"T3N1"，再查"新辅助"，再查"化疗"...（多次重复）

## 输出要求
- 完成任务后，将最终结果包裹在 <report></report> 标签中
- 报告应简洁（500字以内），包含关键发现
- 列出主要参考来源（标注 Page 信息）

## 示例输出
<report>
根据 NCCN 直肠癌指南（2024版），对于 cT3N1 期低位直肠癌：

1. **新辅助治疗**：推荐长程同步放化疗（5-FU/卡培他滨 + 50.4Gy 放疗）
2. **手术时机**：放化疗后 8-12 周评估
3. **保肛可能性**：距肛 5cm 可尝试括约肌保留手术

参考来源：NCCN Guidelines Rectal Cancer v2.2024, p.45-52
</report>
"""

WEB_RESEARCHER_SYSTEM_PROMPT = """你是一个专业的网络信息研究员。

## 你的职责
1. 使用网络搜索工具查找最新信息
2. 识别权威来源（官方指南、医学期刊、学术机构）
3. 过滤广告和不可靠信息
4. 综合信息，生成研究报告

## 工作流程
1. 制定搜索策略（关键词组合）
2. 执行搜索并筛选结果
3. 深入阅读相关页面
4. 综合信息，撰写报告

## 输出要求
- 完成任务后，将最终结果包裹在 <report></report> 标签中
- 报告应标注信息来源
- 区分已确认的事实和存在争议的观点

## 注意事项
- 如果搜索失败，尝试换一个关键词
- 优先引用权威来源
- 对于无法找到的信息，明确说明
"""


# ==============================================================================
# 3. 便捷工厂函数
# ==============================================================================

def create_rag_researcher(
    model,
    task: str,
    patient_context: Dict[str, Any] = None,
    max_iterations: int = 5,
    show_thinking: bool = True
) -> SubAgentContext:
    """
    创建 RAG 检索子智能体
    
    用于 Decision 节点的指南检索任务
    """
    return SubAgentContext(
        model=model,
        system_prompt=RAG_RESEARCHER_SYSTEM_PROMPT,
        task_description=task,
        max_iterations=max_iterations,
        show_thinking=show_thinking
    )


def create_web_researcher(
    model,
    task: str,
    max_iterations: int = 5,
    show_thinking: bool = True
) -> SubAgentContext:
    """
    创建网络研究子智能体
    
    用于 Knowledge 节点的联网搜索任务
    """
    return SubAgentContext(
        model=model,
        system_prompt=WEB_RESEARCHER_SYSTEM_PROMPT,
        task_description=task,
        max_iterations=max_iterations,
        show_thinking=show_thinking
    )


# ==============================================================================
# 4. 异步执行辅助函数（用于现有节点集成）
# ==============================================================================

async def run_isolated_rag_search(
    model,
    queries: List[str],
    rag_tools: List[BaseTool],
    patient_context: Dict[str, Any] = None,
    show_thinking: bool = True,
    batch_mode: bool = True,  # [优化] 默认启用批量模式
) -> SubAgentResult:
    """
    执行隔离的 RAG 检索（异步版本）

    【v5.2 重要说明】
    这是治疗决策流程中唯一的 RAG 检索入口。
    - Planner 不再为 treatment_decision 意图生成知识检索步骤
    - 所有指南检索由 Decision 节点调用此函数统一执行
    - 避免了 Planner task2 与 SubAgent 的重复检索问题

    【v5.1 批量检索优化】
    - 合并多个查询为单次批量检索，减少重复查询
    - 使用 top_k=12 一次性获取足够的结果
    - 减少迭代次数，提高效率

    Args:
        model: LLM 模型
        queries: 检索关键词列表
        rag_tools: RAG 工具列表
        patient_context: 患者上下文
        show_thinking: 是否显示调试信息
        batch_mode: 是否使用批量检索模式（默认True）

    Returns:
        SubAgentResult: 蒸馏后的检索结果
    """
    # [优化] 批量模式：合并查询，一次检索
    if batch_mode and len(queries) > 1:
        # 合并查询词为单个组合查询
        combined_query = " ".join(queries)

        if show_thinking:
            print(f"📦 [SubAgent] 批量检索模式：合并 {len(queries)} 个查询")
            print(f"   组合查询: {combined_query[:80]}...")

        task = f"""
请使用提供的检索工具，查找以下临床问题的相关医学指南信息：

**组合查询**（一次性检索）：
{combined_query}

**原始查询点**（供参考）：
{chr(10).join(f"- {q}" for q in queries)}

**要求**：
1. 只执行 1 次检索（使用组合查询，top_k=12）
2. 从检索结果中筛选最相关的 6-8 条
3. 综合信息，生成简洁的报告（500字以内）
4. 在报告中标注主要参考来源

**重要**：不要对每个原始查询分别检索，只执行一次批量检索即可。
"""
        max_iter = 3  # 批量模式只需 3 次迭代
    else:
        # 单查询或禁用批量模式
        task = f"""
请使用提供的检索工具，查找以下关键词相关的医学指南信息：

检索关键词：
{chr(10).join(f"- {q}" for q in queries)}

要求：
1. 对每个关键词执行检索
2. 筛选最相关的结果
3. 综合信息，生成简洁的报告（500字以内）
4. 在报告中标注主要参考来源
"""
        max_iter = len(queries) + 3

    agent = create_rag_researcher(
        model=model,
        task=task,
        patient_context=patient_context,
        max_iterations=max_iter,
        show_thinking=show_thinking
    )

    return await agent.execute_with_tools(rag_tools, patient_context)


async def run_isolated_web_search(
    model,
    query: str,
    web_tools: List[BaseTool],
    local_context: str = "",
    patient_context: Dict[str, Any] = None,
    show_thinking: bool = True,
    max_iterations: int = None  # [新增] 允许自定义最大迭代次数
) -> SubAgentResult:
    """
    执行隔离的联网搜索（异步版本）

    用于 Knowledge 节点的联网搜索任务
    子智能体会在隔离的沙箱中进行多次搜索、阅读、筛选
    主智能体只会收到最终的蒸馏报告

    Args:
        model: LLM 模型
        query: 用户问题
        web_tools: 联网搜索工具列表
        local_context: 本地知识库已检索到的内容（供参考）
        patient_context: 患者上下文
        show_thinking: 是否显示调试信息
        max_iterations: 最大迭代次数（默认3，知识问答可传1提升速度）

    Returns:
        SubAgentResult: 蒸馏后的搜索结果
    """
    # 构建任务描述
    task_parts = [f"用户问题：{query}"]

    if local_context:
        # 只提供本地内容的摘要，避免上下文过长
        local_summary = local_context[:1000] + "..." if len(local_context) > 1000 else local_context
        task_parts.append(f"\n已有本地知识库信息（仅供参考）：\n{local_summary}")

    task_parts.append("""
要求：
1. 使用联网搜索工具查找权威信息
2. 优先查找官方指南、医学期刊、学术机构发布的内容
3. 过滤广告和不可靠来源
4. 综合所有信息，生成简洁的研究报告（800字以内）
5. 在报告中标注信息来源和可信度

如果搜索某个关键词失败，尝试换一个关键词。
如果多次搜索都失败，说明目前没有找到相关信息。
""")

    task = "\n".join(task_parts)

    # 如果没有指定 max_iterations，使用默认值 6
    if max_iterations is None:
        max_iterations = 6

    agent = create_web_researcher(
        model=model,
        task=task,
        max_iterations=max_iterations,
        show_thinking=show_thinking
    )

    return await agent.execute_with_tools(web_tools, patient_context)
