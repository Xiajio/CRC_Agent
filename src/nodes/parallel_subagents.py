"""
并行子智能体执行节点

功能：
- 读取 Planner 输出的 parallel_group + assignee
- 同组并行执行多个子任务
- 汇总子智能体报告与引用
"""
import asyncio
from typing import Dict, Any, List, Tuple

from langchain_core.tools import BaseTool

from ..state import CRCAgentState, PlanStep
from .sub_agent import (
    create_rag_researcher,
    create_web_researcher,
    SubAgentContext,
)


CASE_DATABASE_SYSTEM_PROMPT = """你是一个病例与影像信息检索助理。
请使用可用的数据库/病例/影像查询工具获取用户请求所需的信息，
并将结果简要整理为 <report></report>。
"""

GENERIC_TASK_SYSTEM_PROMPT = """你是一个专业医学任务助理。
请根据任务描述使用合适工具完成检索或分析，并将结论整理为 <report></report>。
"""


def _infer_assignee(step: PlanStep) -> str:
    tool = (step.tool_needed or "").lower()
    if "web" in tool:
        return "web_search"
    if any(k in tool for k in ["toc", "chapter", "search", "guideline", "knowledge"]):
        return "knowledge"
    if any(k in tool for k in ["database", "case"]):
        return "case_database"
    if any(k in tool for k in ["imaging", "ct", "mri", "radiology"]):
        return "rad_agent"
    if any(k in tool for k in ["pathology", "clam"]):
        return "path_agent"
    return "knowledge"


def _select_tools_for_step(step: PlanStep, tools: List[BaseTool]) -> List[BaseTool]:
    tool = (step.tool_needed or "").lower()
    tool_map = {t.name: t for t in tools if hasattr(t, "name")}
    if "web" in tool:
        return [t for t in tools if "web" in getattr(t, "name", "").lower()]
    if any(k in tool for k in ["database", "case"]):
        return [
            t for t in tools
            if any(k in getattr(t, "name", "").lower() for k in ["database", "case", "patient", "imaging"])
        ]
    # 默认返回全部工具，让子智能体自行选择
    return list(tool_map.values())


async def _run_subagent_for_step(
    step: PlanStep,
    model,
    tools: List[BaseTool],
    show_thinking: bool
) -> Tuple[str, bool, str, List[Dict[str, Any]], str]:
    assignee = (step.assignee or _infer_assignee(step)).lower()
    selected_tools = _select_tools_for_step(step, tools)

    if assignee == "web_search":
        agent = create_web_researcher(
            model=model,
            task=step.description,
            max_iterations=4,
            show_thinking=show_thinking,
        )
    elif assignee == "knowledge":
        agent = create_rag_researcher(
            model=model,
            task=step.description,
            max_iterations=4,
            show_thinking=show_thinking,
        )
    elif assignee == "case_database":
        agent = SubAgentContext(
            model=model,
            system_prompt=CASE_DATABASE_SYSTEM_PROMPT,
            task_description=step.description,
            max_iterations=3,
            show_thinking=show_thinking,
        )
    else:
        # rad_agent/path_agent 暂时走通用逻辑（若无对应工具将失败）
        agent = SubAgentContext(
            model=model,
            system_prompt=GENERIC_TASK_SYSTEM_PROMPT,
            task_description=step.description,
            max_iterations=3,
            show_thinking=show_thinking,
        )

    if not selected_tools:
        return step.id, False, "无可用工具，无法执行并行子任务", [], assignee

    result = await agent.execute_with_tools(selected_tools)
    return step.id, result.success, result.report, result.references, assignee


def node_parallel_subagents(
    model,
    tools: List[BaseTool],
    streaming: bool = False,
    show_thinking: bool = True,
):
    """
    并行子智能体执行节点（异步版本）
    """

    async def _run(state: CRCAgentState) -> Dict[str, Any]:
        plan = state.current_plan or []
        pending_parallel = [
            s for s in plan
            if s.status == "pending" and getattr(s, "parallel_group", None)
        ]

        if not pending_parallel:
            return {}

        target_group = pending_parallel[0].parallel_group
        group_steps = [s for s in pending_parallel if s.parallel_group == target_group]

        if show_thinking:
            print(f"\n[ParallelSubAgents] ⚡ 并行组: {target_group} | 步骤数: {len(group_steps)}")

        results: Dict[str, Dict[str, Any]] = {}
        references: List[Dict[str, Any]] = []
        reports: List[Dict[str, Any]] = []

        # 使用 asyncio.gather 并行执行异步任务
        tasks = [
            _run_subagent_for_step(step, model, tools, show_thinking)
            for step in group_steps
        ]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for step, task_result in zip(group_steps, task_results):
            if isinstance(task_result, Exception):
                step_id = step.id
                success = False
                report = f"并行子任务执行异常: {task_result}"
                refs = []
                assignee = step.assignee or _infer_assignee(step)
            else:
                step_id, success, report, refs, assignee = task_result

            results[step_id] = {
                "success": success,
                "report": report,
                "assignee": assignee,
            }
            references.extend(refs or [])
            reports.append({
                "step_id": step_id,
                "assignee": assignee,
                "tool_needed": step.tool_needed,
                "parallel_group": step.parallel_group,
                "success": success,
                "report": report,
                "references": refs or [],
            })

        # 更新计划状态
        updated_plan = []
        for step in plan:
            if step.id in results:
                outcome = results[step.id]
                if outcome["success"]:
                    step.status = "completed"
                else:
                    step.status = "failed"
                    step.error_message = outcome["report"]
                    step.retry_count += 1
            updated_plan.append(step)

        merged_refs = list(state.retrieved_references or [])
        if references:
            merged_refs.extend(references)

        return {
            "current_plan": updated_plan,
            "retrieved_references": merged_refs,
            "subagent_reports": reports,
        }

    return _run
