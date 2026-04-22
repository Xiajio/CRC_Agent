"""
src/nodes/tools_executor.py
通用工具执行节点
"""

import json
from typing import List, Dict, Any
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.tools import BaseTool
from ..state import CRCAgentState

# 导入所有可用工具的注册表
from ..tools import list_all_tools 


def node_tool_executor(state: CRCAgentState) -> Dict[str, Any]:
    """
    通用工具执行器
    
    职责：
    1. 监听: 获取上一条消息中的 tool_calls
    2. 执行: 查找并运行对应的工具函数
    3. 格式化: 将结果转换为 UI 卡片或标准 ToolMessage
    4. 反馈: 将结果写回 State
    """
    
    # 1. 获取最后一条消息 (一定是 AIMessage 且包含 tool_calls)
    if not state.messages:
        return {"error": "No tool calls found in the last message."}

    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        # 理论上不应走到这里，但也做个防御
        return {"error": "No tool calls found in the last message."}

    # 2. 准备工具集 (这里可以优化为只加载当前上下文需要的工具)
    # 为了简单通用，我们加载所有工具字典
    tools_list = list_all_tools()
    tool_map = {t.name: t for t in tools_list}

    results = []
    
    # 3. 遍历并执行工具
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        print(f"🛠️ [Tool Executor] Executing: {tool_name}")
        
        if tool_name not in tool_map:
            content = f"Error: Tool '{tool_name}' not found."
        else:
            try:
                # === 执行工具 ===
                tool_instance = tool_map[tool_name]
                output = tool_instance.invoke(tool_args)
                
                # === 4. 统一格式化 (核心逻辑移到这里) ===
                content = _format_tool_output(tool_name, output, tool_args)
                
            except Exception as e:
                content = f"Error executing {tool_name}: {str(e)}"

        # 生成 ToolMessage
        results.append(ToolMessage(
            tool_call_id=tool_id,
            name=tool_name,
            content=str(content) # 这里可以是 JSON 字符串，也可以是自然语言
        ))

    # 返回结果更新 State
    return {"messages": results}


def _format_tool_output(tool_name: str, output: Any, args: Dict) -> str:
    """
    集中处理所有工具的输出格式化 (UI Adapter)
    在这里处理您的"肿瘤卡片"、"影像卡片"逻辑
    """
    import json
    
    # 导入卡片格式化器
    from ..tools.card_formatter import formatter
    
    # 调试日志：打印工具输出
    print(f"[DEBUG] _format_tool_output: tool_name={tool_name}")
    print(f"[DEBUG] output type: {type(output)}")
    if isinstance(output, dict):
        print(f"[DEBUG] output keys: {list(output.keys())}")
    
    # 1. 肿瘤检测工具的特殊处理
    if tool_name == "perform_comprehensive_tumor_check":
        if isinstance(output, dict):
            # 使用 CardFormatter 格式化结果
            formatted_result = formatter.format_tumor_screening_result(output)
            # 标记为肿瘤检测卡片
            formatted_result["_ui_type"] = "tumor_detection_card"
            print(f"[DEBUG] Formatted tumor result: {formatted_result.get('type', 'unknown')}")
            return json.dumps(formatted_result, ensure_ascii=False)
            
    # 2. 影像查询工具
    elif tool_name == "get_patient_imaging":
        if isinstance(output, dict):
            output["_ui_type"] = "imaging_card"
            return json.dumps(output, ensure_ascii=False)

    # 2.5 患者已有信息汇总工具
    elif tool_name == "summarize_patient_existing_info":
        if isinstance(output, dict):
            return output.get("summary") or json.dumps(output, ensure_ascii=False)

    # 2.6 病例简要摘要工具
    elif tool_name == "get_patient_case_info":
        if isinstance(output, dict) and "error" not in output:
            cm_stage = output.get("cm_stage") or "0"
            return (
                f"病例 {output.get('patient_id', 'N/A')}: "
                f"{output.get('gender', 'N/A')}/{output.get('age', 'N/A')}岁, "
                f"{output.get('tumor_location', 'N/A')} {output.get('histology_type', 'N/A')}分化癌, "
                f"分期 cT{output.get('ct_stage', '')}N{output.get('cn_stage', '')}M{cm_stage}\n\n"
                "已生成结构化卡片，可在侧边栏按需展开查看。"
            )

    # 3. 默认处理：直接返回字符串或原始对象
    if isinstance(output, (dict, list)):
        return json.dumps(output, ensure_ascii=False)
    
    return str(output)
