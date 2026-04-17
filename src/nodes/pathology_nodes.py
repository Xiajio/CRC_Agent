"""
病理医生智能体节点 (Pathology Agent)

负责调度病理 CLAM 工具链：
- 病理切片分类（完整分析）
- 快速病理筛查
- 工具状态查询
- 综合病理分析（按患者ID自动查找切片）
"""

import os
import re
from typing import List, Optional, Literal

from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from ..state import CRCAgentState
from .node_utils import _latest_user_text
from .planner import get_current_pending_step, mark_step_completed
from ..tools.pathology_clam_tools import CLAM_TOOL_DIR


PathologyMode = Literal["full", "quick", "status"]


def _detect_pathology_mode(user_text: str) -> PathologyMode:
    """根据用户输入检测病理分析模式"""
    text = user_text.lower()

    status_keywords = ["状态", "可用", "依赖", "版本", "status", "gpu", "模型"]
    if any(kw in text for kw in status_keywords):
        return "status"

    quick_keywords = ["快速", "快检", "筛查", "初筛", "只要结果", "不需要热力图"]
    if any(kw in text for kw in quick_keywords):
        return "quick"

    return "full"


def _find_tool_by_name(tools: List[BaseTool], tool_name: str) -> Optional[BaseTool]:
    """根据工具名称查找工具"""
    for tool in tools:
        if hasattr(tool, "name") and tool.name == tool_name:
            return tool
    return None


def _extract_patient_id(user_text: str) -> Optional[str]:
    """从文本中提取患者ID（1-3位数字，补零到3位）"""
    match = re.search(r"(\d{1,3})号", user_text)
    if not match:
        match = re.search(r"(?:患者|病人|编号)\s*(\d{1,3})", user_text)
    if not match:
        match = re.search(r"(?<!\d)(\d{1,3})(?!\d)", user_text)
    if match:
        return match.group(1).zfill(3)
    return None


def _extract_slide_path(user_text: str) -> Optional[str]:
    """从文本中提取切片文件路径"""
    ext_pattern = r"(?:svs|tif|tiff|ndpi|mrxs|vms|vmu)"
    windows_pattern = rf"([A-Za-z]:\\[^\s]+\.{ext_pattern})"
    unix_pattern = rf"(/[^\\s]+\.{ext_pattern})"

    match = re.search(windows_pattern, user_text)
    if not match:
        match = re.search(unix_pattern, user_text)
    return match.group(1) if match else None


def _build_status_text(status: dict) -> str:
    """构建状态文本"""
    missing = status.get("missing_dependencies") or []
    missing_text = "无" if not missing else ", ".join(missing)
    return (
        "🧪 **病理 CLAM 工具状态**\n\n"
        f"- 模型已加载: {status.get('model_loaded')}\n"
        f"- 模型路径: {status.get('model_path')}\n"
        f"- 模型文件存在: {status.get('model_exists')}\n"
        f"- 工具目录: {status.get('clam_tool_dir')}\n"
        f"- 工具目录存在: {status.get('clam_tool_exists')}\n"
        f"- 缺失依赖: {missing_text}\n"
        f"- GPU可用: {status.get('gpu_available')}\n"
        f"- GPU名称: {status.get('gpu_name')}\n"
        f"- 工具版本: {status.get('tool_version')}\n"
    )


def _build_pathology_card_base() -> dict:
    """构造病理卡片基础字段"""
    tool_dir = os.path.abspath(CLAM_TOOL_DIR)
    tool_root_dir = os.path.abspath(os.path.join(CLAM_TOOL_DIR, os.pardir))
    return {
        "type": "pathology_card",
        "data": {
            "tool_dir": tool_dir,
            "tool_root_dir": tool_root_dir,
        },
    }


def node_pathology_agent(
    tools: List[BaseTool],
    model,
    streaming: bool = False,
    show_thinking: bool = True,
) -> Runnable:
    """
    病理医生智能体节点
    """

    def _run(state: CRCAgentState):
        user_text = _latest_user_text(state)

        if show_thinking:
            print(f"\n{'='*60}")
            print("🧑‍⚕️ [PathAgent] 病理医生智能体启动")
            print(f"{'='*60}")
            print(f"用户请求: {user_text[:100]}...")

        mode = _detect_pathology_mode(user_text)

        def _finalize_return(return_dict: dict) -> dict:
            pending_step = get_current_pending_step(state)
            if pending_step:
                updated_plan = mark_step_completed(state, pending_step.id)
                return_dict["current_plan"] = updated_plan
                if show_thinking:
                    print(f"[PathAgent] 标记步骤完成: [{pending_step.id}] {pending_step.description}")
            return return_dict

        # --- 工具状态查询 ---
        if mode == "status":
            status_tool = _find_tool_by_name(tools, "get_pathology_clam_status")
            if not status_tool:
                error_msg = "系统错误：未找到病理 CLAM 状态查询工具。"
                return _finalize_return({
                    "messages": [AIMessage(content=error_msg)],
                    "clinical_stage": "PathologyAnalysis_Error",
                    "error": "Pathology status tool not found",
                })

            status = status_tool.invoke({})
            report_text = _build_status_text(status)
            pathology_card = _build_pathology_card_base()
            pathology_card["data"]["status"] = status

            return _finalize_return({
                "messages": [AIMessage(
                    content=report_text,
                    additional_kwargs={"pathology_card": pathology_card},
                )],
                "findings": {
                    "pathology_tool_status": status,
                    "pathology_card": pathology_card,
                },
                "clinical_stage": "PathologyStatus_Completed",
                "error": None,
            })

        # --- 提取输入 ---
        slide_path = _extract_slide_path(user_text)
        patient_id = state.current_patient_id or _extract_patient_id(user_text)

        if not slide_path and not patient_id:
            error_msg = (
                "抱歉，我需要患者编号或切片文件路径才能进行病理分析。\n"
                "请提供例如：\n"
                "- '对93号患者进行病理分析'\n"
                "- '分析这张切片 E:\\\\slides\\\\sample.svs'"
            )
            return _finalize_return({
                "messages": [AIMessage(content=error_msg)],
                "clinical_stage": "PathologyAnalysis_Error",
                "error": "Missing patient ID or slide path for pathology analysis",
            })

        # --- 按输入类型执行 ---
        if slide_path:
            if mode == "quick":
                tool = _find_tool_by_name(tools, "quick_pathology_check")
            else:
                tool = _find_tool_by_name(tools, "pathology_slide_classify")

            if not tool:
                error_msg = "系统错误：未找到病理切片分析工具。"
                return _finalize_return({
                    "messages": [AIMessage(content=error_msg)],
                    "clinical_stage": "PathologyAnalysis_Error",
                    "error": "Pathology analysis tool not found",
                })

            if show_thinking:
                mode_text = "快速筛查" if mode == "quick" else "完整分析"
                print(f"🔬 [PathAgent] 开始对切片进行{mode_text}...")

            result = tool.invoke({"slide_path": slide_path})
            if not result.get("success", False):
                error_msg = result.get("error_message", "病理分析失败")
                return _finalize_return({
                    "messages": [AIMessage(content=f"病理分析失败：{error_msg}")],
                    "clinical_stage": "PathologyAnalysis_Error",
                    "error": error_msg,
                })

            prediction = result.get("prediction", "UNKNOWN")
            tumor_prob = result.get("tumor_probability", "N/A")
            confidence = result.get("confidence", "N/A")

            report_text = (
                "🧪 **病理切片AI分析报告**\n\n"
                f"- 切片路径: {slide_path}\n"
                f"- 预测结果: {prediction}\n"
                f"- 肿瘤概率: {tumor_prob}\n"
                f"- 置信度: {confidence}\n"
            )

            pathology_card = _build_pathology_card_base()
            pathology_card["data"].update({
                "analysis_mode": "quick" if mode == "quick" else "full",
                "slide_path": slide_path,
                "prediction": prediction,
                "tumor_probability": tumor_prob,
                "confidence": confidence,
                "heatmap_path": result.get("heatmap_path"),
                "topk_patches_dir": result.get("topk_patches_dir"),
                "report_path": result.get("report_path"),
                "raw_result": result,
            })

            return _finalize_return({
                "messages": [AIMessage(
                    content=report_text,
                    additional_kwargs={"pathology_card": pathology_card},
                )],
                "findings": {
                    "pathology_report": {
                        "analysis_mode": "quick" if mode == "quick" else "full",
                        "slide_path": slide_path,
                        "prediction": prediction,
                        "tumor_probability": tumor_prob,
                        "confidence": confidence,
                        "raw_result": result,
                    },
                    "pathology_card": pathology_card,
                },
                "clinical_stage": "PathologyAnalysis_Completed",
                "error": None,
            })

        # --- 患者ID综合分析 ---
        tool = _find_tool_by_name(tools, "perform_comprehensive_pathology_analysis")
        if not tool:
            error_msg = "系统错误：未找到综合病理分析工具。"
            return _finalize_return({
                "messages": [AIMessage(content=error_msg)],
                "clinical_stage": "PathologyAnalysis_Error",
                "error": "Comprehensive pathology analysis tool not found",
            })

        if show_thinking:
            print(f"🔬 [PathAgent] 开始对患者 {patient_id} 进行病理综合分析...")

        result = tool.invoke({"patient_id": patient_id})
        if not result.get("success", False):
            error_msg = result.get("error_message", "病理综合分析失败")
            return _finalize_return({
                "messages": [AIMessage(content=f"病理综合分析失败：{error_msg}")],
                "clinical_stage": "PathologyAnalysis_Error",
                "error": error_msg,
            })

        report_text = (
            "🧪 **病理综合分析报告**\n\n"
            f"- 患者编号: {result.get('patient_id', patient_id)}\n"
            f"- 分析切片数: {result.get('slides_analyzed')}\n"
            f"- 肿瘤切片数: {result.get('tumor_slides')}\n"
            f"- 正常切片数: {result.get('normal_slides')}\n"
            f"- 总体判断: {result.get('overall_diagnosis')}\n"
        )

        pathology_card = _build_pathology_card_base()
        pathology_card["data"].update({
            "analysis_mode": "comprehensive",
            "patient_id": result.get("patient_id", patient_id),
            "slides_analyzed": result.get("slides_analyzed"),
            "tumor_slides": result.get("tumor_slides"),
            "normal_slides": result.get("normal_slides"),
            "overall_diagnosis": result.get("overall_diagnosis"),
            "results": result.get("results", []),
            "raw_result": result,
        })

        return _finalize_return({
            "messages": [AIMessage(
                content=report_text,
                additional_kwargs={"pathology_card": pathology_card},
            )],
            "findings": {
                "pathology_report": {
                    "analysis_mode": "comprehensive",
                    "patient_id": result.get("patient_id", patient_id),
                    "slides_analyzed": result.get("slides_analyzed"),
                    "tumor_slides": result.get("tumor_slides"),
                    "normal_slides": result.get("normal_slides"),
                    "overall_diagnosis": result.get("overall_diagnosis"),
                    "raw_result": result,
                },
                "pathology_card": pathology_card,
            },
            "current_patient_id": patient_id,
            "clinical_stage": "PathologyAnalysis_Completed",
            "error": None,
        })

    return _run
