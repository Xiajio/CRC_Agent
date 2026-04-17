"""
影像医生智能体节点 (Radiology Agent)

本模块包含影像分析相关的节点，负责调度影像AI工具链：
- YOLO 肿瘤检测（快速筛查）
- U-Net 分割（精确分割）
- PyRadiomics 特征提取（1500维影像组学特征）
- LASSO 特征筛选（Top-20关键特征）
- 完整影像组学分析（一键执行完整流程）

版本：v2.0 - 完整版，支持多种分析模式
"""

import re
from typing import List, Optional, Literal
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool

from ..state import CRCAgentState
from .node_utils import (
    _latest_user_text,
    _ensure_message,
)
from .planner import get_current_pending_step, mark_step_completed


# 定义分析模式
AnalysisMode = Literal["detection", "segmentation", "radiomics", "comprehensive"]


def _detect_analysis_mode(user_text: str) -> AnalysisMode:
    """
    根据用户输入检测所需的分析模式
    
    Args:
        user_text: 用户输入文本
        
    Returns:
        分析模式：detection | segmentation | radiomics | comprehensive
    """
    user_text_lower = user_text.lower()
    
    # 完整影像组学分析
    comprehensive_keywords = [
        "完整分析", "全面分析", "影像组学", "radiomics", "comprehensive",
        "组学特征", "特征提取", "lasso", "深度分析"
    ]
    if any(kw in user_text_lower for kw in comprehensive_keywords):
        return "comprehensive"
    
    # U-Net 分割
    segmentation_keywords = [
        "分割", "segmentation", "unet", "u-net", "mask", "掩膜",
        "边界", "轮廓", "区域分割"
    ]
    if any(kw in user_text_lower for kw in segmentation_keywords):
        return "segmentation"
    
    # PyRadiomics 特征提取
    radiomics_keywords = [
        "特征", "feature", "纹理", "glcm", "形状", "灰度",
        "pyradiomics", "提取特征"
    ]
    if any(kw in user_text_lower for kw in radiomics_keywords):
        return "radiomics"
    
    # 默认：YOLO 肿瘤检测
    return "detection"


def _find_tool_by_name(tools: List[BaseTool], tool_name: str) -> Optional[BaseTool]:
    """根据工具名称查找工具"""
    for tool in tools:
        if hasattr(tool, 'name') and tool.name == tool_name:
            return tool
    return None


def node_rad_agent(tools: List[BaseTool], model, streaming: bool = False, show_thinking: bool = True) -> Runnable:
    """
    影像医生智能体节点（完整版 v2.0）
    
    职责：
    1. 解析用户的影像分析需求，识别分析模式
    2. 识别患者ID
    3. 根据分析模式调用对应的影像AI工具：
       - detection: YOLO 肿瘤检测（快速筛查）
       - segmentation: U-Net 肿瘤分割（精确分割）
       - radiomics: PyRadiomics 特征提取（1500维特征）
       - comprehensive: 完整影像组学分析（分割+提取+筛选）
    4. 生成专业的影像报告
    5. 将报告存入 state.findings["radiology_report"]
    
    支持的工具：
    - perform_comprehensive_tumor_check: YOLO 肿瘤检测
    - unet_segmentation_tool: U-Net 分割
    - radiomics_feature_extraction_tool: PyRadiomics 特征提取
    - lasso_feature_selection_tool: LASSO 特征筛选
    - comprehensive_radiomics_analysis: 完整影像组学分析
    
    Args:
        tools: 可用工具列表（包含 tumor_screening_tools 和 radiomics_tools）
        model: LLM 模型实例
        streaming: 是否启用流式输出
        show_thinking: 是否显示思考过程
    
    Returns:
        Runnable: 可执行的节点函数
    """
    
    def _run(state: CRCAgentState):
        user_text = _latest_user_text(state)
        
        if show_thinking:
            print(f"\n{'='*60}")
            print(f"👨\u200d⚕️ [RadAgent] 影像医生智能体启动 (v2.0)")
            print(f"{'='*60}")
            print(f"用户请求: {user_text[:100]}...")
        
        # 检测分析模式
        analysis_mode = _detect_analysis_mode(user_text)
        if show_thinking:
            mode_names = {
                "detection": "YOLO 肿瘤检测（快速筛查）",
                "segmentation": "U-Net 肿瘤分割（精确分割）",
                "radiomics": "PyRadiomics 特征提取",
                "comprehensive": "完整影像组学分析（分割+提取+筛选）"
            }
            print(f"📋 [RadAgent] 检测到分析模式: {mode_names.get(analysis_mode, analysis_mode)}")
        
        # [修复] 辅助函数：在返回前标记计划步骤为完成
        def _finalize_return(return_dict: dict) -> dict:
            """统一处理返回前的计划状态更新"""
            pending_step = get_current_pending_step(state)
            if pending_step:
                updated_plan = mark_step_completed(state, pending_step.id)
                return_dict["current_plan"] = updated_plan
                if show_thinking:
                    print(f"[RadAgent] 标记步骤完成: [{pending_step.id}] {pending_step.description}")
            return return_dict
        
        # === 步骤1：提取患者ID ===
        # 简化实现：从用户输入或 state 中提取患者ID
        patient_id = state.current_patient_id
        
        # 尝试从用户输入中提取数字（如 "093号"、"患者93"、"93号病人"）
        if not patient_id:
            # 使用多种模式匹配中文上下文中的患者ID
            # 模式1: "93号" 或 "093号" 格式
            match = re.search(r'(\d{1,3})号', user_text)
            if not match:
                # 模式2: "患者93" 或 "病人93" 格式
                match = re.search(r'(?:患者|病人|编号)\s*(\d{1,3})', user_text)
            if not match:
                # 模式3: 通用数字匹配（1-3位数字，不在其他数字中间）
                match = re.search(r'(?<!\d)(\d{1,3})(?!\d)', user_text)
            
            if match:
                patient_id = match.group(1).zfill(3)  # 补零到3位
                if show_thinking:
                    print(f"🔍 [RadAgent] 从用户输入中提取患者ID: {patient_id}")
        
        if not patient_id:
            # 无法识别患者ID，返回错误提示
            error_msg = (
                "抱歉，我需要知道具体的患者编号才能进行影像分析。\n"
                "请提供患者编号，例如：\n"
                "- '请分析093号患者的CT影像'\n"
                "- '帮我检测93号患者的肿瘤'"
            )
            return _finalize_return({
                "messages": [AIMessage(content=error_msg)],
                "clinical_stage": "RadiologyAnalysis_Error",
                "error": "Missing patient ID for imaging analysis"
            })
        
        # [优化] 检查是否已经执行过相同类型的分析（避免重复调用）
        findings = state.findings or {}
        existing_radiology_report = findings.get("radiology_report")

        # 检查是否需要重新分析：如果有报告但分析模式不匹配，则重新执行
        should_reanalyze = False
        if existing_radiology_report and existing_radiology_report.get("patient_id") == patient_id:
            existing_mode = existing_radiology_report.get("analysis_mode")
            # 如果当前是完整影像组学分析，但之前只是检测或分割，则需要重新分析
            if analysis_mode == "comprehensive" and existing_mode != "comprehensive":
                should_reanalyze = True
                if show_thinking:
                    print(f"🔄 [RadAgent] 检测到分析模式升级（{existing_mode} -> {analysis_mode}），重新执行分析")
            # 如果当前是分割，但之前只是检测，也需要重新分析
            elif analysis_mode == "segmentation" and existing_mode == "detection":
                should_reanalyze = True
                if show_thinking:
                    print(f"🔄 [RadAgent] 检测到分析模式升级（{existing_mode} -> {analysis_mode}），重新执行分析")

        if existing_radiology_report and existing_radiology_report.get("patient_id") == patient_id and not should_reanalyze:
            if show_thinking:
                print(f"✅ [RadAgent] 发现已有影像报告（患者 {patient_id}），跳过重复检测")
                print(f"   - 分析模式: {existing_radiology_report.get('analysis_mode', 'unknown')}")
                print(f"   - 检出率: {existing_radiology_report.get('detection_details', {}).get('tumor_detection_rate', 'N/A')}")
                print(f"   - 检测时间: {existing_radiology_report.get('timestamp', 'Unknown')}")

            # 提取关键数据
            has_tumor = existing_radiology_report.get("has_tumor", False)
            total_images = existing_radiology_report.get("total_images", 0) or existing_radiology_report.get("detection_details", {}).get("total_images", 0)
            images_with_tumor = existing_radiology_report.get("images_with_tumor", 0) or existing_radiology_report.get("detection_details", {}).get("images_with_tumor", 0)
            analyzed_images_count = existing_radiology_report.get("analyzed_images_count", 0)
            max_confidence = existing_radiology_report.get("detection_details", {}).get("max_confidence", 0)
            existing_mode = existing_radiology_report.get("analysis_mode", "unknown")
            top_features = existing_radiology_report.get("top_features", [])
            yolo_screening = existing_radiology_report.get("yolo_screening", {})
            summary = existing_radiology_report.get("summary", {})
            analyzed_images = existing_radiology_report.get("analyzed_images", [])

            # 使用大模型对之前的分析结果进行解读
            if show_thinking:
                print(f"🧠 [RadAgent] 使用大模型对已有分析结果进行解读...")

            # 构造分析结果摘要（用于大模型解读）
            analysis_summary = f"""
患者编号: {patient_id}
分析模式: {existing_mode}

## 模型分析结果：

### YOLO肿瘤检测结果：
- 总图像数: {total_images} 张
- 检测到肿瘤: {images_with_tumor} 张
- 最高置信度: {max_confidence:.4f}
- 置信度阈值: {yolo_screening.get('threshold', 0.5)}

"""

            if analyzed_images_count > 0:
                analysis_summary += f"""
### 分割和特征提取结果：
- 成功分析图像: {analyzed_images_count} 张
"""

                if analyzed_images:
                    analysis_summary += f"\n主要分析图像（前5张）:\n"
                    for i, img in enumerate(analyzed_images[:5], 1):
                        if img.get("success"):
                            yolo_conf = img.get("yolo_confidence", 0)
                            seg = img.get("segmentation", {})
                            analysis_summary += f"  {i}. {img['image_name']} (YOLO置信度: {yolo_conf:.2f}, 肿瘤占比: {seg.get('tumor_ratio', 'N/A')})\n"

                if top_features:
                    analysis_summary += f"\nTop-10 关键影像组学特征:\n"
                    for i, feat in enumerate(top_features[:10], 1):
                        analysis_summary += f"  {i}. {feat.get('feature_name', 'unknown')}: {feat.get('value', 0):.4f} (重要性: {feat.get('importance_score', 0):.4f})\n"

            # 使用大模型生成解读报告
            try:
                interpretation_prompt = f"""
你是一位专业的放射科医生和影像组学专家。请基于以下AI模型分析结果，为患者{patient_id}提供专业的CT影像解读报告。

{analysis_summary}

请提供：
1. 检测结果的专业解读
2. 影像组学特征的临床意义
3. 基于分析结果的临床建议
4. 需要关注的重点
5. 后续检查或随访建议

请用中文回答，格式清晰，专业准确。
"""

                # 调用大模型进行解读
                response = model.invoke([
                    HumanMessage(content=interpretation_prompt)
                ])

                report_text = response.content
                original_report_text = report_text or ""
                original_length = len(original_report_text)

                # [修复] 检测并处理重复报告：如果大模型返回了多份报告，只保留第一份
                # 通过检测重复的报告标题模式来判断
                import re
                # 查找所有报告标题（匹配多种可能的标题格式）
                report_patterns = [
                    # 优先匹配“标题式”的报告行，减少误判
                    r'^(?:#+\s*)?患者\d+号.*CT影像.*报告',
                    r'^(?:#+\s*)?患者\d+号.*影像.*报告',
                    r'^(?:#+\s*)?患者\d+号.*分析报告',
                    r'^\*\*患者\d+号'  # 匹配加粗的患者编号
                ]

                all_matches = []
                for pattern in report_patterns:
                    all_matches.extend(re.finditer(pattern, report_text, flags=re.MULTILINE))

                # 按出现位置排序
                all_matches.sort(key=lambda x: x.start())

                if len(all_matches) > 1:
                    # 发现多份报告，只保留第一份
                    if show_thinking:
                        print(f"⚠️ [RadAgent] 检测到大模型返回了 {len(all_matches)} 份报告，只保留第一份")

                    # 改进策略：找到报告的自然结束位置
                    first_match = all_matches[0]
                    second_match = all_matches[1]

                    # 从第一份报告的标题开始
                    report_start = first_match.start()

                    # 寻找第一份报告的结束位置：查找下一份报告标题之前的内容
                    # 或者查找常见的报告结束标记（如结论、建议等）
                    end_markers = [
                        "\n## 结论", "\n### 结论", "\n**结论**", "\n## 总结", "\n### 总结",
                        "\n**总结**", "\n## 临床建议", "\n### 临床建议", "\n**临床建议**",
                        "\n## 后续检查", "\n### 后续检查", "\n**后续检查**",
                        "\n---", "\n===", "\n免责声明", "\n报告医师"
                    ]

                    report_end = second_match.start() - 50  # 默认使用第二份报告前50个字符

                    # 查找更自然的结束位置
                    for marker in end_markers:
                        pos = report_text.find(marker, report_start, second_match.start())
                        if pos != -1:
                            report_end = pos + len(marker)
                            break

                    # 确保不截断得太短，至少保留第一份报告的开头部分
                    min_length = min(1000, len(report_text) // 3)  # 至少保留1/3的内容或1000字符
                    if report_end - report_start < min_length:
                        report_end = min(second_match.start() - 50, report_start + min_length)

                    # 如果结束位置异常（<= 起始），直接回退到保留第一份报告的开头
                    if report_end <= report_start:
                        report_end = min(report_start + min_length, len(report_text))

                    report_text = report_text[report_start:report_end].strip()

                    if show_thinking:
                        truncated_length = len(report_text)
                        print(f"[RadAgent] 报告截断详情:")
                        print(f"  - 原始长度: {original_length} 字符")
                        print(f"  - 截断后长度: {truncated_length} 字符")
                        if original_length:
                            print(f"  - 保留比例: {truncated_length / original_length:.1%}")

                    # 最终保护：如果截断后内容太短，保留更多内容
                    if len(report_text) < 200:
                        if show_thinking:
                            print(f"⚠️ [RadAgent] 截断后内容过短({len(report_text)}字符)，保留更多内容")
                        # 保留到第二份报告前200字符，至少保证有基本内容
                        safe_end = min(second_match.start() - 10, report_start + 800)
                        report_text = report_text[:safe_end - report_start].strip()

                if show_thinking:
                    print(f"✅ [RadAgent] 大模型解读完成")

            except Exception as e:
                if show_thinking:
                    print(f"⚠️ [RadAgent] 大模型解读失败，使用备用报告: {str(e)}")

                # 备用：使用已有的报告文本，如果没有则重新生成
                report_text = existing_radiology_report.get("ai_interpretation")
                if not report_text:
                    if has_tumor:
                        report_text = f"""
📊 **影像学AI分析报告**

**患者编号**: {patient_id}

**检测概况**:
- 总CT切片数: {total_images} 张
- 肿瘤阳性切片: {images_with_tumor} 张
- 最高置信度: {max_confidence:.4f} (AI检测置信度)

**初步判读** (AI辅助):
✓ **检测到疑似肿瘤病灶**
- 在 {images_with_tumor} 张CT切片中观察到疑似肿瘤征象
- AI模型置信度达到 {max_confidence:.2%}，建议进一步人工复核

**注意事项**:
本报告基于已存在的检测结果，仅供临床参考，最终诊断需结合病理学检查和临床医生判断。
"""
                    else:
                        report_text = f"""
📊 **影像学AI分析报告**

**患者编号**: {patient_id}

**检测概况**:
- 总CT切片数: {total_images} 张
- 肿瘤阳性切片: 0 张

**初步判读** (AI辅助):
✗ **未检测到明显肿瘤病灶**
- AI模型在所有 {total_images} 张CT切片中未发现疑似肿瘤征象

**注意事项**:
本报告基于已存在的检测结果，阴性结果不能完全排除病变可能。
"""

            # [修复] 构造影像组学报告卡片数据（用于前端渲染）
            radiomics_report_card = None
            if existing_mode == "comprehensive" or existing_mode == "segmentation" or existing_mode == "radiomics":
                # 对于包含影像组学分析的报告，构造卡片数据
                radiomics_report_card = {
                    "type": "radiomics_report_card",
                    "data": {
                        "patient_id": patient_id,
                        "total_images": total_images,
                        "images_with_tumor": images_with_tumor,
                        "analyzed_images_count": analyzed_images_count,
                        "has_tumor": has_tumor,
                        "analysis_mode": "完整影像组学分析" if existing_mode == "comprehensive" else f"{existing_mode}分析",
                        "timestamp": existing_radiology_report.get("timestamp", ""),
                        "report_file": existing_radiology_report.get("report_file", ""),
                        "yolo_screening": yolo_screening,
                        "top_features": top_features,
                        "summary": summary,
                        "analyzed_images": analyzed_images
                    }
                }
                if show_thinking:
                    print(f"📊 [RadAgent] 构造影像组学报告卡片数据用于前端渲染")

            if show_thinking:
                print(f"\n📄 [RadAgent] 返回大模型解读报告")
                print(f"{'='*60}\n")

            # 构建返回消息
            message_kwargs = {"content": report_text}
            if radiomics_report_card:
                message_kwargs["additional_kwargs"] = {"radiomics_report_card": radiomics_report_card}

            # 构建findings
            findings = {"has_tumor_imaging": has_tumor}
            if radiomics_report_card:
                findings["radiology_report"] = existing_radiology_report
                findings["radiomics_report_card"] = radiomics_report_card

            return _finalize_return({
                "messages": [AIMessage(**message_kwargs)],
                "findings": findings,
                "current_patient_id": patient_id,
                "clinical_stage": "RadiologyAnalysis_Completed",
                "error": None
            })
        
        # === 步骤2：根据分析模式调用对应工具 ===
        
        # 根据分析模式选择工具和执行逻辑
        if analysis_mode == "comprehensive":
            # 完整影像组学分析
            return _run_comprehensive_analysis(
                state, patient_id, tools, show_thinking, _finalize_return
            )
        elif analysis_mode == "segmentation":
            # U-Net 分割
            return _run_segmentation_analysis(
                state, patient_id, tools, show_thinking, _finalize_return
            )
        elif analysis_mode == "radiomics":
            # PyRadiomics 特征提取（需要先分割）
            return _run_radiomics_analysis(
                state, patient_id, tools, show_thinking, _finalize_return
            )
        else:
            # 默认：YOLO 肿瘤检测
            return _run_detection_analysis(
                state, patient_id, tools, show_thinking, _finalize_return
            )
        
    return _run


def _run_detection_analysis(state, patient_id, tools, show_thinking, _finalize_return):
    """
    执行 YOLO 肿瘤检测分析（快速筛查模式）
    """
    if show_thinking:
        print(f"🔬 [RadAgent] 开始对患者 {patient_id} 进行 YOLO 肿瘤检测...")
    
    # 查找 perform_comprehensive_tumor_check 工具
    tumor_check_tool = _find_tool_by_name(tools, "perform_comprehensive_tumor_check")
    
    if not tumor_check_tool:
        error_msg = "系统错误：未找到肿瘤检测工具 (perform_comprehensive_tumor_check)。请联系技术支持。"
        return _finalize_return({
            "messages": [AIMessage(content=error_msg)],
            "clinical_stage": "RadiologyAnalysis_Error",
            "error": "Tumor detection tool not found"
        })
    
    # 调用工具
    try:
        detection_result = tumor_check_tool.invoke({"patient_id": patient_id})
        
        if show_thinking:
            print(f"✅ [RadAgent] YOLO 肿瘤检测完成")
            print(f"   - 总影像数: {detection_result.get('total_images', 0)}")
            print(f"   - 检测到肿瘤: {detection_result.get('images_with_tumor', 0)} 张")
            print(f"   - 检出率: {detection_result.get('tumor_detection_rate', 'N/A')}")
    
    except Exception as e:
        error_msg = f"影像分析过程中发生错误：{str(e)}\n请检查患者编号是否正确，或稍后重试。"
        return _finalize_return({
            "messages": [AIMessage(content=error_msg)],
            "clinical_stage": "RadiologyAnalysis_Error",
            "error": str(e)
        })
    
    # 生成报告
    has_tumor = detection_result.get("has_tumor", False)
    total_images = detection_result.get("total_images", 0)
    images_with_tumor = detection_result.get("images_with_tumor", 0)
    tumor_rate = detection_result.get("tumor_detection_rate", "0%")
    max_confidence = detection_result.get("max_confidence")
    
    if has_tumor:
        report_text = f"""
📊 **影像学AI分析报告 - YOLO快速筛查**

**患者编号**: {patient_id}

**检测概况**:
- 总CT切片数: {total_images} 张
- 肿瘤阳性切片: {images_with_tumor} 张
- 肿瘤检出率: {tumor_rate}
- 最高置信度: {max_confidence:.4f} (AI检测置信度)

**初步判读** (AI辅助):
✓ **检测到疑似肿瘤病灶**
- 在 {images_with_tumor} 张CT切片中观察到疑似肿瘤征象
- AI模型置信度达到 {max_confidence:.2%}，建议进一步人工复核

**建议**:
1. 结合病理学检查进行综合诊断
2. 建议由放射科医生进行专业判读
3. 可考虑进行"完整影像组学分析"获取更详细的特征信息

**注意事项**:
本报告基于AI辅助分析，仅供临床参考，最终诊断需结合病理学检查和临床医生判断。
"""
    else:
        report_text = f"""
📊 **影像学AI分析报告 - YOLO快速筛查**

**患者编号**: {patient_id}

**检测概况**:
- 总CT切片数: {total_images} 张
- 肿瘤阳性切片: 0 张
- 肿瘤检出率: 0%

**初步判读** (AI辅助):
✗ **未检测到明显肿瘤病灶**
- AI模型在所有 {total_images} 张CT切片中未发现疑似肿瘤征象

**建议**:
1. 如临床症状与影像不符，建议增加其他影像学检查
2. 定期随访监测
3. AI辅助诊断存在局限性，建议由放射科医生进行专业判读

**注意事项**:
本报告基于AI辅助分析，仅供临床参考，阴性结果不能完全排除病变可能。
"""
    
    # 存储报告
    radiology_report = {
        "patient_id": patient_id,
        "report_type": "tumor_detection",
        "analysis_mode": "detection",
        "has_tumor": has_tumor,
        "detection_details": {
            "total_images": total_images,
            "images_with_tumor": images_with_tumor,
            "tumor_detection_rate": tumor_rate,
            "max_confidence": max_confidence
        },
        "ai_interpretation": report_text,
        "timestamp": detection_result.get("processing_timestamp"),
        "raw_result": detection_result
    }
    
    if show_thinking:
        print(f"\n📄 [RadAgent] 影像报告已生成")
        print(f"{'='*60}\n")
    
    # [修复] 构造肿瘤检测卡片数据，供前端渲染
    tumor_detection_card = {
        "type": "tumor_detection_card",
        "data": {
            "patient_id": patient_id,
            "total_images": total_images,
            "images_with_tumor": images_with_tumor,
            "images_without_tumor": total_images - images_with_tumor,
            "tumor_detection_rate": tumor_rate,
            "has_tumor": has_tumor,
            "max_confidence": max_confidence,
            "total_detections": detection_result.get("total_detections", 0),  # 从结果中获取检测数
            "sample_images_with_tumor": detection_result.get("sample_images_with_tumor", []),  # 从结果中获取样本图片
            "all_results": detection_result.get("all_results", []),  # 从结果中获取详细结果
            "confidence_threshold": 0.5,  # 默认阈值
            "analysis_mode": "YOLO快速筛查",
            "timestamp": detection_result.get("processing_timestamp", "")
        }
    }
    
    return _finalize_return({
        "messages": [AIMessage(
            content=report_text,
            additional_kwargs={"tumor_detection_card": tumor_detection_card}  # 将卡片数据放在 additional_kwargs 中
        )],
        "findings": {
            "radiology_report": radiology_report,
            "has_tumor_imaging": has_tumor,
            "tumor_detection_card": tumor_detection_card  # 同时保存到 findings 中
        },
        "current_patient_id": patient_id,
        "clinical_stage": "RadiologyAnalysis_Completed",
        "error": None
    })


def _run_segmentation_analysis(state, patient_id, tools, show_thinking, _finalize_return):
    """
    执行 U-Net 肿瘤分割分析
    """
    if show_thinking:
        print(f"🔬 [RadAgent] 开始对患者 {patient_id} 进行 U-Net 肿瘤分割...")
    
    # 查找 U-Net 分割工具
    unet_tool = _find_tool_by_name(tools, "unet_segmentation_tool")
    
    if not unet_tool:
        error_msg = (
            "系统错误：未找到 U-Net 分割工具。\n"
            "请确保已安装依赖：pip install torch opencv-python\n"
            "并确保模型文件存在。"
        )
        return _finalize_return({
            "messages": [AIMessage(content=error_msg)],
            "clinical_stage": "RadiologyAnalysis_Error",
            "error": "U-Net segmentation tool not found"
        })
    
    # 构建图像路径（使用正确的影像数据目录）
    import os
    from pathlib import Path
    base_path = str(Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / "data" / "Case Database" / "Radiographic Imaging" / patient_id)
    
    # 查找第一张图像进行分割演示
    if os.path.exists(base_path):
        images = [f for f in os.listdir(base_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            image_path = os.path.join(base_path, images[0])
        else:
            error_msg = f"患者 {patient_id} 目录下未找到有效的图像文件。"
            return _finalize_return({
                "messages": [AIMessage(content=error_msg)],
                "clinical_stage": "RadiologyAnalysis_Error",
                "error": "No images found"
            })
    else:
        error_msg = f"患者 {patient_id} 的影像目录不存在。"
        return _finalize_return({
            "messages": [AIMessage(content=error_msg)],
            "clinical_stage": "RadiologyAnalysis_Error",
            "error": f"Patient directory not found: {base_path}"
        })
    
    # 调用 U-Net 分割工具
    try:
        seg_result = unet_tool.invoke({"image_path": image_path})
        
        if not seg_result.get("success"):
            error_msg = f"分割失败: {seg_result.get('error_message')}"
            return _finalize_return({
                "messages": [AIMessage(content=error_msg)],
                "clinical_stage": "RadiologyAnalysis_Error",
                "error": seg_result.get('error_message')
            })
        
        if show_thinking:
            print(f"✅ [RadAgent] U-Net 分割完成")
            print(f"   - 肿瘤区域: {seg_result.get('tumor_area', 0)} 像素")
            print(f"   - 肿瘤占比: {seg_result.get('tumor_ratio', '0%')}")
    
    except Exception as e:
        error_msg = f"分割过程中发生错误：{str(e)}"
        return _finalize_return({
            "messages": [AIMessage(content=error_msg)],
            "clinical_stage": "RadiologyAnalysis_Error",
            "error": str(e)
        })
    
    # 生成报告
    tumor_area = seg_result.get("tumor_area", 0)
    tumor_ratio = seg_result.get("tumor_ratio", "0%")
    has_tumor = tumor_area > 0
    
    report_text = f"""
📊 **影像学AI分析报告 - U-Net精确分割**

**患者编号**: {patient_id}

**分割结果**:
- 分析图像: {os.path.basename(image_path)}
- 肿瘤区域面积: {tumor_area} 像素
- 肿瘤区域占比: {tumor_ratio}
- 边界框: {seg_result.get('bounding_box', 'N/A')}

**初步判读** (AI辅助):
{"✓ **检测到肿瘤区域**" if has_tumor else "✗ **未检测到明显肿瘤区域**"}
- 掩膜文件已保存: {seg_result.get('mask_path', 'N/A')}
- 可视化叠加图: {seg_result.get('overlay_path', 'N/A')}

**后续分析建议**:
1. 可使用"影像组学分析"提取1500维特征
2. 使用"LASSO特征筛选"获取Top-20关键特征
3. 或使用"完整影像组学分析"一键执行全流程

**注意事项**:
本报告基于AI辅助分析，分割结果仅供参考，建议由放射科医生进行专业判读。
"""
    
    # 存储报告
    radiology_report = {
        "patient_id": patient_id,
        "report_type": "segmentation",
        "analysis_mode": "segmentation",
        "has_tumor": has_tumor,
        "segmentation_details": {
            "tumor_area": tumor_area,
            "tumor_ratio": tumor_ratio,
            "bounding_box": seg_result.get("bounding_box"),
            "mask_path": seg_result.get("mask_path"),
            "overlay_path": seg_result.get("overlay_path")
        },
        "ai_interpretation": report_text,
        "timestamp": seg_result.get("timestamp"),
        "raw_result": seg_result
    }
    
    if show_thinking:
        print(f"\n📄 [RadAgent] 分割报告已生成")
        print(f"{'='*60}\n")
    
    # [修复] 构造肿瘤检测卡片数据（虽然这是分割结果，但使用统一的卡片格式）
    tumor_detection_card = {
        "type": "tumor_detection_card",
        "data": {
            "patient_id": patient_id,
            "total_images": 1,  # U-Net 只处理单张图像
            "images_with_tumor": 1 if has_tumor else 0,
            "images_without_tumor": 0 if has_tumor else 1,
            "tumor_detection_rate": "100%" if has_tumor else "0%",
            "has_tumor": has_tumor,
            "max_confidence": 1.0 if has_tumor else 0.0,
            "total_detections": 1 if has_tumor else 0,
            "sample_images_with_tumor": [seg_result.get("image_path")] if has_tumor else [],
            "all_results": [{
                "image_name": os.path.basename(image_path),
                "image_path": image_path,
                "has_tumor": has_tumor,
                "confidence": 1.0 if has_tumor else 0.0,
                "total_detections": 1 if has_tumor else 0,
                "bounding_boxes": [seg_result.get("bounding_box")] if has_tumor else []
            }],
            "confidence_threshold": 0.5,
            "analysis_mode": "U-Net精确分割",
            "timestamp": seg_result.get("timestamp", "")
        }
    }
    
    return _finalize_return({
        "messages": [AIMessage(
            content=report_text,
            additional_kwargs={"tumor_detection_card": tumor_detection_card}
        )],
        "findings": {
            "radiology_report": radiology_report,
            "has_tumor_imaging": has_tumor,
            "segmentation_mask_path": seg_result.get("mask_path"),
            "tumor_detection_card": tumor_detection_card
        },
        "current_patient_id": patient_id,
        "clinical_stage": "RadiologyAnalysis_Completed",
        "error": None
    })


def _run_radiomics_analysis(state, patient_id, tools, show_thinking, _finalize_return):
    """
    执行 PyRadiomics 影像组学特征提取
    注：需要先有分割掩膜，如果没有则先执行分割
    """
    if show_thinking:
        print(f"🔬 [RadAgent] 开始对患者 {patient_id} 进行影像组学特征提取...")
    
    # 检查是否已有分割掩膜
    findings = state.findings or {}
    mask_path = findings.get("segmentation_mask_path")
    
    if not mask_path:
        if show_thinking:
            print(f"⚠️ [RadAgent] 未找到分割掩膜，需要先执行分割")
        # 提示用户先进行分割
        error_msg = (
            "影像组学特征提取需要先进行肿瘤分割。\n"
            "请先执行以下操作之一：\n"
            "1. 输入'分割'进行 U-Net 肿瘤分割\n"
            "2. 输入'完整影像组学分析'一键执行全流程"
        )
        return _finalize_return({
            "messages": [AIMessage(content=error_msg)],
            "clinical_stage": "RadiologyAnalysis_Pending",
            "error": "Segmentation mask required"
        })
    
    # 查找 PyRadiomics 工具
    radiomics_tool = _find_tool_by_name(tools, "radiomics_feature_extraction_tool")
    
    if not radiomics_tool:
        error_msg = (
            "系统错误：未找到 PyRadiomics 特征提取工具。\n"
            "请确保已安装依赖：pip install pyradiomics SimpleITK"
        )
        return _finalize_return({
            "messages": [AIMessage(content=error_msg)],
            "clinical_stage": "RadiologyAnalysis_Error",
            "error": "PyRadiomics tool not found"
        })
    
    # TODO: 获取原始图像路径（从 state 中）
    # 暂时返回提示信息
    error_msg = (
        "影像组学特征提取功能正在开发中。\n"
        "建议使用'完整影像组学分析'一键执行全流程。"
    )
    return _finalize_return({
        "messages": [AIMessage(content=error_msg)],
        "clinical_stage": "RadiologyAnalysis_Pending",
        "error": "Feature extraction in development"
    })


def _run_comprehensive_analysis(state, patient_id, tools, show_thinking, _finalize_return):
    """
    执行完整影像组学分析（分割 + 特征提取 + 特征筛选）
    """
    if show_thinking:
        print(f"🔬 [RadAgent] 开始对患者 {patient_id} 进行完整影像组学分析...")
        print(f"   流程: U-Net分割 → PyRadiomics特征提取 → LASSO特征筛选")
    
    # 查找完整分析工具
    comprehensive_tool = _find_tool_by_name(tools, "comprehensive_radiomics_analysis")
    
    if not comprehensive_tool:
        error_msg = (
            "系统错误：未找到完整影像组学分析工具。\n"
            "请确保已安装以下依赖：\n"
            "- pip install torch opencv-python\n"
            "- pip install pyradiomics SimpleITK\n"
            "- pip install scikit-learn"
        )
        return _finalize_return({
            "messages": [AIMessage(content=error_msg)],
            "clinical_stage": "RadiologyAnalysis_Error",
            "error": "Comprehensive analysis tool not found"
        })
    
    # 构建图像目录路径（使用正确的影像数据目录）
    import os
    from pathlib import Path
    patient_image_dir = str(Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / "data" / "Case Database" / "Radiographic Imaging" / patient_id)
    
    # 检查目录是否存在
    if not os.path.exists(patient_image_dir):
        error_msg = f"患者 {patient_id} 的影像目录不存在: {patient_image_dir}"
        return _finalize_return({
            "messages": [AIMessage(content=error_msg)],
            "clinical_stage": "RadiologyAnalysis_Error",
            "error": f"Patient directory not found"
        })
    
    # 检查目录下是否有图像文件
    images = [f for f in os.listdir(patient_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        error_msg = f"患者 {patient_id} 目录下未找到有效的图像文件。"
        return _finalize_return({
            "messages": [AIMessage(content=error_msg)],
            "clinical_stage": "RadiologyAnalysis_Error",
            "error": "No images found"
        })
    
    if show_thinking:
        print(f"   影像目录: {patient_image_dir}")
        print(f"   图像数量: {len(images)} 张")
        print(f"   流程: YOLO筛选 → U-Net分割 → PyRadiomics特征提取 → LASSO特征筛选")
    
    # 调用完整分析工具（传入目录路径，遍历所有图像）
    try:
        result = comprehensive_tool.invoke({
            "input_path": patient_image_dir,  # 传入目录路径，而不是单张图像
            "top_k_features": 20,
            "skip_yolo_screening": False,  # 启用YOLO筛选
            "yolo_confidence_threshold": 0.5
        })
        
        if not result.get("success"):
            error_msg = f"分析失败: {result.get('error_message')}"
            return _finalize_return({
                "messages": [AIMessage(content=error_msg)],
                "clinical_stage": "RadiologyAnalysis_Error",
                "error": result.get('error_message')
            })
        
        if show_thinking:
            print(f"✅ [RadAgent] 完整影像组学分析完成")
            summary = result.get("summary", {})
            print(f"   - 总图像数: {result.get('total_images', 0)}")
            print(f"   - YOLO检测到肿瘤: {result.get('images_with_tumor', 0)} 张")
            print(f"   - 成功分析: {result.get('images_analyzed', 0)} 张")
    
    except Exception as e:
        error_msg = f"完整分析过程中发生错误：{str(e)}"
        return _finalize_return({
            "messages": [AIMessage(content=error_msg)],
            "clinical_stage": "RadiologyAnalysis_Error",
            "error": str(e)
        })
    
    # 处理跳过的情况（所有图像均未检测到肿瘤）
    if result.get("skipped"):
        summary = result.get("summary", {})
        report_text = f"""
📊 **完整影像组学分析报告**

**患者编号**: {patient_id}

---

## 🔍 YOLO 肿瘤检测筛选
- 总图像数: {result.get('total_images', 0)} 张
- 检测到肿瘤: 0 张
- 跳过原因: {result.get('skip_reason', 'YOLO未检测到肿瘤')}

---

## 📋 分析总结
✗ **所有CT切片均未检测到肿瘤区域**
- YOLO模型在所有图像中未发现疑似肿瘤征象
- 后续的分割、特征提取步骤已跳过

## 💡 临床建议
1. AI检测结果为阴性，但不能完全排除病变可能
2. 建议结合临床症状和其他检查综合判断
3. 如有临床疑虑，建议由放射科医生人工判读

**注意事项**:
本报告基于AI辅助分析，仅供临床参考。
"""
        radiology_report = {
            "patient_id": patient_id,
            "report_type": "comprehensive_radiomics",
            "analysis_mode": "comprehensive",
            "has_tumor": False,
            "summary": summary,
            "yolo_screening": result.get("yolo_screening"),
            "ai_interpretation": report_text,
            "timestamp": summary.get("analysis_timestamp"),
            "raw_result": result
        }
        
        if show_thinking:
            print(f"\n📄 [RadAgent] 完整影像组学分析报告已生成（无肿瘤）")
            print(f"{'='*60}\n")
        
        # [修复] 构造肿瘤检测卡片数据
        tumor_detection_card = {
            "type": "tumor_detection_card",
            "data": {
                "patient_id": patient_id,
                "total_images": result.get('total_images', 0),
                "images_with_tumor": 0,
                "images_without_tumor": result.get('total_images', 0),
                "tumor_detection_rate": "0%",
                "has_tumor": False,
                "max_confidence": 0.0,
                "total_detections": 0,
                "sample_images_with_tumor": [],
                "all_results": [],
                "confidence_threshold": yolo_screening.get('threshold', 0.5),
                "analysis_mode": "完整影像组学分析",
                "timestamp": summary.get("analysis_timestamp", ""),
                "skip_reason": result.get('skip_reason', 'YOLO未检测到肿瘤')
            }
        }
        
        return _finalize_return({
            "messages": [AIMessage(
                content=report_text,
                additional_kwargs={"tumor_detection_card": tumor_detection_card}
            )],
            "findings": {
                "radiology_report": radiology_report,
                "has_tumor_imaging": False,
                "tumor_detection_card": tumor_detection_card
            },
            "current_patient_id": patient_id,
            "clinical_stage": "RadiologyAnalysis_Completed",
            "error": None
        })
    
    # 生成详细报告（有肿瘤图像）
    summary = result.get("summary", {})
    analyzed_images = result.get("analyzed_images", [])
    yolo_screening = result.get("yolo_screening", {})
    
    # 汇总所有分析的特征
    all_selected_features = []
    total_feature_count = 0
    for img_result in analyzed_images:
        if img_result.get("success"):
            fs = img_result.get("feature_selection", {})
            if fs.get("selected_features"):
                all_selected_features.extend(fs["selected_features"])
            rad = img_result.get("radiomics", {})
            total_feature_count = max(total_feature_count, rad.get("feature_count", 0))
    
    # 构建分析图像列表
    analyzed_images_text = ""
    for i, img in enumerate(analyzed_images[:5], 1):  # 只显示前5张
        if img.get("success"):
            yolo_conf = img.get("yolo_confidence", 0)
            seg = img.get("segmentation", {})
            analyzed_images_text += f"  {i}. {img['image_name']} (YOLO置信度: {yolo_conf:.2f}, 肿瘤占比: {seg.get('tumor_ratio', 'N/A')})\n"
    if len(analyzed_images) > 5:
        analyzed_images_text += f"  ... 共 {len(analyzed_images)} 张图像\n"
    
    # 构建 Top 特征列表（取前10个）
    top_features_text = ""
    unique_features = {}
    for feat in all_selected_features:
        name = feat.get('feature_name', '')
        if name not in unique_features:
            unique_features[name] = feat
    sorted_features = sorted(unique_features.values(), key=lambda x: x.get('importance_score', 0), reverse=True)[:10]
    for i, feat in enumerate(sorted_features, 1):
        top_features_text += f"  {i}. {feat['feature_name']}: {feat['value']:.4f}\n"
    if len(unique_features) > 10:
        top_features_text += f"  ... 共 {len(unique_features)} 个关键特征\n"
    
    report_text = f"""
📊 **完整影像组学分析报告**

**患者编号**: {patient_id}

---

## 🔍 Step 1: YOLO 肿瘤检测筛选
- 总图像数: {result.get('total_images', 0)} 张
- 检测到肿瘤: {result.get('images_with_tumor', 0)} 张
- YOLO置信度阈值: {yolo_screening.get('threshold', 0.5)}

## 🔬 Step 2: U-Net 肿瘤分割
- 成功分割: {result.get('images_analyzed', 0)} 张
- 分析的图像:
{analyzed_images_text}

## 📊 Step 3: PyRadiomics 特征提取
- 每张图像提取特征: {total_feature_count} 维
- 特征类别: 形状、一阶统计、GLCM、GLRLM、GLSZM、GLDM、NGTDM

## 🎯 Step 4: LASSO 特征筛选
- 每张图像筛选 Top-20 关键特征
- **汇总 Top-10 关键特征**:
{top_features_text}

---

## 📋 分析总结
- 肿瘤检测结果: ✓ 检测到肿瘤
- 分析图像数: {result.get('images_analyzed', 0)}/{result.get('total_images', 0)} 张
- 分析报告路径: {result.get('report_file', 'N/A')}

## 💡 临床建议
1. Top-20 影像组学特征可用于预后预测模型
2. 建议结合临床病理信息进行综合分析
3. 完整特征数据已保存，可用于后续机器学习建模

**注意事项**:
本报告基于AI辅助分析，仅供临床参考。影像组学特征需结合临床验证。
"""
    
    # 存储报告
    radiology_report = {
        "patient_id": patient_id,
        "report_type": "comprehensive_radiomics",
        "analysis_mode": "comprehensive",
        "has_tumor": result.get('images_with_tumor', 0) > 0,
        "summary": summary,
        "yolo_screening": yolo_screening,
        "analyzed_images_count": result.get('images_analyzed', 0),
        "total_images": result.get('total_images', 0),
        "top_features": sorted_features,
        "analyzed_images": analyzed_images,  # 添加 analyzed_images 字段
        "ai_interpretation": report_text,
        "timestamp": summary.get("analysis_timestamp"),
        "report_file": result.get("report_file"),
        "raw_result": result
    }
    
    # [修复] 构造影像组学报告卡片数据
    radiomics_report_card = {
        "type": "radiomics_report_card",
        "data": {
            "patient_id": patient_id,
            "total_images": result.get('total_images', 0),
            "images_with_tumor": result.get('images_with_tumor', 0),
            "analyzed_images_count": result.get('images_analyzed', 0),
            "has_tumor": result.get('images_with_tumor', 0) > 0,
            "analysis_mode": "完整影像组学分析",
            "timestamp": summary.get("analysis_timestamp", ""),
            "report_file": result.get('report_file', ""),
            "yolo_screening": yolo_screening,
            "top_features": sorted_features,
            "summary": summary,
            "analyzed_images": analyzed_images  # 添加 analyzed_images 字段
        }
    }
    
    if show_thinking:
        print(f"\n📄 [RadAgent] 完整影像组学分析报告已生成")
        print(f"{'='*60}\n")
    
    return _finalize_return({
        "messages": [AIMessage(
            content=report_text,
            additional_kwargs={"radiomics_report_card": radiomics_report_card}  # 使用 radiomics_report_card
        )],
        "findings": {
            "radiology_report": radiology_report,
            "has_tumor_imaging": result.get('images_with_tumor', 0) > 0,
            "radiomics_features": sorted_features,
            "radiomics_report_card": radiomics_report_card  # 同时保存到 findings
        },
        "current_patient_id": patient_id,
        "clinical_stage": "RadiologyAnalysis_Completed",
        "error": None
    })


# === 导出函数 ===
__all__ = [
    "node_rad_agent",
]
