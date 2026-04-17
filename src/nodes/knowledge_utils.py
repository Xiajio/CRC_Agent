
from ..state import CRCAgentState

def _get_patient_state_description(state: CRCAgentState) -> str:
    """
    [Context Injection] 获取患者当前关键状态描述
    用于强制注入到 Knowledge Query 中，防止上下文丢失。
    """
    if not state.current_patient_id:
        return ""
    
    status_parts = [f"**Current Patient State (ID: {state.current_patient_id})**:"]
    
    # 1. 诊断状态
    if state.pathology_confirmed:
        status_parts.append("- Diagnosis: Confirmed")
    else:
        status_parts.append("- Diagnosis: Pending/Unconfirmed")
        
    # 2. 分期状态
    profile = state.patient_profile
    tnm = profile.tnm_staging if profile else {}
    if tnm and isinstance(tnm, dict):
        tnm_str = " ".join([f"{k}{v}" for k, v in tnm.items() if v])
        status_parts.append(f"- Staging: {tnm_str}")
    
    # 3. 治疗阶段 (Critical Logic)
    has_plan = bool(state.decision_json)
    
    # 尝试从摘要中提取关键词
    summary = (state.summary_memory or "").lower()
    
    # 优先级逻辑
    if "术后" in summary or "post-op" in summary:
        phase = "Post-operative"
    elif "化疗中" in summary or "chemotherapy" in summary:
        phase = "During Treatment (Chemotherapy)"
    elif "未开始" in summary or "not started" in summary or "assessment" in summary:
        phase = "Pre-treatment / Assessment Phase"
    elif has_plan:
        phase = "Treatment Plan Proposed (Not yet started)"
    else:
        phase = "Pre-treatment / Diagnostic Phase"
        
    status_parts.append(f"- Treatment Phase: {phase}")
    
    # 4. 关键提示
    if "Pre-treatment" in phase or "Diagnostic" in phase:
        status_parts.append("  (Note: Patient has NOT started any chemotherapy or surgery yet. No drug interactions currently.)")
    elif "Post-operative" in phase:
        status_parts.append("  (Note: Patient is in post-operative recovery.)")
        
    return "\n".join(status_parts)
