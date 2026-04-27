import json
import openpyxl
import os
import re
from ..services.virtual_database_service import CLASSIFICATION_FILE
from ..services.case_excel_service import find_case_record, upsert_case_record
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import Runnable
from ..state import CRCAgentState
from .node_utils import _ensure_message
from ..tools.database_tools import get_patient_case_info, upsert_patient_info

xlsx = str(CLASSIFICATION_FILE)

# ==============================================================================
# 1. Define Tools from chat_main0119.py
# ==============================================================================

@tool
def add(a: int, b: int) -> int:
    """加法函数，返回两个整数的和"""
    return a + b + 3

@tool
def subtract(a: int, b: int) -> int:
    """减法函数，返回两个整数的差"""
    return a - b - 3

@tool
def analyze_ct_image(ct_image_path: str, body_part: str = "abdomen") -> str:
    """分析CT图像并返回结果
    参数:
    - ct_image_path: CT图像路径
    - body_part: 扫描部位，默认腹部
    返回:
    - 包含肿瘤大小、位置、密度等信息的字符串
    """
    result = {
        "tumor_detected": True,
        "tumor_size": "3.5cm",
        "tumor_location": "升结肠",
        "density": "不均匀",
        "lymph_nodes": "多发肿大",
        "metastasis": "无远处转移",
        "stage": "T3N1M0"
    }
    return str(result)

@tool
def analyze_colonoscopy_report(report_content: str) -> str:
    """分析肠镜报告并返回结果
    参数:
    - report_content: 肠镜报告内容
    返回:
    - 包含病变位置、形态、病理类型等信息的字符串
    """
    result = {
        "lesion_location": "乙状结肠",
        "lesion_morphology": "溃疡型",
        "biopsy_result": "腺癌",
        "polyp_detected": False,
        "recommendation": "建议进一步检查和治疗"
    }
    return str(result)

@tool
def predict_msi_from_wsi(wsi_image_path: str) -> str:
    """从WSI图像预测微卫星不稳定状态
    参数:
    - wsi_image_path: 全玻片图像路径
    返回:
    - 包含MSI状态、置信度等信息的字符串
    """
    result = {
        "msi_status": "MSI-H",
        "confidence": "0.92",
        "mismatch_repair": {
            "MLH1": "缺失",
            "MSH2": "正常",
            "MSH6": "正常",
            "PMS2": "缺失"
        },
        "recommendation": "建议进行基因检测确认"
    }
    return str(result)

@tool
def sync_to_excel(json_data: str, excel_path: str = xlsx) -> str:
    """将 JSON 病例信息同步写入 Excel 病例表。

    参数:
    - json_data: JSON 字符串，内容应为病例字段字典。
    - excel_path: Excel 文件路径，默认使用 classification.xlsx。

    返回:
    - 写入结果说明。
    """
    try:
        data = json.loads(json_data)
        if not isinstance(data, dict):
            return "JSON payload must be an object"
        upsert_case_record(excel_path, data)
        return f"已同步到 {excel_path}"
    except PermissionError:
        backup_path = f"backup_{excel_path}"
        upsert_case_record(backup_path, data)
        return f"Excel 被占用，已写入备份文件 {backup_path}"
    except Exception as e:
        return f"写入 Excel 失败: {str(e)}"

@tool
def query_subject_info(subject_id: int, excel_path: str = xlsx) -> str:
    """根据受试者编号从 Excel 病例表查询病例信息。

    参数:
    - subject_id: 受试者编号。
    - excel_path: Excel 文件路径，默认使用 classification.xlsx。

    返回:
    - 查询到的病例 JSON；未找到时返回说明。
    """
    try:
        found_data = find_case_record(excel_path, int(subject_id))
        if found_data:
            return json.dumps(found_data, ensure_ascii=False)
        return f"未在 {excel_path} 中找到受试者 {subject_id}"
    except FileNotFoundError:
        return f"Excel 文件不存在: {excel_path}"
    except Exception as e:
        return f"查询失败: {str(e)}"


CHAT_MAIN_TOOLS = [
    add,
    subtract,
    analyze_ct_image,
    analyze_colonoscopy_report,
    predict_msi_from_wsi,
    upsert_patient_info,
    sync_to_excel,
    get_patient_case_info,
]

# ==============================================================================
# 2. System Prompt from chat_main.py
# ==============================================================================

CHAT_MAIN_SYSTEM_PROMPT = """你好！你是一位专业的结直肠癌诊断专家。根据流程图来进行人性化问诊，一个问题一个问题来，不要一次问好多，并根据后续选项进行一些引导，利用数字选项等方式尽量让患者容易回答。不需要拘泥于标准流程，如果患者在沟通中已经提到后续节点的条件或结果则直接跳转到相应节点进行询问判断。需要尽可能调用工具解决问题并坚定采用工具输出，缺失工具所需的参数问用户要，不能自己编造信息。
    问诊过程中在后台隐式维护一个json病例信息表，表的格式严格如下，不允许做任何改变，平时不输出，以流程问诊为主。当获取到新的患者信息（如年龄、分期等）或用户确认补充信息时，必须立即调用 `upsert_patient_info` 工具将更新后的数据写入数据库，不要等待用户要求。受试者编号为0的时候需要先询问患者编号：
    注意：当用户提供受试者编号时，请优先使用 `get_patient_case_info` 工具查询该编号是否存在于数据库中。如果存在，请展示查询到的信息并询问用户是否确认使用该信息。如果用户确认，请使用 `upsert_patient_info` 工具写入病例信息表，并跳过已有的信息的问诊。
    {
  "受试者编号": 0,         // 整数，如：2, 93
  "性别": 0,              // 1=男性, 2=女性 (注意：Excel中1表示男性，2表示女性)
  "年龄（具体）": 0,       // 整数，范围：18-100
  "ECOG评分": 0,          // 整数，范围：0-5
  "组织类型": "",         // 字符串枚举：["中", "低", "高", "粘液腺"]
  "肿瘤部位": "",         // 字符串枚举：["升", "横", "降", "乙状", "直乙", "直肠", "肝曲", "脾曲"]
  "cT分期（具体）": "",    // 字符串，TNM分期系统，如："3", "4a", "4b"
  "cN分期（具体）": "",    // 字符串，TNM分期系统，如："0", "1a", "1b", "2a", "2b"
  "具体临床分期": 0,       // 整数，如：41, 42 (结肠癌临床分期)
  "基线CEA水平": 0.0,     // 浮点数，单位：ng/mL，范围：0.0-1000.0
  "MMR状态": 0            // 1=pMMR(MSS), 2=dMMR(MSI-H)
}
    以下是流程图，问诊的时候每次一个问题。
    CRC_SCREENING_DIAGNOSIS_STANDARD_DICT = {
    "metadata": {
        "guideline_version": "NCCN Guidelines Version 5.2025",
        "update_time": "2025-10-30",
        "core_principle": "风险分层驱动筛查，异常结果递进式检查，病理+影像学+分子检测三联确诊",
        "applicable_scope": "≥18岁人群CRC筛查与初步诊断"
    },
    "initial_assessment": {
        "node_id": "IA-001",
        "node_name": "初始临床评估",
        "input_fields": [
            {"field_name": "age", "data_type": "integer", "unit": "岁", "required": True},
            {"field_name": "family_history", "data_type": "boolean", "description": "一级亲属<60岁CRC/腺瘤史", "required": True},
            {"field_name": "genetic_syndrome", "data_type": "enum", "options": ["none", "lynch", "fap", "attenuated_fap"], "required": True},
            {"field_name": "inflammatory_bowel_disease", "data_type": "object", "properties": {"type": ["uc", "cd", "none"], "duration": "integer"}},
            {"field_name": "previous_history", "data_type": "enum", "options": ["none", "crc", "adenoma", "positive_screening"], "required": True},
            {"field_name": "symptoms", "data_type": "array", "items": ["hematochezia", "abdominal_pain", "weight_loss", "bowel_habit_change", "none"]},
            {"field_name": "previous_screening_result", "data_type": "enum", "options": ["negative", "positive", "unknown"], "required": True}
        ],
        "decision_logic": {
            "condition_high_risk": "family_history==True OR genetic_syndrome!=none OR (inflammatory_bowel_disease.type!=none AND inflammatory_bowel_disease.duration≥8) OR previous_history in ['crc', 'adenoma', 'positive_screening']",
            "condition_general_≥45y": "age≥45 AND NOT condition_high_risk",
            "condition_general_<45y": "age<45 AND NOT condition_high_risk"
        },
        "branching_output": {
            "high_risk": {"next_node_id": "SP-001", "action": "执行高危人群结肠镜筛查"},
            "general_≥45y": {"next_node_id": "SP-002", "action": "执行普通人群≥45岁基础筛查"},
            "general_<45y": {"next_node_id": "SP-003", "action": "执行普通人群<45岁常规监测"}
        }
    },
    "screening_protocols": [
        {
            "protocol_id": "SP-001",
            "protocol_name": "高危人群结肠镜筛查",
            "precondition": "满足初始评估高危判定条件",
            "execution_spec": {
                "preparation": [
                    {"item": "diet", "requirement": "检查前48小时低渣饮食"},
                    {"item": "bowel_cleaning", "requirement": "口服聚乙二醇泻药，肠道清洁度≥Boston 2级"}
                ],
                "examination": {
                    "method": "全结肠镜检查",
                    "core_content": "全结肠黏膜可视化评估，息肉内镜下切除，可疑病灶活检",
                    "alternative_method": "腹部CT结肠成像（CTC）",
                    "alternative_condition": "肠道准备不佳（Boston<2级）或患者不耐受结肠镜"
                },
                "quality_control": {"image_requirement": "每个结肠段≥3张清晰图像", "biopsy_requirement": "可疑病灶≥2块组织样本"}
            },
            "result_definition": {
                "normal": {"code": "R-001", "description": "无息肉或仅增生性息肉<5mm，无异常黏膜表现"},
                "adenomatous_polyp": {"code": "R-002", "description": "腺瘤性息肉（管状/绒毛状/管状绒毛状）"},
                "suspicious_tumor": {"code": "R-003", "description": "黏膜不规则增厚、溃疡、占位性病变，疑似恶性"}
            },
            "result_processing": {
                "R-001": {
                    "follow_up": {
                        "general_high_risk": {"frequency": "1年", "method": "结肠镜"},
                        "lynch_fap": {"frequency": "6-12个月", "method": "结肠镜+基因检测"}
                    },
                    "next_node_id": "FU-001",
                    "output": "进入常规随访流程"
                },
                "R-002": {
                    "immediate_action": {"code": "A-001", "description": "内镜下息肉完整切除+切缘评估"},
                    "mandatory_tests": [
                        {
                            "test_id": "T-001",
                            "test_name": "息肉病理检测",
                            "parameters": ["histological_grade", "invasion_depth", "vascular_invasion"],
                            "normal_range": {"histological_grade": "low", "invasion_depth": "pTis", "vascular_invasion": "negative"}
                        },
                        {
                            "test_id": "T-002",
                            "test_name": "MMR/MSI检测",
                            "parameters": ["MLH1", "MSH2", "MSH6", "PMS2", "MSI_status"],
                            "normal_range": {"MSI_status": "MSS", "MMR_status": "pMMR"}
                        },
                        {
                            "test_id": "T-003",
                            "test_name": "PIK3CA突变检测",
                            "applicable_condition": "息肉>2cm OR 高级别 OR 脉管侵犯 OR pT1浸润",
                            "normal_range": "wild_type"
                        }
                    ],
                    "risk_stratification": {
                        "low_risk": "息肉<1cm + 低级别 + 无脉管侵犯 → next_node_id: FU-001",
                        "medium_risk": "息肉1-2cm + 低级别 / 多个小息肉 → next_node_id: FU-002",
                        "high_risk": "息肉>2cm / 高级别 / 脉管侵犯 / pT1浸润 → next_node_id: ST-001"
                    }
                },
                "R-003": {
                    "immediate_action": {"code": "A-002", "description": "病灶活检+MMR/MSI检测"},
                    "next_node_id": "ST-001",
                    "output": "启动肿瘤分期检查"
                }
            }
        },
        {
            "protocol_id": "SP-002",
            "protocol_name": "普通人群≥45岁基础筛查",
            "precondition": "age≥45且非高危人群",
            "screening_options": [
                {
                    "option_id": "SO-001",
                    "method": "粪便免疫化学检测（FIT）",
                    "frequency": "1年/次",
                    "test_spec": {
                        "target": "粪便血红蛋白",
                        "normal_threshold": "<10μg/g",
                        "result_codes": ["FIT-neg", "FIT-pos", "FIT-borderline"]
                    },
                    "result_processing": {
                        "FIT-neg": {"next_node_id": "FU-001", "follow_up": "每年复查FIT，每10年结肠镜确认"},
                        "FIT-pos": {"next_node_id": "SP-001", "action": "转诊结肠镜检查"},
                        "FIT-borderline": {"next_node_id": "RE-001", "action": "2-4周后复查FIT"}
                    }
                },
                {
                    "option_id": "SO-002",
                    "method": "多靶点粪便DNA检测",
                    "frequency": "3年/次",
                    "test_spec": {
                        "targets": ["KRAS突变", "粪便隐血", "甲基化基因"],
                        "normal_result": "无异常突变+隐血阴性",
                        "result_codes": ["DNA-neg", "DNA-pos", "DNA-uncertain"]
                    },
                    "result_processing": {
                        "DNA-neg": {"next_node_id": "FU-001", "follow_up": "3年复查，期间可每年联合FIT"},
                        "DNA-pos": {"next_node_id": "SP-001", "action": "转诊结肠镜检查"},
                        "DNA-uncertain": {"next_node_id": "RE-002", "action": "1个月内复查DNA+FIT"}
                    }
                },
                {
                    "option_id": "SO-003",
                    "method": "结肠镜检查",
                    "frequency": "10年/次",
                    "result_processing": "直接复用SP-001的result_processing逻辑"
                }
            ]
        },
        {
            "protocol_id": "SP-003",
            "protocol_name": "普通人群<45岁常规监测",
            "precondition": "age<45且非高危人群",
            "monitoring_items": [
                {
                    "item_id": "MI-001",
                    "item_name": "粪便隐血试验",
                    "frequency": "1年/次",
                    "normal_result": "阴性"
                },
                {
                    "item_id": "MI-002",
                    "item_name": "CEA检测",
                    "frequency": "1年/次",
                    "test_spec": {"target": "癌胚抗原", "normal_threshold": "<5ng/mL", "result_codes": ["CEA-normal", "CEA-elevated"]}
                }
            ],
            "result_processing": {
                "both_normal": {"next_node_id": "FU-001", "follow_up": "每年复查，≥45岁转入SP-002"},
                "CEA-elevated": {
                    "immediate_action": {"code": "A-003", "description": "腹部超声+2-4周复查CEA"},
                    "branching": {
                        "CEA_normalized": {"next_node_id": "FU-001"},
                        "CEA_persistently_elevated": {"next_node_id": "SP-001", "action": "胸腹盆CT+结肠镜"}
                    }
                },
                "fecal_occult_blood_positive": {"next_node_id": "SP-001", "action": "转诊结肠镜检查"}
            }
        }
    ],
    "staging_workup": {
        "node_id": "ST-001",
        "node_name": "肿瘤分期检查",
        "trigger_conditions": ["SP-001.R-003", "SP-001.R-002.high_risk", "SP-003.CEA_persistently_elevated"],
        "mandatory_examinations": [
            {
                "exam_id": "E-001",
                "exam_name": "胸腹盆增强CT",
                "purpose": "评估T/N/M分期",
                "parameters": ["tumor_invasion", "lymph_node_status", "distant_metastasis"],
                "normal_result": "无明确肿瘤浸润、淋巴结无肿大、无远处转移"
            },
            {
                "exam_id": "E-002",
                "exam_name": "CEA基线检测",
                "purpose": "疗效监测基线",
                "normal_threshold": "<5ng/mL"
            },
            {
                "exam_id": "E-003",
                "exam_name": "分子分型检测",
                "parameters": ["KRAS", "NRAS", "BRAF", "HER2", "POLE", "POLD1"],
                "extended_parameters": ["RET", "NTRK"] if "metastatic_suspected" else None,
                "normal_range": "wild_type for all"
            }
        ],
        "optional_examination": {
            "exam_id": "E-004",
            "exam_name": "FDG-PET/CT",
            "indication": "CT结果不确定、疑似微小转移、鉴别良恶性病灶",
            "contraindication": "常规分期检查"
        },
        "output": {
            "data_fields": ["tnm_stage", "molecular_subtype", "pathologic_diagnosis"],
            "next_node_id": "DI-001"
        }
    },
    "diagnostic_conclusion": {
        "node_id": "DI-001",
        "node_name": "最终诊断",
        "required_evidence": [
            {"type": "pathologic", "source": "活检/息肉切除样本"},
            {"type": "imaging", "source": "胸腹盆CT"},
            {"type": "molecular", "source": "MMR/MSI+KRAS/NRAS/BRAF检测"}
        ],
        "diagnostic_output": {
            "core_fields": [
                {"field_name": "pathologic_diagnosis", "data_type": "string"},
                {"field_name": "tnm_stage", "data_type": "string", "format": "AJCC 8th edition"},
                {"field_name": "molecular_subtype", "data_type": "object", "properties": ["mmr_status", "msi_status", "gene_mutations"]},
                {"field_name": "resectability", "data_type": "enum", "options": ["resectable", "unresectable", "potentially_resectable"]}
            ]
        },
        "closure_condition": "三类型证据齐全，明确分期与分型",
        "next_action": "衔接CRC治疗方案"
    },
    "follow_up": [
        {
            "follow_up_id": "FU-001",
            "name": "低危人群随访",
            "frequency": "1年",
            "items": ["对应筛查方法", "CEA检测"],
            "closure_condition": "连续3次正常→延长至3年随访"
        },
        {
            "follow_up_id": "FU-002",
            "name": "中危人群随访",
            "frequency": "6个月",
            "items": ["结肠镜", "CEA检测"],
            "closure_condition": "连续2次正常→转为1年随访"
        }
    ],
    "re-examination": [
        {
            "re_exam_id": "RE-001",
            "name": "FIT复查",
            "frequency": "2-4周",
            "trigger_condition": "SP-002.SO-001.FIT-borderline"
        },
        {
            "re_exam_id": "RE-002",
            "name": "粪便DNA+FIT联合复查",
            "frequency": "1个月",
            "trigger_condition": "SP-002.SO-002.DNA-uncertain"
        }
    ],
    "clinical_adjustments": {
        "flexible_rules": [
            {"rule_id": "FR-001", "description": "粪便筛查方案可按患者意愿切换", "constraint": "需符合筛查频率要求"},
            {"rule_id": "FR-002", "description": "结肠镜准备不佳时可用CTC替代", "constraint": "后续需补做结肠镜确认"},
            {"rule_id": "FR-003", "description": "高龄（≥80岁）/基础疾病多患者可简化筛查", "allowed_methods": ["FIT", "CEA检测"]}
        ],
        "forbidden_actions": [
            {"action_id": "FA-001", "description": "常规筛查不推荐FDG-PET/CT"},
            {"action_id": "FA-002", "description": "<45岁非高危人群不常规做结肠镜"},
            {"action_id": "FA-003", "description": "息肉切除后无需常规做全身PET/CT"}
        ]
    }
}
"""

# ==============================================================================
# 3. Context Compression Function from chat_main0119.py
# ==============================================================================

def compress_context(messages, max_length=30):
    """
    压缩上下文，保留系统提示和最近的消息，并严格保证工具调用链的完整性。
    """
    if len(messages) <= max_length:
        return messages

    system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
    other_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]

    kept_messages = []
    current_length = 0
    limit = max_length - len(system_messages)

    i = len(other_messages) - 1
    while i >= 0:
        msg = other_messages[i]

        block = []

        if isinstance(msg, ToolMessage):
            tool_msgs = []
            while i >= 0 and isinstance(other_messages[i], ToolMessage):
                tool_msgs.insert(0, other_messages[i])
                i -= 1

            if i >= 0:
                ai_msg = other_messages[i]
                if isinstance(ai_msg, AIMessage) and ai_msg.tool_calls:
                    block = [ai_msg] + tool_msgs
                    i -= 1
                else:
                    block = []
            else:
                block = []

        elif isinstance(msg, AIMessage) and msg.tool_calls:
             block = []
             i -= 1

        else:
            block = [msg]
            i -= 1

        if block:
            if current_length + len(block) <= limit:
                for m in reversed(block):
                    kept_messages.append(m)
                current_length += len(block)
            else:
                break

    return system_messages + list(reversed(kept_messages))

# ==============================================================================
# 4. Node Implementation
# ==============================================================================

def node_chat_main(model, tools=None, streaming: bool = True, show_thinking: bool = True) -> Runnable:
    """
    Chat Main Node - Implements the streaming dialog logic from chat_main0119.py
    It uses a specific set of tools and system prompt, while maintaining compatibility with the project.
    """
    tools = CHAT_MAIN_TOOLS
    llm_with_tools = model.bind_tools(tools)
    available_tools = {tool.name: tool for tool in tools}

    def _run(state: CRCAgentState):
        messages = state.messages
        findings = state.findings or {}
        active_inquiry = findings.get("active_inquiry", False)
        inquiry_message = findings.get("inquiry_message", "")
        missing = state.missing_critical_data or []

        # 合成模式：影像查询/数据库查询等非问诊场景下，跳过患者问诊流程直接进行 LLM 合成
        _NON_INTERVIEW_INTENTS = {
            "imaging_query", "case_database_query",
            "imaging_analysis", "knowledge_query",
        }
        _current_intent = findings.get("user_intent", "")
        is_synthesis_mode = (
            _current_intent in _NON_INTERVIEW_INTENTS
            or bool(
                state.current_plan
                and all(s.status == "completed" for s in state.current_plan)
            )
        )

        if not messages:
            default_message = "您好！我是您的结直肠癌诊断专家。根据流程图来进行问诊，每次一个问题并根据后续选项进行一些引导，利用数字选项等方式尽量让患者容易回答。请提供您的受试者编号，以便我能为您提供更准确的诊疗建议。"
            return {
                "messages": [AIMessage(content=default_message)],
                "clinical_stage": "ChatMain_Active",
                "findings": findings,
            }

        human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]

        if not human_messages:
            prompt_message = "请提供您的受试者编号，以便我能为您提供更准确的诊疗建议。"
            return {
                "messages": [AIMessage(content=prompt_message)],
                "clinical_stage": "ChatMain_Active",
                "findings": findings,
            }

        user_input = human_messages[-1]
        user_text = user_input.content or ""
        pending_data = findings.get("pending_patient_data")
        pending_id = findings.get("pending_patient_id")
        patient_record = dict(findings.get("patient_record") or {})
        record_template = {
            "受试者编号": 0,
            "性别": 0,
            "年龄（具体）": 0,
            "ECOG评分": 0,
            "组织类型": "",
            "肿瘤部位": "",
            "cT分期（具体）": "",
            "cN分期（具体）": "",
            "具体临床分期": 0,
            "基线CEA水平": 0.0,
            "MMR状态": 0,
        }
        for key, value in record_template.items():
            patient_record.setdefault(key, value)
        if pending_data:
            confirm_keywords = ["确认", "是的", "可以", "使用", "好的", "好", "对", "行", "没问题", "没错", "同意", "确定"]
            deny_keywords = ["不", "否", "不是", "不用", "不需要", "不对", "不要", "拒绝", "取消"]
            if any(k in user_text for k in confirm_keywords) and not any(k in user_text for k in deny_keywords):
                tool_func = available_tools.get("upsert_patient_info")
                if tool_func:
                    try:
                        result = tool_func.invoke({"json_data": json.dumps(pending_data, ensure_ascii=False)})
                    except Exception as e:
                        result = {"error": f"写入失败: {str(e)}"}
                else:
                    result = {"error": "未找到写入工具"}
                message = result.get("message") if isinstance(result, dict) else str(result)
                normalized_id = None
                if pending_id is not None:
                    try:
                        normalized_id = str(int(pending_id)).zfill(3)
                    except Exception:
                        normalized_id = str(pending_id)
                if normalized_id:
                    message = f"{message}\n\n已记录患者ID：{normalized_id}"
                findings_update = dict(findings)
                findings_update.pop("pending_patient_data", None)
                findings_update.pop("pending_patient_id", None)
                findings_update["patient_record"] = pending_data
                if normalized_id:
                    findings_update["current_patient_id"] = normalized_id
                return {
                    "messages": [AIMessage(content=message)],
                    "clinical_stage": "ChatMain_Active",
                    "findings": findings_update,
                    "current_patient_id": normalized_id,
                }
            if any(k in user_text for k in deny_keywords):
                findings_update = dict(findings)
                findings_update.pop("pending_patient_data", None)
                findings_update.pop("pending_patient_id", None)
                return {
                    "messages": [AIMessage(content="已取消使用该信息，请提供正确的患者信息。")],
                    "clinical_stage": "ChatMain_Active",
                    "findings": findings_update,
                }
        active_field = findings.get("active_field")
        if active_field:
            value = None
            if active_field == "受试者编号":
                m = re.search(r"\b(\d{1,3})\b", user_text)
                if m:
                    value = int(m.group(1))
            elif active_field == "性别":
                if "男" in user_text:
                    value = 1
                if "女" in user_text:
                    value = 2
            elif active_field == "年龄（具体）":
                m = re.search(r"(\d{1,3})", user_text)
                if m:
                    value = int(m.group(1))
            elif active_field == "ECOG评分":
                m = re.search(r"(\d)", user_text)
                if m:
                    value = int(m.group(1))
            elif active_field == "组织类型":
                if "粘液" in user_text:
                    value = "粘液腺"
                elif "低" in user_text:
                    value = "低"
                elif "高" in user_text:
                    value = "高"
                elif "中" in user_text:
                    value = "中"
            elif active_field == "肿瘤部位":
                location_map = [
                    ("升", "升"),
                    ("横", "横"),
                    ("降", "降"),
                    ("乙状", "乙状"),
                    ("直乙", "直乙"),
                    ("直肠", "直肠"),
                    ("肝曲", "肝曲"),
                    ("脾曲", "脾曲"),
                ]
                for key, label in location_map:
                    if key in user_text:
                        value = label
                        break
            elif active_field == "cT分期（具体）":
                m = re.search(r"\bT([0-4][a-c]?)\b", user_text, re.IGNORECASE)
                if m:
                    value = m.group(1)
            elif active_field == "cN分期（具体）":
                m = re.search(r"\bN([0-3][a-c]?)\b", user_text, re.IGNORECASE)
                if m:
                    value = m.group(1)
            elif active_field == "具体临床分期":
                m = re.search(r"(\d{1,2})", user_text)
                if m:
                    value = int(m.group(1))
            elif active_field == "基线CEA水平":
                m = re.search(r"(\d+(?:\.\d+)?)", user_text)
                if m:
                    value = float(m.group(1))
            elif active_field == "MMR状态":
                text_upper = user_text.upper()
                if "PMMR" in text_upper or "MSS" in text_upper or "稳定" in user_text:
                    value = 1
                if "DMMR" in text_upper or "MSI" in text_upper or "不稳定" in user_text:
                    value = 2
            if value is not None:
                patient_record[active_field] = value

                # [Auto-Save] 当捕获到有效值时，自动调用工具保存到数据库
                if patient_record.get("受试者编号") and patient_record.get("受试者编号") != 0:
                    tool_func = available_tools.get("upsert_patient_info")
                    if tool_func:
                        try:
                            # 构造符合工具要求的 JSON 数据
                            save_data = dict(patient_record)
                            tool_func.invoke({"json_data": json.dumps(save_data, ensure_ascii=False)})
                            if show_thinking:
                                print(f"[ChatMain] Auto-saved patient info: {active_field}={value}")
                        except Exception as e:
                            print(f"[ChatMain] Auto-save failed: {str(e)}")

                findings = dict(findings)
                findings.pop("active_field", None)
                findings["patient_record"] = patient_record
                findings["active_inquiry"] = False
                findings["inquiry_message"] = ""
            else:
                field_questions = {
                    "受试者编号": "请提供您的受试者编号（如 2 或 93）。",
                    "性别": "请问您的性别？1=男性，2=女性。",
                    "年龄（具体）": "请问您的年龄是多少岁？",
                    "ECOG评分": "请提供ECOG评分（0-5）。",
                    "组织类型": "请提供组织类型：中 / 低 / 高 / 粘液腺。",
                    "肿瘤部位": "请提供肿瘤部位：升 / 横 / 降 / 乙状 / 直乙 / 直肠 / 肝曲 / 脾曲。",
                    "cT分期（具体）": "请提供cT分期（如 T3、T4a）。",
                    "cN分期（具体）": "请提供cN分期（如 N1a、N2b）。",
                    "具体临床分期": "请提供具体临床分期（如 41、42）。",
                    "基线CEA水平": "请提供基线CEA水平（ng/mL）。",
                    "MMR状态": "请提供MMR状态：1=pMMR(MSS)，2=dMMR(MSI-H)。",
                }
                inquiry = field_questions.get(active_field, "请提供相关信息。")
                return {
                    "messages": [AIMessage(content=inquiry)],
                    "clinical_stage": "ChatMain_Active",
                    "findings": findings,
                }
        # 问诊流程：仅在非合成模式（即真正的患者问诊场景）下执行
        if not is_synthesis_mode:
            id_match = re.search(r"(?:患者|病人|病例|病历|编号|受试者编号|受试者|ID|id|我的id|我的ID)\s*(?:是|为)?\s*[:：#]?\s*(\d{1,3})", user_text)
            if id_match:
                subject_id = id_match.group(1)
                normalized_id = str(int(subject_id)).zfill(3)
                patient_record["受试者编号"] = int(subject_id)
                query_tool = available_tools.get("get_patient_case_info")
                if query_tool:
                    try:
                        result = query_tool.invoke({"patient_id": int(subject_id)})
                    except Exception as e:
                        result = {"error": f"\u67e5\u8be2\u5931\u8d25: {str(e)}", "patient_id": int(subject_id)}
                else:
                    result = {"error": "\u672a\u627e\u5230\u60a3\u8005\u67e5\u8be2\u5de5\u5177", "patient_id": int(subject_id)}
                display_result = result
                parsed = None
                if isinstance(result, dict):
                    display_result = json.dumps(result, ensure_ascii=False, indent=2)
                    if "error" not in result:
                        parsed = result
                else:
                    try:
                        parsed = json.loads(result)
                        display_result = json.dumps(parsed, ensure_ascii=False, indent=2)
                        if not isinstance(parsed, dict) or "error" in parsed:
                            parsed = None
                    except Exception:
                        parsed = None
                findings_update = dict(findings)
                findings_update["current_patient_id"] = normalized_id
                findings_update["patient_record"] = patient_record
                if isinstance(parsed, dict):
                    reply = (
                        f"\u5df2\u8bb0\u5f55\u60a3\u8005ID\uff1a{normalized_id}\u3002\n\n"
                        f"\u67e5\u8be2\u7ed3\u679c\uff1a\n{display_result}\n\n"
                        "\u8bf7\u786e\u8ba4\u662f\u5426\u4f7f\u7528\u8be5\u4fe1\u606f\u3002"
                    )
                    findings_update["pending_patient_data"] = parsed
                    findings_update["pending_patient_id"] = subject_id
                    return {
                        "messages": [AIMessage(content=reply)],
                        "clinical_stage": "ChatMain_Active",
                        "findings": findings_update,
                        "current_patient_id": normalized_id,
                    }
                findings = findings_update

            missing_fields = []
            field_order = [
                "受试者编号",
                "性别",
                "年龄（具体）",
                "ECOG评分",
                "组织类型",
                "肿瘤部位",
                "cT分期（具体）",
                "cN分期（具体）",
                "具体临床分期",
                "基线CEA水平",
                "MMR状态",
            ]
            for field in field_order:
                value = patient_record.get(field)
                if value in [0, "", None, 0.0]:
                    missing_fields.append(field)
            if missing_fields:
                field_questions = {
                    "受试者编号": "请提供您的受试者编号（如 2 或 93）。",
                    "性别": "请问您的性别？1=男性，2=女性。",
                    "年龄（具体）": "请问您的年龄是多少岁？",
                    "ECOG评分": "请提供ECOG评分（0-5）。",
                    "组织类型": "请提供组织类型：中 / 低 / 高 / 粘液腺。",
                    "肿瘤部位": "请提供肿瘤部位：升 / 横 / 降 / 乙状 / 直乙 / 直肠 / 肝曲 / 脾曲。",
                    "cT分期（具体）": "请提供cT分期（如 T3、T4a）。",
                    "cN分期（具体）": "请提供cN分期（如 N1a、N2b）。",
                    "具体临床分期": "请提供具体临床分期（如 41、42）。",
                    "基线CEA水平": "请提供基线CEA水平（ng/mL）。",
                    "MMR状态": "请提供MMR状态：1=pMMR(MSS)，2=dMMR(MSI-H)。",
                }
                next_field = missing_fields[0]
                inquiry = field_questions.get(next_field, "请提供相关信息。")
                for msg in messages:
                    if isinstance(msg, AIMessage) and inquiry in (msg.content or ""):
                        return {
                            "messages": [],
                            "clinical_stage": "ChatMain_Active",
                            "findings": findings,
                        }
                findings_update = dict(findings)
                findings_update["patient_record"] = patient_record
                findings_update["active_field"] = next_field
                findings_update["active_inquiry"] = True
                findings_update["inquiry_message"] = inquiry
                return {
                    "messages": [AIMessage(content=inquiry)],
                    "clinical_stage": "ChatMain_Active",
                    "findings": findings_update,
                    "current_patient_id": findings_update.get("current_patient_id"),
                }
        if show_thinking:
            print(f"[ChatMain] Received user input: {user_input.content[:100]}...")
            print(f"[ChatMain] Running with {len(tools)} tools...")

        input_messages = compress_context(
            [SystemMessage(content=CHAT_MAIN_SYSTEM_PROMPT)] + list(messages)
        )

        response = llm_with_tools.invoke(input_messages)

        # 只收集本轮新增消息（增量），不把历史消息混入返回值
        new_messages: list = []

        if response.tool_calls:
            new_messages.append(response)

            for tool_call in response.tool_calls:
                if hasattr(tool_call, 'name'):
                    tool_name = tool_call.name
                    tool_args = tool_call.args
                    tool_id = tool_call.id
                else:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]

                tool_func = available_tools.get(tool_name)
                if tool_func:
                    try:
                        result = tool_func.invoke(tool_args)
                    except Exception as e:
                        result = f"Error: {str(e)}"

                    tool_message = ToolMessage(
                        content=result,
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                    new_messages.append(tool_message)
                    if show_thinking:
                        print(f"执行 {tool_name} → {result}")
                else:
                    if show_thinking:
                        print(f"未找到工具：{tool_name}")

            # 第二次 LLM 调用：压缩后的完整历史 + 本轮新消息作为上下文
            second_call_context = compress_context(
                [SystemMessage(content=CHAT_MAIN_SYSTEM_PROMPT)] + list(messages) + new_messages
            )
            final_response = _ensure_message(llm_with_tools.invoke(second_call_context))
            new_messages.append(final_response)
        else:
            final_response = _ensure_message(response)
            new_messages.append(final_response)

        return {
            "messages": new_messages,
            "clinical_stage": "ChatMain_Active",
            "findings": findings,
        }

    return _run

