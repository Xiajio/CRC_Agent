"""
数据库查询工具集
将虚拟数据库封装为 LangChain BaseTool 格式，供智能体调用
"""

import json
from typing import List, Dict, Optional, Any
from langchain_core.tools import tool
from ..services.case_excel_service import upsert_case_record

from ..services.virtual_database_service import (
    get_case_database,
    load_cases_from_database,
    query_cases,
    get_case_statistics,
    get_imaging_by_folder,
    get_imaging_by_patient_id,
    get_pathology_slides_by_patient_id,
    get_all_folder_names,
    CLASSIFICATION_FILE,
    PATHOLOGY_SLIDE_FOLDERS,
)

from ..tools.tumor_screening_tools import (
    perform_comprehensive_tumor_check,
    perform_comprehensive_tumor_check_as_tool
)


# ============================================================
# 原子化工具函数 (Atomic Tools)
# 使用 @tool 装饰器定义的轻量级工具，供 LLM 直接调用
# ============================================================


def _build_case_brief(case: Dict[str, Any]) -> str:
    """Build a short one-line patient summary for tool detail display."""
    cm_stage = case.get("cm_stage") or "0"
    return (
        f"病例 {case.get('patient_id', 'N/A')}: "
        f"{case.get('gender', 'N/A')}/{case.get('age', 'N/A')}岁, "
        f"{case.get('tumor_location', 'N/A')} {case.get('histology_type', 'N/A')}分化癌, "
        f"分期 cT{case.get('ct_stage', '')}N{case.get('cn_stage', '')}M{cm_stage}"
    )


def _count_pathology_slides(patient_id: str) -> int:
    """Count raw pathology slide files without triggering preview generation."""
    folder_name = str(patient_id).zfill(3)
    total = 0
    for base_path in PATHOLOGY_SLIDE_FOLDERS:
        patient_dir = base_path / folder_name
        if not patient_dir.exists():
            continue
        for ext in ("*.svs", "*.tif", "*.tiff", "*.ndpi", "*.mrxs", "*.vms", "*.vmu"):
            total += len(list(patient_dir.glob(ext)))
    return total

@tool
def get_patient_case_info(patient_id: int) -> Dict[str, Any]:
    """
    根据患者ID查询详细病历信息。

    参数:
        patient_id: 患者编号 (数字，例如 93)

    返回:
        包含患者完整病历信息的字典，包含：
        - patient_id: 患者编号
        - gender: 性别
        - age: 年龄
        - tumor_location: 肿瘤部位
        - histology_type: 组织类型
        - tnm_stage: TNM分期
        - ct_stage: cT分期
        - cn_stage: cN分期
        - cm_stage: cM分期
        - cea_level: CEA水平
        - mmr_status: MMR状态
        - ecog_score: ECOG评分
        - clinical_stage: 临床分期
    """
    db = get_case_database()
    case = db.get_case_by_id(patient_id)

    if not case:
        return {
            "error": f"未找到 ID {patient_id}",
            "patient_id": patient_id
        }

    # 返回原始数据（不在这里做UI格式化，保持原子化）
    return case


@tool
def upsert_patient_info(json_data: str) -> Dict[str, Any]:
    """
    ???????????????????????????????????????????????????lassification.xlsx??????????
    """
    excel_path = str(CLASSIFICATION_FILE)
    try:
        data = json.loads(json_data)
    except Exception as e:
        return {"error": f"JSON????: {str(e)}"}

    if not isinstance(data, dict):
        return {"error": "JSON payload must be an object"}

    try:
        upsert_case_record(excel_path, data)
        return {"status": "success", "message": f"???????? {excel_path}"}
    except PermissionError:
        backup_path = f"backup_{excel_path}"
        upsert_case_record(backup_path, data)
        return {"status": "success", "message": f"?? Excel ??????????? {backup_path}"}
    except ValueError as exc:
        return {"error": f"????????: {exc}"}
    except Exception as exc:
        return {"error": f"?? Excel ??: {exc}"}

@tool
def get_patient_imaging(patient_id: str) -> Dict[str, Any]:
    """
    获取患者的影像资料/图片/CT/MRI。

    参数:
        patient_id: 患者ID (例如 "093" 或 "93")

    返回:
        包含影像信息的字典，包含：
        - folder_name: 文件夹名称
        - images: 影像列表，每项包含 image_path, image_name, description
        - total_images: 影像总数
    """
    folder_name = str(patient_id).zfill(3)
    result = get_imaging_by_folder(folder_name)

    if not result:
        return {
            "error": "未找到影像",
            "folder_name": folder_name
        }

    return result


@tool
def summarize_patient_existing_info(patient_id: str) -> Dict[str, Any]:
    """
    汇总患者当前数据库中已收录的资料，用于回答“目前有哪些数据/已有信息”等问题。

    参数:
        patient_id: 患者ID (例如 "093" 或 "93")

    返回:
        包含患者已收录资料概览的字典，包括病例摘要、影像数量、病理切片数量和自然语言汇总。
    """
    folder_name = str(patient_id).zfill(3)
    try:
        patient_id_int = int(str(patient_id))
    except ValueError:
        return {
            "error": f"无效的患者ID: {patient_id}",
            "patient_id": folder_name,
        }

    db = get_case_database()
    case = db.get_case_by_id(patient_id_int)
    imaging = get_imaging_by_folder(folder_name)
    pathology_count = _count_pathology_slides(folder_name)

    has_case = isinstance(case, dict) and "error" not in case
    has_imaging = isinstance(imaging, dict) and "error" not in imaging
    has_pathology = pathology_count > 0

    if not any([has_case, has_imaging, has_pathology]):
        return {
            "error": f"未找到患者 {folder_name} 的已收录信息",
            "patient_id": folder_name,
            "available_data": {
                "case_info": False,
                "imaging": False,
                "pathology_slides": False,
            },
        }

    available_items: List[str] = []
    missing_items: List[str] = []

    if has_case:
        available_items.append("病例资料")
    else:
        missing_items.append("病例资料")

    if has_imaging:
        available_items.append(f"影像资料（{imaging.get('total_images', 0)}张）")
    else:
        missing_items.append("影像资料")

    if has_pathology:
        available_items.append(f"病理切片（{pathology_count}张）")
    else:
        missing_items.append("病理切片")

    summary_lines: List[str] = []
    if has_case:
        summary_lines.append(_build_case_brief(case))
    else:
        summary_lines.append(f"患者 {folder_name}: 未找到结构化病例资料")

    if available_items:
        summary_lines.append(f"已收录数据：{'、'.join(available_items)}")
    if missing_items:
        summary_lines.append(f"暂未收录：{'、'.join(missing_items)}")

    return {
        "patient_id": folder_name,
        "summary": "\n".join(summary_lines),
        "case_info": case if has_case else None,
        "imaging_info": {
            "folder_name": folder_name,
            "total_images": imaging.get("total_images", 0),
        } if has_imaging else None,
        "pathology_info": {
            "folder_name": folder_name,
            "total_slides": pathology_count,
        } if has_pathology else None,
        "available_data": {
            "case_info": has_case,
            "imaging": has_imaging,
            "pathology_slides": has_pathology,
        },
    }


@tool
def get_patient_pathology_slides(patient_id: str) -> Dict[str, Any]:
    """
    获取患者的病理切片预览图（将大切片转换为 256px PNG）。

    参数:
        patient_id: 患者ID (例如 "093" 或 "93")

    返回:
        包含病理切片信息的字典，包含：
        - folder_name: 文件夹名称
        - images: 预览图列表，每项包含 image_path, image_name, image_base64, source_slide
        - total_images: 预览图数量
        - slides: 原始切片路径列表
    """
    folder_name = str(patient_id).zfill(3)
    result = get_pathology_slides_by_patient_id(folder_name)

    if not result:
        return {
            "error": "未找到病理切片",
            "folder_name": folder_name
        }

    return result


@tool
def get_database_statistics() -> Dict[str, Any]:
    """
    查询数据库整体统计信息。

    返回:
        包含以下统计信息的字典：
        - total_cases: 总病例数
        - gender_distribution: 性别分布
        - age_statistics: 年龄统计（min, max, mean）
        - tumor_location_distribution: 肿瘤部位分布
        - ct_stage_distribution: cT分期分布
        - mmr_status_distribution: MMR状态分布
        - cea_statistics: CEA水平统计
    """
    return get_case_statistics()


@tool
def search_cases(
    tumor_location: Optional[str] = None,
    ct_stage: Optional[str] = None,
    cn_stage: Optional[str] = None,
    histology_type: Optional[str] = None,
    mmr_status: Optional[int] = None,
    age_min: Optional[int] = None,
    age_max: Optional[int] = None,
    cea_max: Optional[float] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    搜索虚拟病例数据库，查找符合条件的临床病例数据。

    支持的搜索条件：
    - tumor_location: 肿瘤部位 (如: 乙状, 直乙, 升, 降, 肝曲, 脾曲, 横)
    - ct_stage: cT分期 (如: 3, 4a, 4b)
    - cn_stage: cN分期 (如: 0, 1a, 1b, 2a, 2b)
    - histology_type: 组织类型 (如: 中, 低, 高, 粘液腺)
    - mmr_status: MMR状态 (1=pMMR/MSS, 2=dMMR/MSI-H)
    - age_min, age_max: 年龄范围
    - cea_max: 最大CEA水平
    - limit: 返回结果数量 (默认10)

    返回:
        符合条件的病例字典列表
    """
    results = query_cases(
        tumor_location=tumor_location,
        ct_stage=ct_stage,
        cn_stage=cn_stage,
        histology_type=histology_type,
        mmr_status=mmr_status,
        age_min=age_min,
        age_max=age_max,
        cea_max=cea_max,
        limit=limit
    )
    return results


@tool
def list_imaging_folders() -> List[str]:
    """
    列出所有可用的影像文件夹名称。

    返回:
        所有患者影像文件夹名称列表（如 ['001', '002', '093', ...]）
    """
    return get_all_folder_names()


@tool
def get_random_case(
    tumor_location: Optional[str] = None,
    ct_stage: Optional[str] = None,
    mmr_status: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    随机获取一个病例（可指定部位）。

    用于演示或模拟。

    可选筛选条件：
    - tumor_location: 限定肿瘤部位
    - ct_stage: 限定cT分期
    - mmr_status: 限定MMR状态

    返回:
        随机病例的字典，如果未找到则返回None
    """
    db = get_case_database()
    case = db.get_random_case(
        tumor_location=tumor_location,
        ct_stage=ct_stage,
        mmr_status=mmr_status
    )
    return case


# ============================================================
# 原子化工具列表（供 bind_tools 使用）
# ============================================================

# 导出原子工具列表
ATOMIC_DATABASE_TOOLS = [
    get_patient_case_info,
    summarize_patient_existing_info,
    upsert_patient_info,
    get_patient_imaging,
    get_patient_pathology_slides,
    get_database_statistics,
    search_cases,
    list_imaging_folders,
    get_random_case,
    perform_comprehensive_tumor_check_as_tool,  # 使用包装后的工具
]


# 工具工厂函数
def get_database_tools() -> List[Any]:
    """获取所有数据库工具"""
    return ATOMIC_DATABASE_TOOLS


def list_database_tools() -> List[str]:
    """列出所有数据库工具名称"""
    return [
        "search_case_database",
        "get_case_by_id",
        "summarize_patient_existing_info",
        "upsert_patient_info",
        "get_all_cases",
        "get_case_statistics",
        "get_random_case",
        "get_imaging_by_folder",
        "list_imaging_folders",
    ]
