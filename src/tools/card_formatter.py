"""Card formatting helpers for frontend rendering."""

from __future__ import annotations

import base64
import os
from datetime import datetime
from typing import Any, Dict

from ..services.virtual_database_service import get_case_database


class CardFormatter:
    def format_patient_card(self, patient_id: int) -> Dict[str, Any]:
        db = get_case_database()
        case = db.get_case_by_id(patient_id)

        if not case:
            return {"error": f"Patient {patient_id} not found."}

        history_block = {
            "chief_complaint": case.get("chief_complaint"),
            "symptom_duration": case.get("symptom_duration"),
            "family_history": case.get("family_history"),
            "family_history_details": case.get("family_history_details"),
            "biopsy_confirmed": case.get("biopsy_confirmed"),
            "biopsy_details": case.get("biopsy_details"),
            "risk_factors": list(case.get("risk_factors") or []),
        }

        return {
            "type": "patient_card",
            "patient_id": case["patient_id"],
            "data": {
                "diagnosis_block": {
                    "confirmed": f"结直肠癌 ({case['histology_type']})",
                    "histology": case["histology_type"],
                    "primary_site": case["tumor_location"],
                    "mmr_status": case["mmr_status"],
                },
                "staging_block": {
                    "clinical_stage": case["clinical_stage"] or case["tnm_stage"],
                    "ct_stage": case["ct_stage"],
                    "cn_stage": case["cn_stage"],
                    "cm_stage": "M0",
                },
                "patient_info": {
                    "gender": case["gender"],
                    "age": case["age"],
                    "ecog": case.get("ecog_score"),
                    "cea": case["cea_level"],
                },
                "history_block": history_block,
                "raw_data": case,
            },
            "text_summary": self._format_text_summary(case),
        }

    def format_imaging_card(self, imaging_data: Dict[str, Any]) -> Dict[str, Any]:
        if not imaging_data or "error" in imaging_data:
            return {"error": "Imaging data is unavailable."}

        folder_name = imaging_data.get("folder_name", "Unknown")
        total = imaging_data.get("total_images", 0)
        return {
            "type": "imaging_card",
            "data": imaging_data,
            "text_summary": f"影像目录 {folder_name}，共 {total} 张图像。",
        }

    def format_pathology_slide_card(self, slide_data: Dict[str, Any]) -> Dict[str, Any]:
        if not slide_data or "error" in slide_data:
            return {"error": "Pathology slide data is unavailable."}

        folder_name = slide_data.get("folder_name", "Unknown")
        total = slide_data.get("total_images", 0)
        return {
            "type": "pathology_slide_card",
            "data": slide_data,
            "text_summary": f"病理切片目录 {folder_name}，共 {total} 张预览。",
        }

    def format_tumor_screening_result(self, screening_result: Dict[str, Any]) -> Dict[str, Any]:
        if not screening_result or "error" in screening_result:
            return {"error": screening_result.get("error", "Tumor screening result is unavailable.")}

        success = screening_result.get("success", False)
        has_tumor = screening_result.get("has_tumor", False)
        patient_id = screening_result.get("patient_id", "Unknown")
        image_path = screening_result.get("image_path", "")
        confidence = screening_result.get("confidence")
        bounding_boxes = screening_result.get("bounding_boxes", [])
        total_detections = screening_result.get("total_detections", 0)
        timestamp = screening_result.get("processing_timestamp", "")

        formatted_time = self._format_timestamp(timestamp)
        if not success:
            conclusion = "检测失败"
        elif has_tumor:
            conclusion = f"发现肿瘤可疑区域 ({total_detections} 处)"
        else:
            conclusion = "未发现肿瘤可疑区域"

        image_url = image_path.replace("\\", "/") if image_path else ""

        return {
            "type": "tumor_screening_result",
            "data": {
                "success": success,
                "patient_id": patient_id,
                "has_tumor": has_tumor,
                "conclusion": conclusion,
                "confidence": confidence,
                "total_detections": total_detections,
                "bounding_boxes": bounding_boxes,
                "image_url": image_url,
                "image_path": image_path,
                "processing_time": formatted_time,
            },
            "text_summary": f"患者 {patient_id} 肿瘤筛查结论：{conclusion}",
            "display": {
                "show_image": True,
                "image_url": image_url,
                "image_alt": f"患者 {patient_id} 肿瘤筛查图像",
                "result_badge": "negative" if not has_tumor else "positive",
                "result_color": "green" if not has_tumor else "red",
            },
        }

    def _read_image_as_base64(self, image_path: str) -> str:
        if not image_path or not os.path.exists(image_path):
            return ""

        try:
            with open(image_path, "rb") as handle:
                return base64.b64encode(handle.read()).decode()
        except Exception:
            return ""

    def format_comprehensive_tumor_detection(
        self,
        screening_result: Dict[str, Any],
        include_images: bool = True,
    ) -> Dict[str, Any]:
        if not screening_result or "error" in screening_result:
            return {"error": screening_result.get("error", "Tumor detection result is unavailable.")}

        patient_id = screening_result.get("patient_id", "Unknown")
        formatted_time = self._format_timestamp(screening_result.get("processing_timestamp", ""))

        sample_images = screening_result.get("sample_images_with_tumor", [])
        processed_sample_images = []
        if include_images and sample_images:
            for image_info in sample_images:
                image_path = image_info.get("image_path", "")
                processed_sample_images.append(
                    {
                        "image_name": image_info.get("image_name", "Unknown"),
                        "image_path": image_path,
                        "image_base64": self._read_image_as_base64(image_path),
                        "confidence": image_info.get("confidence", 0.0),
                        "total_detections": image_info.get("total_detections", 0),
                        "bounding_boxes": image_info.get("bounding_boxes", []),
                    }
                )

        return {
            "type": "tumor_detection_card",
            "patient_id": patient_id,
            "data": {
                "total_images": screening_result.get("total_images", 0),
                "images_with_tumor": screening_result.get("images_with_tumor", 0),
                "images_without_tumor": screening_result.get("images_without_tumor", 0),
                "tumor_detection_rate": screening_result.get("tumor_detection_rate", "0%"),
                "has_tumor": screening_result.get("has_tumor", False),
                "max_confidence": screening_result.get("max_confidence", 0.0),
                "total_detections": screening_result.get("total_detections", 0),
                "processing_timestamp": formatted_time,
                "sample_images_with_tumor": processed_sample_images,
                "all_results": screening_result.get("all_results", []),
            },
            "text_summary": (
                f"患者 {patient_id} 共分析 {screening_result.get('total_images', 0)} 张图像，"
                f"检出阳性样本 {screening_result.get('images_with_tumor', 0)} 张。"
            ),
        }

    def _format_text_summary(self, case: Dict[str, Any]) -> str:
        summary = f"患者 {case['patient_id']}: {case['gender']}/{case['age']}岁"
        diagnosis = f"{case['tumor_location']} {case['histology_type']}"
        stage = case.get("clinical_stage") or case.get("tnm_stage") or "分期未明"
        complaint = case.get("chief_complaint")
        if complaint:
            return f"{summary}，主诉 {complaint}，诊断 {diagnosis}，当前分期 {stage}。"
        return f"{summary}，诊断 {diagnosis}，当前分期 {stage}。"

    def format_radiomics_report_card(self, report_data: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
        if not report_data:
            return {"error": "Radiomics report is unavailable."}

        summary = report_data.get("summary", {})
        yolo_screening = report_data.get("yolo_screening", {})
        analyzed_images = report_data.get("analyzed_images", [])

        total_images = summary.get("total_images", 0)
        images_with_tumor = yolo_screening.get("passed", 0)
        analyzed_count = len(analyzed_images)
        top_features = []
        for image_result in analyzed_images:
            if "top_features" in image_result:
                top_features = image_result["top_features"]
                break

        return {
            "type": "radiomics_report_card",
            "patient_id": patient_id,
            "data": {
                "patient_id": patient_id,
                "analysis_mode": "complete_radiomics_analysis",
                "total_images": total_images,
                "images_with_tumor": images_with_tumor,
                "analyzed_images_count": analyzed_count,
                "has_tumor": images_with_tumor > 0,
                "yolo_screening": yolo_screening,
                "analyzed_images": analyzed_images,
                "top_features": top_features,
                "summary": summary,
                "report_file": (
                    f"data/Case Database/Radiographic Imaging/{patient_id}/"
                    "radiomics_analysis/comprehensive_analysis_report.json"
                ),
                "timestamp": summary.get("timestamp", ""),
            },
            "text_summary": (
                f"患者 {patient_id} 影像组学分析完成：总图像 {total_images} 张，"
                f"阳性筛中 {images_with_tumor} 张，深入分析 {analyzed_count} 张。"
            ),
        }

    @staticmethod
    def _format_timestamp(timestamp: str) -> str:
        if not timestamp:
            return "未知"
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00").replace("T", " "))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return timestamp


formatter = CardFormatter()
