from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

from .case_excel_service import load_case_records


DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "Case Database"
CLINICAL_CASE_DIR = DATA_ROOT / "Clinical Case"
CLASSIFICATION_FILE = CLINICAL_CASE_DIR / "classification.xlsx"
RADIOGRAPHIC_IMAGING_DIR = DATA_ROOT / "Radiographic Imaging"
PATHOLOGY_THUMBNAILS_DIR = DATA_ROOT / "Pathology Thumbnails"
PATHOLOGY_SLIDES_DIR = DATA_ROOT / "Silds"
PATHOLOGY_SLIDE_FOLDERS = [PATHOLOGY_SLIDES_DIR]

EMPTY_CASE_STATISTICS: dict[str, Any] = {
    "total_cases": 0,
    "gender_distribution": {},
    "age_statistics": {"min": None, "max": None, "mean": None},
    "tumor_location_distribution": {},
    "ct_stage_distribution": {},
    "mmr_status_distribution": {},
    "cea_statistics": {"min": None, "max": None, "mean": None},
}


def _gender_label(value: Any) -> str:
    if value in (1, "1", "男", "male", "Male"):
        return "男"
    if value in (2, "2", "女", "female", "Female"):
        return "女"
    return ""


def _mmr_label(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"1", "pmmr", "mss", "pmmr (mss)"} or "pmmr" in raw or "mss" in raw:
        return "pMMR (MSS)"
    if raw in {"2", "dmmr", "msi-h", "dmmr / msi-h"} or "dmmr" in raw or "msi" in raw:
        return "dMMR / MSI-H"
    return str(value or "").strip()


def _coerce_number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _numeric_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 2),
    }


def _patient_folder_name(patient_id: Any) -> str:
    try:
        return f"{int(patient_id):03d}"
    except Exception:
        return str(patient_id or "").zfill(3)


def _serialize_case(case: dict[str, Any]) -> dict[str, Any]:
    ct_stage = str(case.get("ct_stage") or "").strip()
    cn_stage = str(case.get("cn_stage") or "").strip()
    cm_stage = str(case.get("cm_stage") or "0").strip() or "0"

    return {
        **case,
        "gender": _gender_label(case.get("gender")) or case.get("gender"),
        "mmr_status": _mmr_label(case.get("mmr_status")),
        "ct_stage": ct_stage,
        "cn_stage": cn_stage,
        "cm_stage": cm_stage,
        "tnm_stage": case.get("tnm_stage") or f"cT{ct_stage}N{cn_stage}M{cm_stage}",
        "clinical_stage": case.get("clinical_stage") or "",
        "risk_factors": list(case.get("risk_factors") or []),
    }


class VirtualCaseDatabase:
    def __init__(self, classification_file: str | Path = CLASSIFICATION_FILE) -> None:
        self.classification_file = Path(classification_file)
        self.cases: list[dict[str, Any]] = []
        self._load_data()

    def _load_data(self) -> None:
        self.cases = [_serialize_case(case) for case in load_case_records(self.classification_file)]

    def get_all_cases(self) -> list[dict[str, Any]]:
        return [dict(case) for case in self.cases]

    def get_case_by_id(self, patient_id: int) -> dict[str, Any] | None:
        for case in self.cases:
            if int(case.get("patient_id") or 0) == int(patient_id):
                return dict(case)
        return None

    def get_random_case(
        self,
        *,
        tumor_location: str | None = None,
        ct_stage: str | None = None,
        mmr_status: int | None = None,
    ) -> dict[str, Any] | None:
        candidates = self.cases
        if tumor_location:
            candidates = [case for case in candidates if tumor_location in str(case.get("tumor_location") or "")]
        if ct_stage:
            candidates = [case for case in candidates if str(case.get("ct_stage") or "") == str(ct_stage)]
        if mmr_status is not None:
            expected = _mmr_label(mmr_status)
            candidates = [case for case in candidates if _mmr_label(case.get("mmr_status")) == expected]
        if not candidates:
            return None
        return dict(random.choice(candidates))

    def get_statistics(self) -> dict[str, Any]:
        if not getattr(self, "cases", None):
            return dict(EMPTY_CASE_STATISTICS)

        genders = Counter(
            gender for gender in (_gender_label(case.get("gender")) for case in self.cases) if gender
        )
        locations = Counter(str(case.get("tumor_location") or "").strip() for case in self.cases if case.get("tumor_location"))
        ct_stages = Counter(str(case.get("ct_stage") or "").strip() for case in self.cases if case.get("ct_stage"))
        mmr_status = Counter(_mmr_label(case.get("mmr_status")) for case in self.cases if _mmr_label(case.get("mmr_status")))
        ages = [value for value in (_coerce_number(case.get("age")) for case in self.cases) if value is not None]
        cea_values = [value for value in (_coerce_number(case.get("cea_level")) for case in self.cases) if value is not None]

        return {
            "total_cases": len(self.cases),
            "gender_distribution": dict(genders),
            "age_statistics": _numeric_stats(ages),
            "tumor_location_distribution": dict(locations),
            "ct_stage_distribution": dict(ct_stages),
            "mmr_status_distribution": dict(mmr_status),
            "cea_statistics": _numeric_stats(cea_values),
        }


_CASE_DATABASE: VirtualCaseDatabase | None = None


def get_case_database() -> VirtualCaseDatabase:
    global _CASE_DATABASE
    if _CASE_DATABASE is None:
        _CASE_DATABASE = VirtualCaseDatabase()
    return _CASE_DATABASE


def load_cases_from_database() -> list[dict[str, Any]]:
    return get_case_database().get_all_cases()


def get_case_statistics() -> dict[str, Any]:
    return get_case_database().get_statistics()


def query_cases(
    *,
    tumor_location: str | None = None,
    ct_stage: str | None = None,
    cn_stage: str | None = None,
    histology_type: str | None = None,
    mmr_status: int | None = None,
    age_min: int | None = None,
    age_max: int | None = None,
    cea_max: float | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    cases = get_case_database().get_all_cases()
    results: list[dict[str, Any]] = []

    expected_mmr = _mmr_label(mmr_status) if mmr_status is not None else ""
    for case in cases:
        if tumor_location and tumor_location not in str(case.get("tumor_location") or ""):
            continue
        if ct_stage and str(case.get("ct_stage") or "") != str(ct_stage):
            continue
        if cn_stage and str(case.get("cn_stage") or "") != str(cn_stage):
            continue
        if histology_type and histology_type not in str(case.get("histology_type") or ""):
            continue
        if expected_mmr and _mmr_label(case.get("mmr_status")) != expected_mmr:
            continue

        age = _coerce_number(case.get("age"))
        if age_min is not None and (age is None or age < age_min):
            continue
        if age_max is not None and (age is None or age > age_max):
            continue

        cea = _coerce_number(case.get("cea_level"))
        if cea_max is not None and (cea is None or cea > cea_max):
            continue

        results.append(case)
        if len(results) >= max(limit, 1):
            break

    return results


def _collect_image_entries(folder: Path) -> list[dict[str, Any]]:
    if not folder.exists():
        return []
    image_paths = sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        ]
    )
    return [
        {
            "image_path": str(path),
            "image_name": path.name,
            "description": "",
        }
        for path in image_paths
    ]


def get_imaging_by_folder(folder_name: str) -> dict[str, Any]:
    folder = RADIOGRAPHIC_IMAGING_DIR / str(folder_name)
    images = _collect_image_entries(folder)
    if not images:
        return {"error": "Imaging data is unavailable.", "folder_name": str(folder_name)}
    return {
        "folder_name": str(folder_name),
        "images": images,
        "total_images": len(images),
    }


def get_imaging_by_patient_id(patient_id: int | str) -> dict[str, Any]:
    return get_imaging_by_folder(_patient_folder_name(patient_id))


def get_pathology_slides_by_patient_id(patient_id: str) -> dict[str, Any]:
    folder_name = _patient_folder_name(patient_id)
    folder = PATHOLOGY_THUMBNAILS_DIR / folder_name
    images = _collect_image_entries(folder)
    if not images:
        return {"error": "Pathology slide data is unavailable.", "folder_name": folder_name}
    return {
        "folder_name": folder_name,
        "images": images,
        "total_images": len(images),
        "slides": [image["image_path"] for image in images],
    }


def get_all_folder_names() -> list[str]:
    folders: set[str] = set()
    for root in (RADIOGRAPHIC_IMAGING_DIR, PATHOLOGY_THUMBNAILS_DIR, PATHOLOGY_SLIDES_DIR):
        if not root.exists():
            continue
        folders.update(path.name for path in root.iterdir() if path.is_dir())
    return sorted(folders)


__all__ = [
    "CLASSIFICATION_FILE",
    "PATHOLOGY_SLIDE_FOLDERS",
    "EMPTY_CASE_STATISTICS",
    "VirtualCaseDatabase",
    "get_case_database",
    "load_cases_from_database",
    "query_cases",
    "get_case_statistics",
    "get_imaging_by_folder",
    "get_imaging_by_patient_id",
    "get_pathology_slides_by_patient_id",
    "get_all_folder_names",
]
