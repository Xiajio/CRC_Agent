from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import openpyxl

import backend.api.services.database_service as database_service_module
import src.tools.card_formatter as card_formatter_module
from backend.api.schemas.database import DatabaseFilters
from backend.api.services.database_service import _apply_filters, get_database_stats
from src.services.case_excel_service import (
    load_case_records,
    normalize_case_payload,
    upsert_case_record,
)
from src.services.virtual_database_service import VirtualCaseDatabase
from src.tools.card_formatter import CardFormatter


EMPTY_STATS = {
    "total_cases": 0,
    "gender_distribution": {},
    "age_statistics": {"min": None, "max": None, "mean": None},
    "tumor_location_distribution": {},
    "ct_stage_distribution": {},
    "mmr_status_distribution": {},
    "cea_statistics": {"min": None, "max": None, "mean": None},
}


def _make_workspace_tmp_dir() -> Path:
    base_dir = Path("tests") / "backend" / ".tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = base_dir / uuid4().hex
    tmp_dir.mkdir()
    return tmp_dir


def test_normalize_case_payload_accepts_new_history_fields() -> None:
    normalized, extras = normalize_case_payload(
        {
            "patient_id": "101",
            "gender": "女",
            "age": "54",
            "ecog_score": "2",
            "histology_type": "腺癌",
            "tumor_location": "直肠",
            "ct_stage": "cT3",
            "cn_stage": "cN1",
            "clinical_stage": "III期",
            "cea_level": "12.4",
            "mmr_status": "pMMR",
            "chief_complaint": "腹痛伴排便习惯改变",
            "symptom_duration": "3个月",
            "family_history": "是",
            "family_history_details": "父亲结直肠癌",
            "biopsy_confirmed": "0",
            "biopsy_details": "外院肠镜活检提示腺癌",
            "risk_factors": '["吸烟", "肥胖"]',
        }
    )

    assert extras == {}
    assert normalized["patient_id"] == 101
    assert normalized["ecog_score"] == 2
    assert normalized["family_history"] is True
    assert normalized["biopsy_confirmed"] is False
    assert normalized["chief_complaint"] == "腹痛伴排便习惯改变"
    assert normalized["risk_factors"] == ["吸烟", "肥胖"]


def test_upsert_and_load_case_records_round_trip_new_history_fields() -> None:
    tmp_dir = _make_workspace_tmp_dir()
    try:
        excel_path = tmp_dir / "classification.xlsx"

        upsert_case_record(
            excel_path,
            {
                "patient_id": 102,
                "gender": 1,
                "age": 61,
                "ecog_score": 1,
                "histology_type": "中分化腺癌",
                "tumor_location": "乙状结肠",
                "ct_stage": "3",
                "cn_stage": "1",
                "clinical_stage": "III期",
                "cea_level": 8.6,
                "mmr_status": 2,
                "chief_complaint": "便血",
                "symptom_duration": "2周",
                "family_history": True,
                "family_history_details": "母亲胃癌",
                "biopsy_confirmed": False,
                "biopsy_details": "待补充",
                "risk_factors": ["吸烟", "饮酒"],
            },
        )

        records = load_case_records(excel_path)

        assert len(records) == 1
        assert records[0]["ecog_score"] == 1
        assert records[0]["family_history"] is True
        assert records[0]["biopsy_confirmed"] is False
        assert records[0]["family_history_details"] == "母亲胃癌"
        assert records[0]["risk_factors"] == ["吸烟", "饮酒"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_load_case_records_accepts_legacy_rows_and_risk_factor_fallback() -> None:
    tmp_dir = _make_workspace_tmp_dir()
    try:
        excel_path = tmp_dir / "legacy.xlsx"
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(
            [
                "patient_id",
                "gender",
                "age",
                "histology_type",
                "tumor_location",
                "ct_stage",
                "cn_stage",
                "clinical_stage",
                "cea_level",
                "mmr_status",
                "risk_factors",
            ]
        )
        sheet.append([201, 1, 58, "腺癌", "直肠", "3", "1", "III期", 7.2, 1, "吸烟, 饮酒"])
        workbook.save(excel_path)

        records = load_case_records(excel_path)

        assert len(records) == 1
        assert records[0]["ecog_score"] is None
        assert records[0]["chief_complaint"] is None
        assert records[0]["family_history"] is None
        assert records[0]["biopsy_confirmed"] is None
        assert records[0]["risk_factors"] == ["吸烟", "饮酒"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_apply_filters_supports_history_booleans_and_ecog_range() -> None:
    items = [
        {
            "patient_id": 1,
            "age": 60,
            "ecog_score": 1,
            "family_history": True,
            "biopsy_confirmed": False,
        },
        {
            "patient_id": 2,
            "age": 68,
            "ecog_score": 3,
            "family_history": False,
            "biopsy_confirmed": True,
        },
        {
            "patient_id": 3,
            "age": 55,
            "ecog_score": None,
            "family_history": True,
            "biopsy_confirmed": True,
        },
    ]

    filtered = _apply_filters(
        items,
        DatabaseFilters(
            family_history=True,
            biopsy_confirmed=False,
            ecog_min=1,
            ecog_max=2,
        ),
    )

    assert [item["patient_id"] for item in filtered] == [1]


def test_get_database_stats_returns_complete_empty_shape(monkeypatch) -> None:
    monkeypatch.setattr(database_service_module, "get_case_statistics", lambda: {"total_cases": 0})

    assert get_database_stats() == EMPTY_STATS


def test_virtual_case_database_empty_statistics_return_complete_shape() -> None:
    database = object.__new__(VirtualCaseDatabase)
    database.cases = []

    assert VirtualCaseDatabase.get_statistics(database) == EMPTY_STATS


def test_format_patient_card_emits_history_block(monkeypatch) -> None:
    class FakeDb:
        def get_case_by_id(self, patient_id: int):
            return {
                "patient_id": patient_id,
                "gender": "女",
                "age": 52,
                "ecog_score": 1,
                "histology_type": "腺癌",
                "tumor_location": "直肠",
                "ct_stage": "3",
                "cn_stage": "1",
                "clinical_stage": "III期",
                "cea_level": 4.2,
                "mmr_status": "pMMR (MSS)",
                "tnm_stage": "cT3N1M0",
                "chief_complaint": "腹痛",
                "symptom_duration": "3天",
                "family_history": True,
                "family_history_details": "父亲结直肠癌",
                "biopsy_confirmed": True,
                "biopsy_details": "肠镜活检已提示腺癌",
                "risk_factors": ["吸烟", "肥胖"],
            }

    monkeypatch.setattr(card_formatter_module, "get_case_database", lambda: FakeDb())

    card = CardFormatter().format_patient_card(301)
    history_block = card["data"]["history_block"]

    assert history_block["chief_complaint"] == "腹痛"
    assert history_block["family_history"] is True
    assert history_block["biopsy_confirmed"] is True
    assert history_block["risk_factors"] == ["吸烟", "肥胖"]