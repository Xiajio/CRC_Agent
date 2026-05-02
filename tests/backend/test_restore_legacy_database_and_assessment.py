from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import openpyxl
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from src.nodes.assessment_nodes import node_assessment, node_staging_router
from src.services.case_excel_service import load_case_records
from src.services.virtual_database_service import VirtualCaseDatabase
from src.state import CRCAgentState


def _make_workspace_tmp_dir() -> Path:
    base_dir = Path("tests") / "backend" / ".tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = base_dir / uuid4().hex
    tmp_dir.mkdir()
    return tmp_dir


class _UnusedModel:
    def with_structured_output(self, _schema):
        def _unexpected_invoke(_payload):
            raise AssertionError("Structured assessment chain should not run in recovery regression tests.")

        return RunnableLambda(_unexpected_invoke)


def test_load_case_records_accepts_historical_classification_headers() -> None:
    tmp_dir = _make_workspace_tmp_dir()
    try:
        excel_path = tmp_dir / "legacy-classification.xlsx"
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(
            [
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
        )
        sheet.append(["002", 1, 57, 1, "中分化腺癌", "降结肠", "4b", "2a", "III期", 7.54, 1])
        workbook.save(excel_path)

        records = load_case_records(excel_path)

        assert len(records) == 1
        assert records[0]["patient_id"] == 2
        assert records[0]["age"] == 57
        assert records[0]["ecog_score"] == 1
        assert records[0]["histology_type"] == "中分化腺癌"
        assert records[0]["tumor_location"] == "降结肠"
        assert records[0]["ct_stage"] == "4b"
        assert records[0]["cn_stage"] == "2a"
        assert records[0]["clinical_stage"] == "III期"
        assert records[0]["cea_level"] == 7.54
        assert records[0]["mmr_status"] == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_virtual_case_database_reads_non_empty_cases_from_historical_headers() -> None:
    tmp_dir = _make_workspace_tmp_dir()
    try:
        excel_path = tmp_dir / "legacy-classification.xlsx"
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(
            [
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
        )
        sheet.append(["093", 1, 31, 0, "中分化腺癌", "横结肠", "4b", "1c", "III期", 10.29, 1])
        workbook.save(excel_path)

        database = VirtualCaseDatabase(excel_path)
        cases = database.get_all_cases()

        assert len(cases) == 1
        assert cases[0]["patient_id"] == 93
        assert cases[0]["tumor_location"] == "横结肠"
        assert database.get_statistics()["total_cases"] == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_assessment_fast_passes_complete_database_query_context() -> None:
    assessment = node_assessment(model=_UnusedModel(), tools=[], show_thinking=False)

    result = assessment(
        CRCAgentState(
            messages=[HumanMessage(content="这个数据库病例下一步怎么治疗？")],
            encounter_track="crc_clinical",
            findings={
                "user_intent": "treatment_decision",
                "data_source": "database_query",
                "db_query_patient_id": "002",
                "pathology_confirmed": True,
                "biopsy_confirmed": True,
                "tumor_location": "colon",
                "histology_type": "腺癌",
                "tnm_staging": {"cT": "cT3", "cN": "cN1", "cM": "cM0"},
                "clinical_stage_summary": "结肠腺癌 cT3N1M0 III期",
                "molecular_markers": {"MSI-H": True},
                "mmr_status": "dMMR",
                "age": 57,
                "gender": "男",
                "ecog_score": 1,
            },
        )
    )

    assert result["clinical_stage"] == "Assessment_Completed"
    assert result["missing_critical_data"] == []
    assert result["findings"]["fast_pass_mode"] is True
    assert result["findings"]["pathology_confirmed"] is True
    assert result["findings"]["tumor_location"] == "colon"
    assert result["findings"]["is_advanced_stage"] is True
    assert result["patient_profile"].is_locked is True
    assert result["patient_profile"].age == 57
    assert result["patient_profile"].gender == "男"
    assert result["patient_profile"].mmr_status == "dMMR"


def test_assessment_preserves_pathology_tnm_location_and_mmr_inquiry_chain() -> None:
    assessment = node_assessment(model=_UnusedModel(), tools=[], show_thinking=False)

    pathology_only = assessment(
        CRCAgentState(
            messages=[HumanMessage(content="术后病理提示腺癌")],
            encounter_track="crc_clinical",
            findings={"user_intent": "clinical_assessment"},
        )
    )
    assert pathology_only["findings"]["inquiry_type"] == "tnm_required"

    missing_location = assessment(
        CRCAgentState(
            messages=[HumanMessage(content="术后病理提示腺癌，cT3N1M0")],
            encounter_track="crc_clinical",
            findings={"user_intent": "clinical_assessment"},
        )
    )
    assert missing_location["findings"]["inquiry_type"] == "location_required"

    missing_mmr = assessment(
        CRCAgentState(
            messages=[HumanMessage(content="术后病理提示结肠腺癌，cT3N1M0")],
            encounter_track="crc_clinical",
            findings={"user_intent": "clinical_assessment"},
        )
    )
    assert missing_mmr["findings"]["inquiry_type"] == "mmr_status_required"

    completed = assessment(
        CRCAgentState(
            messages=[HumanMessage(content="术后病理提示结肠腺癌，cT3N1M0，dMMR")],
            encounter_track="crc_clinical",
            findings={"user_intent": "clinical_assessment"},
        )
    )
    assert completed["clinical_stage"] == "Assessment_Completed"
    assert completed["missing_critical_data"] == []


def test_staging_router_shortcuts_fast_pass_cases_to_decision() -> None:
    route = node_staging_router(
        CRCAgentState(
            clinical_stage="Assessment_Completed",
            findings={
                "fast_pass_mode": True,
                "pathology_confirmed": True,
                "tumor_location": "colon",
                "tnm_staging": {"cT": "cT3", "cN": "cN1", "cM": "cM0"},
            },
        )
    )

    assert route == "decision"
