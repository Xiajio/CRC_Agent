from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from backend.api.services.patient_commands import PatientCommandService
from backend.api.services.patient_registry_service import PatientRegistryService, normalize_patient_number


def _value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _qa_pairs(row: dict[str, str]) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for key, value in row.items():
        if not key.startswith("问题") or not value.strip():
            continue
        text = value.strip()
        if text.startswith("Q: ") and " | A: " in text:
            question, answer = text[3:].split(" | A: ", 1)
            pairs.append({"question": question.strip(), "answer": answer.strip()})
        else:
            pairs.append({"question": key, "answer": text})
    return pairs


def _patient_id_by_number(registry: PatientRegistryService, patient_number: str) -> int | None:
    normalized = normalize_patient_number(patient_number)
    with registry._connect() as connection:
        row = connection.execute(
            """
            SELECT id
            FROM patients
            WHERE patient_number_normalized = ?
            """,
            (normalized,),
        ).fetchone()
    return int(row["id"]) if row is not None else None


def _create_or_find_patient(
    commands: PatientCommandService,
    registry: PatientRegistryService,
    *,
    patient_number: str,
    source_session_id: str,
) -> int:
    existing = _patient_id_by_number(registry, patient_number)
    if existing is not None:
        return existing

    created = commands.create_patient(created_by_session_id=f"crc-import:{patient_number}")
    commands.set_identity(
        patient_id=created.patient_id,
        patient_name=f"CRC导入患者{patient_number}",
        patient_number=patient_number,
        source_session_id=source_session_id,
    )
    return created.patient_id


def _assessment_from_row(row: dict[str, str], *, source_session_id: str) -> dict[str, Any]:
    chief_complaint = _value(row, "用户主诉", "chief_complaint")
    risk_level = _value(row, "风险等级", "risk_level")
    referral_opinion = _value(row, "转诊意见", "referral_opinion")
    diagnosis_opinion = _value(row, "诊断意见", "diagnosis_opinion")
    key_clues = _value(row, "关键线索", "key_clues")
    visit_advice = _value(row, "就诊建议", "visit_advice")
    observation_points = _value(row, "诊前观察要点", "pre_visit_observation_points")
    patient_summary = "；".join(
        item for item in (chief_complaint, risk_level, referral_opinion) if item
    ) or "CRC-client历史问诊记录"

    return {
        "record_type": "crc_triage_assessment",
        "chief_complaint": chief_complaint or "CRC-client历史问诊",
        "symptom_group": "CRC-client历史导入",
        "risk_level": risk_level or "unknown",
        "disposition": referral_opinion or "unknown",
        "red_flags": [key_clues] if key_clues else [],
        "known_crc_signals": {"imported_key_clues": key_clues} if key_clues else {},
        "suggested_tests": [],
        "missing_information": [],
        "qa_summary": _qa_pairs(row),
        "patient_summary": patient_summary,
        "next_step": visit_advice or referral_opinion or "follow_up",
        "source_session_id": source_session_id,
        "source_subflow": "crc_triage",
        "diagnosis_opinion": diagnosis_opinion,
        "key_clues": key_clues,
        "visit_advice": visit_advice,
        "pre_visit_observation_points": observation_points,
        "current_visit_time": _value(row, "本次就诊时间", "current_visit_time"),
        "source": "CRC-client CSV",
    }


def import_crc_archive(
    *,
    csv_path: Path,
    registry: PatientRegistryService,
    source_session_id: str = "crc_archive_import",
) -> dict[str, int]:
    commands = PatientCommandService(registry)
    imported_records = 0
    skipped_rows = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            patient_number = _value(row, "患者ID", "患者编号", "patient_id", "patient_number")
            if not patient_number:
                skipped_rows += 1
                continue

            patient_id = _create_or_find_patient(
                commands,
                registry,
                patient_number=patient_number,
                source_session_id=source_session_id,
            )
            commands.record_crc_triage_assessment(
                patient_id=patient_id,
                assessment=_assessment_from_row(row, source_session_id=source_session_id),
                source_session_id=source_session_id,
            )
            imported_records += 1

    return {"imported_records": imported_records, "skipped_rows": skipped_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Import CRC-client assessment archive into LangG patient registry.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--registry-db", default=Path("runtime/patient_registry.db"), type=Path)
    parser.add_argument("--source-session-id", default="crc_archive_import")
    args = parser.parse_args()

    result = import_crc_archive(
        csv_path=args.csv,
        registry=PatientRegistryService(args.registry_db),
        source_session_id=args.source_session_id,
    )
    print(result)


if __name__ == "__main__":
    main()
