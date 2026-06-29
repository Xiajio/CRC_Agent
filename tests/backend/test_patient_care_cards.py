from __future__ import annotations

from backend.api.services.patient_care_cards import build_patient_care_cards


def test_build_patient_care_cards_from_latest_crc_triage_record() -> None:
    cards = build_patient_care_cards(
        [
            {
                "record_type": "crc_triage_assessment",
                "normalized_payload_json": {
                    "red_flags": ["rectal_bleeding"],
                    "known_crc_signals": {"rectal_bleeding": True},
                    "suggested_tests": ["血常规", "肠镜"],
                    "missing_information": ["内镜关键发现"],
                    "disposition": "urgent_gi_clinic",
                    "next_step": "urgent_gi_clinic",
                },
                "summary_text": "建议尽快消化专科评估。",
            }
        ]
    )

    assert "留意便血或黑便是否加重" in cards["focusMetrics"]
    assert "补充内镜关键发现" in cards["focusMetrics"]
    assert "尽快预约消化专科门诊" in cards["periodicChecks"]
    assert "准备或完成：血常规" in cards["periodicChecks"]
    assert "记录便血颜色、次数和伴随症状" in cards["dailyActions"]


def test_build_patient_care_cards_returns_default_guidance_without_records() -> None:
    cards = build_patient_care_cards([])

    assert cards["focusMetrics"]
    assert cards["periodicChecks"]
    assert cards["dailyActions"]
