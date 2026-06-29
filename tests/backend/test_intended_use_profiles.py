from pathlib import Path

import yaml


CONFIG_PATH = Path("config/intended_use_profiles.yaml")


def test_intended_use_profiles_exist_and_match_safety_boundaries():
    assert CONFIG_PATH.exists()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    profiles = config["profiles"]
    assert set(profiles) == {
        "doctor_review",
        "patient_crc_triage",
        "research_workspace",
    }

    patient_crc_triage = profiles["patient_crc_triage"]
    assert patient_crc_triage["user_type"] == "patient"
    assert "collect_symptoms" in patient_crc_triage["allowed_tasks"]
    assert (
        "suggest_next_information_to_prepare"
        in patient_crc_triage["allowed_tasks"]
    )
    assert "final_diagnosis" in patient_crc_triage["forbidden_tasks"]
    assert "treatment_decision" in patient_crc_triage["forbidden_tasks"]
    assert "screening_conclusion" in patient_crc_triage["forbidden_tasks"]
    assert (
        patient_crc_triage["disclaimer_key"]
        == "patient_crc_triage_disclaimer"
    )
    assert patient_crc_triage["evidence_required"] is False

    doctor_review = profiles["doctor_review"]
    assert "auto_sign" in doctor_review["forbidden_tasks"]
    assert "override_clinician_decision" in doctor_review["forbidden_tasks"]
    assert doctor_review["evidence_required"] is True

    research_workspace = profiles["research_workspace"]
    assert "patient_advice" in research_workspace["forbidden_tasks"]
    assert "clinical_decision" in research_workspace["forbidden_tasks"]
    assert research_workspace["evidence_required"] is True
