from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from backend.api.schemas.doctor_review import (
    DoctorReviewDraft,
    DoctorReviewDraftSection,
    DoctorReviewProvenanceRef,
    DoctorReviewResponse,
    DoctorReviewTimelineItem,
)
from src.contracts.clinical_assertion import ClinicalAssertion
from src.services.clinical_assertion_projection import (
    assertion_refs,
    project_clinical_assertions_from_records,
)

router = APIRouter(prefix="/api/sessions", tags=["doctor-review"])

AVAILABLE_ACTIONS = [
    "accept",
    "edit",
    "reject",
    "escalate",
    "request_evidence",
    "mark_unsafe",
]


def _runtime_dependencies(request: Request) -> tuple[Any, Any]:
    runtime = getattr(request.app.state, "runtime", None)
    session_store = getattr(runtime, "session_store", None)
    patient_registry = getattr(runtime, "patient_registry_service", None)
    if session_store is None or patient_registry is None:
        raise HTTPException(status_code=503, detail="Runtime is not initialized")
    return session_store, patient_registry


def _title_for_record(row: Mapping[str, Any]) -> str:
    for key in ("summary_text", "document_type"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    record_id = row.get("record_id")
    return f"Patient record {record_id}" if record_id is not None else "Patient record"


def _timeline_item(row: Mapping[str, Any]) -> DoctorReviewTimelineItem:
    assertions = project_clinical_assertions_from_records([row])
    return DoctorReviewTimelineItem(
        item_id=str(row["record_id"]),
        kind=str(row.get("record_type") or row.get("document_type") or "patient_record"),
        title=_title_for_record(row),
        created_at=str(row["created_at"]),
        assertion_refs=assertion_refs(assertions),
    )


def _risk_assertion_refs(assertions: list[ClinicalAssertion]) -> list[str]:
    risk_fact_types = {"condition_signal", "symptom", "risk_disposition", "safety_rule_match"}
    return [
        assertion.assertion_id
        for assertion in assertions
        if assertion.normalized_fact.type in risk_fact_types
    ]


def _provenance_refs(assertions: list[ClinicalAssertion]) -> list[DoctorReviewProvenanceRef]:
    refs: list[DoctorReviewProvenanceRef] = []
    seen: set[tuple[str, str, str | None]] = set()
    for assertion in assertions:
        for evidence_ref in assertion.evidence_refs:
            key = (evidence_ref.kind, evidence_ref.id, evidence_ref.field)
            if key in seen:
                continue
            seen.add(key)
            refs.append(
                DoctorReviewProvenanceRef(
                    kind=evidence_ref.kind,
                    id=evidence_ref.id,
                    field=evidence_ref.field,
                )
            )
    return refs


def _build_draft(patient_id: int, assertions: list[ClinicalAssertion]) -> DoctorReviewDraft:
    risk_refs = _risk_assertion_refs(assertions)
    all_refs = assertion_refs(assertions)
    provenance_refs = _provenance_refs(assertions)
    return DoctorReviewDraft(
        draft_id=f"draft_crc_review_{patient_id}_latest",
        sections=[
            DoctorReviewDraftSection(
                section_id="risk_summary",
                kind="traceable_risk_summary",
                title="Traceable risk summary",
                body=(
                    "Structured CRC triage signals indicate risk requiring doctor review. "
                    "Verify each assertion against the cited patient record evidence."
                ),
                assertion_refs=risk_refs or all_refs,
                provenance_refs=provenance_refs,
            ),
            DoctorReviewDraftSection(
                section_id="unverified_note",
                kind="model_generated_unverified",
                title="Unverified draft note",
                body=(
                    "This draft is generated from structured patient records and has not "
                    "been clinically verified."
                ),
                assertion_refs=all_refs,
                provenance_refs=provenance_refs,
            ),
        ],
    )


@router.get("/{session_id}/doctor-review", response_model=DoctorReviewResponse)
async def get_doctor_review(session_id: str, request: Request) -> DoctorReviewResponse:
    session_store, patient_registry = _runtime_dependencies(request)
    meta = session_store.get_session(session_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if meta.scene != "doctor":
        raise HTTPException(status_code=409, detail="NOT_DOCTOR_SESSION")
    if meta.patient_id is None:
        raise HTTPException(status_code=409, detail="PATIENT_BINDING_REQUIRED")

    try:
        patient_registry.get_patient_detail(meta.patient_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Patient not found") from exc

    rows = patient_registry.list_patient_records(meta.patient_id)
    assertions = project_clinical_assertions_from_records(rows)
    return DoctorReviewResponse(
        patient_id=int(meta.patient_id),
        session_id=meta.session_id,
        timeline=[_timeline_item(row) for row in rows],
        assertions=[assertion.to_dict() for assertion in assertions],
        draft=_build_draft(int(meta.patient_id), assertions),
        available_actions=list(AVAILABLE_ACTIONS),
    )
