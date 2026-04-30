from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage

from backend.api.services.patient_commands import PatientCommandService
from backend.api.services.session_store import InMemorySessionStore, SessionMeta
from backend.api.services.upload_fixture_cards import load_fixture_upload_card
from src.services.document_converter import convert_uploaded_file


class UploadSessionNotFoundError(RuntimeError):
    pass


class UploadSessionBusyError(RuntimeError):
    pass


class UploadProcessingError(RuntimeError):
    pass


class UploadValidationError(RuntimeError):
    pass


def ensure_upload_session_available(session_store: InMemorySessionStore, session_id: str) -> SessionMeta:
    return _get_session_meta(session_store, session_id)


def sanitize_asset_filename(filename: str) -> str:
    candidate = Path(filename or "").name.strip()
    if not candidate:
        return "upload.bin"
    return candidate.replace("/", "_").replace("\\", "_")


def compute_file_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def convert_upload_to_medical_card(file_bytes: bytes, filename: str, file_type: str) -> dict[str, Any]:
    if os.getenv("UPLOAD_CONVERTER_MODE", "").strip().lower() == "fixture":
        return load_fixture_upload_card(filename)
    return convert_uploaded_file(
        file_bytes=file_bytes,
        filename=filename,
        file_type=file_type,
    )


def _card_to_dict(medical_card: Any) -> dict[str, Any]:
    if isinstance(medical_card, dict):
        return medical_card
    if hasattr(medical_card, "model_dump"):
        return dict(medical_card.model_dump())
    if hasattr(medical_card, "dict"):
        return dict(medical_card.dict())
    raise TypeError("Unsupported medical card payload")


def _has_value(value: Any) -> bool:
    return value not in {None, "", "not_provided", "Unknown", "pending_evaluation"}


def _has_snapshot_candidate_fields(card_payload: dict[str, Any]) -> bool:
    data = card_payload.get("data")
    if not isinstance(data, dict):
        return False

    patient_summary = data.get("patient_summary")
    diagnosis_block = data.get("diagnosis_block")
    staging_block = data.get("staging_block")

    candidate_blocks = (
        patient_summary if isinstance(patient_summary, dict) else {},
        diagnosis_block if isinstance(diagnosis_block, dict) else {},
        staging_block if isinstance(staging_block, dict) else {},
    )
    return any(
        _has_value(block.get(field))
        for block, fields in (
            (candidate_blocks[0], ("chief_complaint", "age", "gender")),
            (candidate_blocks[1], ("location", "mmr_status")),
            (candidate_blocks[2], ("clinical_stage", "t_stage", "n_stage", "m_stage")),
        )
        for field in fields
    )


def classify_upload_document(
    filename: str,
    card_payload: dict[str, Any],
    *,
    parse_failed: bool = False,
) -> str:
    if parse_failed:
        return "parse_failed"

    lowered_filename = (filename or "").lower()
    if any(keyword in lowered_filename for keyword in ("guideline", "guidelines", "education", "consensus", "handout", "brochure", "protocol")):
        return "guideline_or_education"
    if "pathology" in lowered_filename or "biopsy" in lowered_filename:
        return "pathology_report"
    if any(keyword in lowered_filename for keyword in ("imaging", "scan", "radiology", "ct", "mri", "pet")):
        return "imaging_report"

    payload_type = str(
        card_payload.get("document_type")
        or card_payload.get("type")
        or card_payload.get("card_type")
        or ""
    ).lower()
    if payload_type in {"pathology_card", "pathology_slide_card"}:
        return "pathology_report"
    if payload_type in {"imaging_card", "radiomics_report_card", "tumor_detection_card"}:
        return "imaging_report"
    if payload_type in {"patient_card", "patient_summary"}:
        return "patient_summary"
    if payload_type in {"medical_visualization_card", "medical_card", "decision_card", "triage_card"}:
        return "patient_report" if _has_snapshot_candidate_fields(card_payload) else "unknown"

    if _has_snapshot_candidate_fields(card_payload):
        return "patient_report"
    return "unknown"


def derive_ingest_decision(document_type: str) -> str:
    if document_type == "parse_failed":
        return "asset_only"
    if document_type in {"guideline_or_education", "unknown"}:
        return "record_only"
    return "record_and_snapshot"


def _normalize_age(value: Any) -> Any:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = "".join(character for character in value if character.isdigit())
        if digits:
            return int(digits)
    return value


def flatten_medical_card(medical_card: Any) -> tuple[dict[str, Any], dict[str, Any], str]:
    card_payload = _card_to_dict(medical_card)
    data = card_payload.get("data", {})
    patient_summary = data.get("patient_summary", {})
    diagnosis_block = data.get("diagnosis_block", {})
    staging_block = data.get("staging_block", {})
    key_findings = data.get("key_findings", [])

    patient_snapshot = {
        "chief_complaint": patient_summary.get("chief_complaint"),
        "age": _normalize_age(patient_summary.get("age")),
        "gender": patient_summary.get("gender"),
        "tumor_location": diagnosis_block.get("location"),
        "mmr_status": diagnosis_block.get("mmr_status"),
        "clinical_stage": staging_block.get("clinical_stage"),
        "t_stage": staging_block.get("t_stage"),
        "n_stage": staging_block.get("n_stage"),
        "m_stage": staging_block.get("m_stage"),
    }

    summary_parts: list[str] = []
    chief_complaint = patient_snapshot.get("chief_complaint")
    if _has_value(chief_complaint):
        summary_parts.append(str(chief_complaint))
    clinical_stage = patient_snapshot.get("clinical_stage")
    if _has_value(clinical_stage):
        summary_parts.append(str(clinical_stage))

    finding_texts = [
        str(finding.get("finding"))
        for finding in key_findings
        if isinstance(finding, dict) and _has_value(finding.get("finding"))
    ]
    if finding_texts:
        summary_parts.append(", ".join(finding_texts[:2]))

    summary_text = "; ".join(summary_parts) or "Uploaded medical document"
    return patient_snapshot, card_payload, summary_text


def build_upload_reference_message(
    *,
    filename: str,
    patient_id: int,
    record_id: int,
    medical_card: Any,
    document_type: str | None = None,
    ingest_decision: str | None = None,
) -> HumanMessage:
    patient_snapshot, _, summary_text = flatten_medical_card(medical_card)
    message_parts = [
        f'User uploaded "{filename}".',
        f"Saved for patient_id={patient_id} as record_id={record_id}.",
    ]
    if document_type:
        message_parts.append(f"Document type: {document_type}.")
    if ingest_decision:
        message_parts.append(f"Ingest decision: {ingest_decision}.")
    clinical_stage = patient_snapshot.get("clinical_stage")
    if _has_value(clinical_stage):
        message_parts.append(f"Clinical stage: {clinical_stage}.")
    _, card_payload, _ = flatten_medical_card(medical_card)
    key_findings = card_payload.get("data", {}).get("key_findings", [])
    findings_text = [
        str(finding.get("finding"))
        for finding in key_findings
        if isinstance(finding, dict) and _has_value(finding.get("finding"))
    ]
    if findings_text:
        message_parts.append(f"Key findings: {'; '.join(findings_text[:2])}.")
    elif summary_text:
        message_parts.append(f"Summary: {summary_text}.")
    return HumanMessage(content=" ".join(message_parts))


def _cleanup_asset_root(session_assets_root: Path, asset_root: Path) -> None:
    shutil.rmtree(asset_root, ignore_errors=True)
    if session_assets_root.exists():
        try:
            next(session_assets_root.iterdir())
        except StopIteration:
            session_assets_root.rmdir()


def _cleanup_empty_stable_roots(stable_original_root: Path, stable_derived_root: Path, patient_root: Path) -> None:
    for path in (stable_derived_root, stable_original_root, stable_original_root.parent, patient_root):
        if path.exists():
            try:
                path.rmdir()
            except OSError:
                pass


def _response_payload(asset_record: dict[str, Any], *, reused: bool) -> dict[str, Any]:
    return {
        "asset_id": asset_record["asset_id"],
        "filename": asset_record["filename"],
        "content_type": asset_record["content_type"],
        "size": asset_record["size"],
        "sha256": asset_record["sha256"],
        "reused": reused,
        "derived": dict(asset_record.get("derived", {})),
    }


def _require_session_meta(session_store: InMemorySessionStore, session_id: str) -> SessionMeta:
    meta = session_store.get_session(session_id)
    if meta is None:
        raise UploadSessionNotFoundError(f"Session not found: {session_id}")
    return meta


def _get_session_meta(session_store: InMemorySessionStore, session_id: str) -> SessionMeta:
    meta = _require_session_meta(session_store, session_id)
    if meta.active_run_id is not None:
        raise UploadSessionBusyError(f"Session is busy: {session_id}")
    return meta


def reserve_upload_session(session_store: InMemorySessionStore, session_id: str) -> tuple[SessionMeta, str]:
    meta = _require_session_meta(session_store, session_id)
    run_id = f"upload_{uuid4().hex}"
    if not session_store.try_acquire_run_lock(session_id, run_id):
        raise UploadSessionBusyError(f"Session is busy: {session_id}")
    return meta, run_id


def store_session_upload(
    *,
    session_store: InMemorySessionStore,
    patient_commands: PatientCommandService,
    assets_root: Path,
    session_id: str,
    filename: str,
    content_type: str,
    file_bytes: bytes,
    reserved_run_id: str | None = None,
) -> dict[str, Any]:
    acquired_here = False
    if reserved_run_id is None:
        meta, run_id = reserve_upload_session(session_store, session_id)
        acquired_here = True
    else:
        meta = _require_session_meta(session_store, session_id)
        run_id = reserved_run_id
        if meta.active_run_id != reserved_run_id:
            raise UploadSessionBusyError(f"Session is busy: {session_id}")

    normalized_filename = sanitize_asset_filename(filename)
    normalized_content_type = content_type or "application/octet-stream"
    sha256 = compute_file_sha256(file_bytes)
    patient_id = meta.patient_id
    if patient_id is None:
        raise UploadValidationError("PATIENT_ID_REQUIRED")
    processed_key = f"{patient_id}:{sha256}"
    try:
        processed = meta.processed_files.get(processed_key)
        if isinstance(processed, dict):
            existing_asset_id = processed.get("asset_id")
            if existing_asset_id is not None:
                existing_asset = meta.uploaded_assets.get(str(existing_asset_id))
                if isinstance(existing_asset, dict):
                    return _response_payload(existing_asset, reused=True)
            meta.processed_files.pop(processed_key, None)

        patient_asset_root = assets_root / str(patient_id)
        stable_original_root = patient_asset_root / sha256 / "original"
        stable_derived_root = patient_asset_root / sha256 / "derived"
        context_message: HumanMessage | None = None
        document_type = "parse_failed"
        ingest_decision = "asset_only"
        record_payload: dict[str, Any] = {
            "document_type": document_type,
            "filename": normalized_filename,
            "parse_status": "failed",
        }
        summary_text = f"Parse failed for {normalized_filename}"
        patient_snapshot: dict[str, Any] = {}
        record_id: int | None = None

        try:
            stable_original_root.mkdir(parents=True, exist_ok=True)
            stable_derived_root.mkdir(parents=True, exist_ok=True)

            original_path = stable_original_root / normalized_filename
            original_existed = original_path.exists()
            original_path.write_bytes(file_bytes)
            upload_result = patient_commands.record_upload_received(
                patient_id=patient_id,
                filename=normalized_filename,
                content_type=normalized_content_type,
                size_bytes=len(file_bytes),
                sha256=sha256,
                storage_path=str(original_path),
                source_session_id=session_id,
            )
            asset_id = str(upload_result.asset_id)
            if upload_result.reused and not original_existed:
                original_path.unlink(missing_ok=True)
                _cleanup_empty_stable_roots(stable_original_root, stable_derived_root, patient_asset_root)

            try:
                medical_card = convert_upload_to_medical_card(file_bytes, normalized_filename, normalized_content_type)
            except Exception as exc:
                failed_result = patient_commands.record_upload_parse_failed(
                    patient_id=patient_id,
                    asset_id=int(upload_result.asset_id),
                    error_code="CONVERTER_ERROR",
                    error_message=str(exc),
                    source_session_id=session_id,
                )
                record_payload = {
                    "document_type": "parse_failed",
                    "filename": normalized_filename,
                    "parse_status": "failed",
                    "error": str(exc),
                }
                derived_payload = {
                    "document_type": "parse_failed",
                    "ingest_decision": "asset_only",
                    "medical_card_created": False,
                    "record_id": None,
                    "patient_id": patient_id,
                    "patient_version": failed_result.patient_version,
                }
                (stable_derived_root / "parse_failed.json").write_text(
                    json.dumps(record_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                asset_record = {
                    "asset_id": asset_id,
                    "record_id": None,
                    "patient_id": patient_id,
                    "filename": normalized_filename,
                    "content_type": normalized_content_type,
                    "size": len(file_bytes),
                    "sha256": sha256,
                    "reused": bool(upload_result.reused or failed_result.reused),
                    "derived": derived_payload,
                }
                meta.uploaded_assets[asset_id] = asset_record
                meta.processed_files[processed_key] = {
                    "asset_id": asset_id,
                    "record_id": None,
                }
                session_store.bump_snapshot_version(session_id)
                return _response_payload(asset_record, reused=False)

            card_payload = _card_to_dict(medical_card)
            document_type = classify_upload_document(normalized_filename, card_payload)
            ingest_decision = derive_ingest_decision(document_type)
            if ingest_decision == "record_and_snapshot":
                patient_snapshot, record_payload, summary_text = flatten_medical_card(medical_card)
            else:
                _, record_payload, summary_text = flatten_medical_card(medical_card)
                patient_snapshot = {}

            record_payload = dict(record_payload)
            record_payload["document_type"] = document_type
            record_payload["ingest_decision"] = ingest_decision
            (stable_derived_root / "medical_card.json").write_text(
                json.dumps(record_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            if ingest_decision == "asset_only":
                asset_record = {
                    "asset_id": asset_id,
                    "record_id": None,
                    "patient_id": patient_id,
                    "filename": normalized_filename,
                    "content_type": normalized_content_type,
                    "size": len(file_bytes),
                    "sha256": sha256,
                    "reused": bool(upload_result.reused),
                    "derived": {
                        "medical_card_created": False,
                        "document_type": document_type,
                        "ingest_decision": ingest_decision,
                        "record_id": None,
                        "patient_id": patient_id,
                        "patient_version": upload_result.patient_version,
                    },
                }
                meta.uploaded_assets[asset_id] = asset_record
                meta.processed_files[processed_key] = {
                    "asset_id": asset_id,
                    "record_id": None,
                }
                session_store.bump_snapshot_version(session_id)
                return _response_payload(asset_record, reused=bool(upload_result.reused))

            registry_write = patient_commands.record_medical_card_extracted(
                patient_id=patient_id,
                asset_id=int(upload_result.asset_id),
                patient_snapshot=patient_snapshot,
                record_payload=record_payload,
                summary_text=summary_text,
                document_type=document_type,
                ingest_decision=ingest_decision,
                source_session_id=session_id,
            )
            asset_id = str(registry_write.asset_id)
            record_id = int(registry_write.record_id)
            asset_record = {
                "asset_id": asset_id,
                "record_id": record_id,
                "patient_id": patient_id,
                "filename": normalized_filename,
                "content_type": normalized_content_type,
                "size": len(file_bytes),
                "sha256": sha256,
                "reused": bool(upload_result.reused or registry_write.reused),
                "derived": {
                    "medical_card_created": True,
                    "document_type": document_type,
                    "ingest_decision": ingest_decision,
                    "record_id": record_id,
                    "patient_id": patient_id,
                    "patient_version": registry_write.patient_version,
                },
            }
            context_message = build_upload_reference_message(
                filename=normalized_filename,
                patient_id=patient_id,
                record_id=record_id,
                medical_card=record_payload,
                document_type=document_type,
                ingest_decision=ingest_decision,
            )

            meta.uploaded_assets[asset_id] = asset_record
            meta.processed_files[processed_key] = {
                "asset_id": asset_id,
                "record_id": record_id,
            }
            session_store.enqueue_context_message(session_id, context_message)
            session_store.bump_snapshot_version(session_id)

            return _response_payload(asset_record, reused=bool(upload_result.reused or registry_write.reused))
        except Exception as exc:
            if "asset_id" in locals():
                meta.uploaded_assets.pop(asset_id, None)
            processed_value = meta.processed_files.get(processed_key)
            if isinstance(processed_value, dict) and processed_value.get("asset_id") == locals().get("asset_id"):
                meta.processed_files.pop(processed_key, None)
            if context_message is not None:
                meta.pending_context_messages = [
                    message for message in meta.pending_context_messages if message != context_message
                ]
            raise UploadProcessingError(str(exc)) from exc
    finally:
        if acquired_here:
            session_store.release_run_lock(session_id, run_id)
