from __future__ import annotations

import os

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from backend.api.services.session_store import InMemorySessionStore
from backend.api.services.upload_service import (
    UploadProcessingError,
    UploadSessionBusyError,
    UploadSessionNotFoundError,
    UploadValidationError,
    reserve_upload_session,
    store_session_upload,
)

router = APIRouter(prefix="/api", tags=["uploads"])

UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_UPLOAD_BYTES = int(os.getenv("API_UPLOAD_MAX_BYTES", str(25 * 1024 * 1024)))


def _get_runtime_dependency(request: Request) -> tuple[InMemorySessionStore, object, object]:
    runtime = getattr(request.app.state, "runtime", None)
    session_store = getattr(runtime, "session_store", None)
    assets_root = getattr(runtime, "assets_root", None)
    patient_registry = getattr(runtime, "patient_registry_service", None)
    if session_store is None or assets_root is None or patient_registry is None:
        raise HTTPException(status_code=503, detail="Runtime is not initialized")
    return session_store, assets_root, patient_registry


async def _read_upload_bytes(file: UploadFile) -> bytes:
    buffer = bytearray()
    while True:
        chunk = await file.read(UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > MAX_UPLOAD_BYTES:
            raise UploadValidationError(
                f"UPLOAD_TOO_LARGE: maximum size is {MAX_UPLOAD_BYTES} bytes"
            )
    return bytes(buffer)


@router.post("/sessions/{session_id}/uploads")
async def upload_session_file(
    session_id: str,
    request: Request,
    file: UploadFile = File(...),
):
    session_store, assets_root, patient_registry = _get_runtime_dependency(request)
    try:
        _, run_id = reserve_upload_session(session_store, session_id)
    except UploadSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    except UploadSessionBusyError as exc:
        raise HTTPException(status_code=409, detail="SESSION_BUSY") from exc

    try:
        file_bytes = await _read_upload_bytes(file)
        return store_session_upload(
            session_store=session_store,
            patient_registry=patient_registry,
            assets_root=assets_root,
            session_id=session_id,
            filename=file.filename or "upload.bin",
            content_type=file.content_type or "application/octet-stream",
            file_bytes=file_bytes,
            reserved_run_id=run_id,
        )
    except UploadSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    except UploadSessionBusyError as exc:
        raise HTTPException(status_code=409, detail="SESSION_BUSY") from exc
    except UploadProcessingError as exc:
        raise HTTPException(status_code=500, detail=str(exc) or "UPLOAD_FAILED") from exc
    except UploadValidationError as exc:
        detail = str(exc) or "UPLOAD_INVALID"
        status_code = 413 if detail.startswith("UPLOAD_TOO_LARGE") else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    finally:
        session_store.release_run_lock(session_id, run_id)
