from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import openpyxl

from src.services.case_excel_service import PREFERRED_CASE_HEADERS, normalize_case_payload


CLINICAL_CASE_DIR = "Clinical Case"
IMAGING_DIR = "Radiographic Imaging"


def _load_seed(seed_path: Path) -> dict[str, Any]:
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("seed file must contain a JSON object")
    return payload


def _validate_imaging_folder_name(folder_name: str) -> None:
    path = Path(folder_name)
    if path.is_absolute() or path.drive or path.root or len(path.parts) != 1 or path.name != folder_name:
        raise ValueError(f"invalid imaging folder name: {folder_name!r}")
    if folder_name in {".", ".."}:
        raise ValueError(f"invalid imaging folder name: {folder_name!r}")


def _resolve_seed_imaging_source(seed_root: Path, source: str) -> Path:
    source_path = (seed_root / source).resolve()
    try:
        source_path.relative_to(seed_root)
    except ValueError as exc:
        raise ValueError(f"invalid imaging source path: {source!r}") from exc
    if not source_path.exists():
        raise FileNotFoundError(f"missing imaging fixture: {source}")
    return source_path


def _materialize_classification_workbook(output_root: Path, rows: list[dict[str, Any]]) -> None:
    workbook_path = output_root / CLINICAL_CASE_DIR / "classification.xlsx"
    workbook_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("seed file must define at least one classification row")

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    headers = [PREFERRED_CASE_HEADERS[field] for field in PREFERRED_CASE_HEADERS]
    sheet.append(headers)

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("classification_rows entries must be JSON objects")
        normalized, extras = normalize_case_payload(row)
        if extras:
            unexpected_fields = ", ".join(sorted(extras))
            raise ValueError(f"classification_rows entries must not include extra fields: {unexpected_fields}")
        row_values = [normalized.get(field) for field in PREFERRED_CASE_HEADERS]
        sheet.append(row_values)

    workbook.save(workbook_path)


def _materialize_imaging_tree(seed_path: Path, output_root: Path, imaging: dict[str, Any]) -> None:
    imaging_root = output_root / IMAGING_DIR
    imaging_root.mkdir(parents=True, exist_ok=True)
    seed_root = seed_path.parent.resolve()

    for folder_name, sources in imaging.items():
        if not isinstance(folder_name, str) or not folder_name:
            raise ValueError("imaging folder names must be non-empty strings")
        _validate_imaging_folder_name(folder_name)
        if not isinstance(sources, list):
            raise ValueError(f"imaging entries for {folder_name!r} must be lists of relative paths")

        folder_target = imaging_root / folder_name
        folder_target.mkdir(parents=True, exist_ok=True)

        for source in sources:
            if not isinstance(source, str) or not source:
                raise ValueError(f"imaging entries for {folder_name!r} must be non-empty strings")
            source_path = _resolve_seed_imaging_source(seed_root, source)
            shutil.copyfile(source_path, folder_target / source_path.name)


def _stage_acceptance_case_db(
    *,
    seed_path: Path,
    classification_rows: list[dict[str, Any]],
    imaging: dict[str, Any],
    output_root: Path,
) -> Path:
    staging_root = output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex}"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)

    _materialize_classification_workbook(staging_root, classification_rows)
    _materialize_imaging_tree(seed_path, staging_root, imaging)
    return staging_root


def _commit_managed_subtrees(output_root: Path, staging_root: Path) -> None:
    commit_records: list[tuple[Path, Path | None]] = []

    try:
        for subtree_name in (CLINICAL_CASE_DIR, IMAGING_DIR):
            final_target = output_root / subtree_name
            staged_target = staging_root / subtree_name
            if not staged_target.exists():
                raise FileNotFoundError(f"missing staged subtree: {subtree_name}")

            backup_target: Path | None = None
            if final_target.exists():
                backup_target = output_root.parent / f".{output_root.name}.{subtree_name.replace(' ', '_').lower()}.backup-{uuid.uuid4().hex}"
                if backup_target.exists():
                    shutil.rmtree(backup_target)
                shutil.move(str(final_target), str(backup_target))

            commit_records.append((final_target, backup_target))
            shutil.move(str(staged_target), str(final_target))
    except Exception:
        for final_target, backup_target in reversed(commit_records):
            if final_target.exists():
                shutil.rmtree(final_target)
            if backup_target and backup_target.exists():
                shutil.move(str(backup_target), str(final_target))
        raise
    else:
        for _, backup_target in commit_records:
            if backup_target and backup_target.exists():
                shutil.rmtree(backup_target)


def materialize_acceptance_case_db(*, seed_path: Path, output_root: Path) -> Path:
    seed_path = Path(seed_path)
    output_root = Path(output_root)

    if not seed_path.exists():
        raise FileNotFoundError(seed_path)

    seed = _load_seed(seed_path)
    classification_rows = seed.get("classification_rows", [])
    imaging = seed.get("imaging", {})

    if not isinstance(classification_rows, list):
        raise ValueError("classification_rows must be a list")
    if not isinstance(imaging, dict):
        raise ValueError("imaging must be an object")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = _stage_acceptance_case_db(
        seed_path=seed_path,
        classification_rows=classification_rows,
        imaging=imaging,
        output_root=output_root,
    )
    try:
        _commit_managed_subtrees(output_root, staging_root)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
    return output_root
