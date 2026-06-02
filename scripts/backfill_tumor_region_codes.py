from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from backend.api.services.anatomy_region_map import resolve_region_codes
from backend.api.services.patient_registry_service import PatientRegistryService


def _primary_region_code(region_codes: list[str]) -> str | None:
    if len(region_codes) == 1:
        return region_codes[0]
    return None


def _load_json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return dict(parsed)
    return {}


def _derived_provenance(snapshot_provenance_json: Any, region_codes: list[str]) -> str:
    provenance = _load_json_mapping(snapshot_provenance_json)
    location_provenance = _load_json_mapping(provenance.get("tumor_location"))
    for derived_field in ("tumor_region_code", "tumor_region_codes"):
        if region_codes and location_provenance:
            provenance[derived_field] = {
                **location_provenance,
                "derived_from": "tumor_location",
            }
        else:
            provenance.pop(derived_field, None)
    return json.dumps(provenance, ensure_ascii=False)


def backfill_tumor_region_codes(db_path: str | Path) -> dict[str, int]:
    service = PatientRegistryService(db_path)
    scanned = 0
    updated = 0
    with sqlite3.connect(service.db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                id,
                tumor_location,
                tumor_region_code,
                tumor_region_codes_json,
                snapshot_provenance_json
            FROM patients
            ORDER BY id ASC
            """
        ).fetchall()
        for row in rows:
            scanned += 1
            region_codes = resolve_region_codes(row["tumor_location"])
            primary_code = _primary_region_code(region_codes)
            region_codes_json = json.dumps(region_codes, ensure_ascii=False)
            provenance_json = _derived_provenance(
                row["snapshot_provenance_json"],
                region_codes,
            )
            if (
                row["tumor_region_code"] == primary_code
                and row["tumor_region_codes_json"] == region_codes_json
                and row["snapshot_provenance_json"] == provenance_json
            ):
                continue
            connection.execute(
                """
                UPDATE patients
                SET
                    tumor_region_code = ?,
                    tumor_region_codes_json = ?,
                    snapshot_provenance_json = ?
                WHERE id = ?
                """,
                (primary_code, region_codes_json, provenance_json, int(row["id"])),
            )
            updated += 1
    return {"updated": updated, "scanned": scanned}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill structured tumor anatomy region codes.")
    parser.add_argument(
        "db_path",
        nargs="?",
        default=Path("runtime") / "patient_registry.db",
        help="Path to patient_registry.db",
    )
    args = parser.parse_args()
    result = backfill_tumor_region_codes(Path(args.db_path))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
