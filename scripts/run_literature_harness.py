from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.literature_harness import build_literature_harness_run


LITERATURE_HARNESS_RUN_ID = "literature_harness_20260630_001"
CLAIM_PACK_PATH = ROOT / "tests" / "fixtures" / "literature_claim_pack_v0.json"


def run_literature_harness(
    *,
    output_root: str | Path = ROOT / "reports",
) -> Path:
    output_base = Path(output_root)
    literature_dir = output_base / "literature"
    literature_dir.mkdir(parents=True, exist_ok=True)

    claim_pack = _read_json(CLAIM_PACK_PATH)
    harness_run = build_literature_harness_run(
        run_id=LITERATURE_HARNESS_RUN_ID,
        claim_pack=claim_pack,
    )

    report_path = literature_dir / f"{LITERATURE_HARNESS_RUN_ID}.json"
    _write_json(report_path, harness_run)
    return report_path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    report_file = run_literature_harness()
    print(f"Wrote {report_file}")
