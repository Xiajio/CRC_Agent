from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts.harness import build_harness_run
from src.contracts.release_safety_report import build_release_safety_report
from src.services.clinical_safety_policy import load_clinical_safety_policy


HARNESS_RUN_ID = "harness_20260629_001"
RELEASE_REPORT_ID = "release_safety_20260629_001"
MUTATION_PACK_PATH = ROOT / "tests" / "fixtures" / "crc_mutation_pack_v0.json"


def run_crc_harness_replay(
    *,
    output_root: str | Path = ROOT / "reports",
) -> tuple[Path, Path]:
    output_base = Path(output_root)
    harness_dir = output_base / "harness"
    release_dir = output_base / "release_safety"
    harness_dir.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)

    mutation_pack = _read_json(MUTATION_PACK_PATH)
    policy = load_clinical_safety_policy()
    harness_run = build_harness_run(
        run_id=HARNESS_RUN_ID,
        mutation_pack=mutation_pack,
        policy=policy,
    )
    release_report = build_release_safety_report(
        report_id=RELEASE_REPORT_ID,
        harness_run=harness_run,
    )

    harness_path = harness_dir / f"{HARNESS_RUN_ID}.json"
    release_path = release_dir / f"{RELEASE_REPORT_ID}.json"
    _write_json(harness_path, harness_run)
    _write_json(release_path, release_report)
    return harness_path, release_path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    harness_file, release_file = run_crc_harness_replay()
    print(f"Wrote {harness_file}")
    print(f"Wrote {release_file}")
