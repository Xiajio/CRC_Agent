from __future__ import annotations

import subprocess
from pathlib import Path


P0_CRC_SAFETY_FILES = (
    "docs/safety/README.md",
    "docs/safety/intended_use.md",
    "tests/backend/test_intended_use_profiles.py",
    "tests/backend/test_clinical_safety_policy.py",
    "tests/backend/test_crc_triage_mutation_pack.py",
    "tests/backend/test_crc_triage_save.py",
    "tests/backend/test_crc_harness_replay.py",
    "tests/backend/test_crc_safety_gitignore_contract.py",
    "tests/fixtures/crc_mutation_pack_v0.json",
)


def test_p0_crc_safety_files_are_not_gitignored() -> None:
    root = Path(__file__).resolve().parents[2]

    ignored_paths: list[str] = []
    for path in P0_CRC_SAFETY_FILES:
        result = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "--quiet", "--", path],
            check=False,
        )
        if result.returncode == 0:
            ignored_paths.append(path)

    assert ignored_paths == []
