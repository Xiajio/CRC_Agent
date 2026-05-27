"""Wrapper to run pytest with a project-local tmp root."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TMP_ROOT = REPO_ROOT / "tests" / "backend" / ".tmp" / "pytest"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env["PYTEST_DEBUG_TEMPROOT"] = str(TMP_ROOT)

result = subprocess.run(
    [sys.executable, "-m", "pytest", *sys.argv[1:]],
    cwd=str(REPO_ROOT),
    env=env,
)
sys.exit(result.returncode)
