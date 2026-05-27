"""Wrapper to run vitest from PowerShell-hostile environment."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
FRONTEND = REPO_ROOT / "frontend"
NODE = Path(r"C:/Program Files/nodejs/node.exe")
VITEST_CLI = FRONTEND / "node_modules" / "vitest" / "dist" / "cli.js"

cmd = [str(NODE), str(VITEST_CLI), "run", *sys.argv[1:]]
result = subprocess.run(
    cmd,
    cwd=str(FRONTEND),
    capture_output=True,
    timeout=600,
)

sys.stdout.buffer.write(result.stdout)
sys.stderr.buffer.write(result.stderr)
sys.exit(result.returncode)
