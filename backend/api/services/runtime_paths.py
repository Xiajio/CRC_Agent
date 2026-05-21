from __future__ import annotations

import os
from pathlib import Path


def resolve_runtime_root() -> Path:
    configured = os.getenv("LANGG_RUNTIME_ROOT", "").strip()
    return Path(configured) if configured else Path("runtime")
