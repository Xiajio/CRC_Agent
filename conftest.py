from __future__ import annotations

import os


os.environ.setdefault("AUTH_MODE", "bearer")
os.environ.setdefault("API_BEARER_TOKEN", "test-token")
os.environ.setdefault("API_ADMIN_BEARER_TOKEN", "test-admin-token")
