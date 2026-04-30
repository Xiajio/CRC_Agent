from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from backend.api.services.patient_registry_service import PatientRegistryService
from backend.api.services.session_store import InMemorySessionStore


class PatientContextStaleError(RuntimeError):
    """Raised when authoritative patient context cannot be resolved safely."""


class PatientContextResolver:
    def __init__(
        self,
        registry: PatientRegistryService,
        session_store: InMemorySessionStore,
    ) -> None:
        self._registry = registry
        self._session_store = session_store

    def resolve(self, session_id: str) -> dict[str, Any] | None:
        session = self._session_store.get_session(session_id)
        if session is None:
            raise PatientContextStaleError("PATIENT_CONTEXT_STALE: session not found")
        if session.patient_id is None:
            return None

        try:
            projection = self._registry.get_patient_context_projection(session.patient_id)
        except Exception as exc:
            raise PatientContextStaleError(
                "PATIENT_CONTEXT_STALE: projection unavailable"
            ) from exc

        context_state = session.context_state if isinstance(session.context_state, Mapping) else {}
        cache = context_state.get("patient_context_cache")
        if isinstance(cache, Mapping) and self._cache_matches_projection(
            cache,
            patient_id=session.patient_id,
            projection=projection,
        ):
            self._session_store.clear_legacy_medical_card(session_id)
            return deepcopy(dict(cache))

        self._session_store.set_patient_context_cache(session_id, projection)
        return deepcopy(projection)

    def _cache_matches_projection(
        self,
        cache: Mapping[str, Any],
        *,
        patient_id: int,
        projection: Mapping[str, Any],
    ) -> bool:
        return (
            cache.get("patient_id") == patient_id
            and cache.get("patient_version") == projection.get("patient_version")
            and cache.get("projection_version") == projection.get("projection_version")
        )
