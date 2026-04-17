from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage


def _default_fixtures_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "graph_ticks"


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    return str(content or "")


class FixtureGraphRunner:
    def __init__(self, fixtures_dir: Path | None = None, default_case: str = "database_case") -> None:
        self.fixtures_dir = fixtures_dir or _default_fixtures_dir()
        self.default_case = default_case
        self._states_by_thread: dict[str, dict[str, Any]] = {}

    def _load_case(self, case_name: str) -> dict[str, Any]:
        path = self.fixtures_dir / f"{case_name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Fixture graph case not found: {case_name}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _merge_state(self, state: dict[str, Any], node_output: Mapping[str, Any]) -> None:
        for key, value in node_output.items():
            if key == "messages" and isinstance(value, list):
                existing = state.setdefault("messages", [])
                if isinstance(existing, list):
                    existing.extend(copy.deepcopy(value))
                continue
            if key == "findings" and isinstance(value, Mapping):
                existing_findings = state.setdefault("findings", {})
                if isinstance(existing_findings, dict):
                    existing_findings.update(copy.deepcopy(dict(value)))
                continue
            state[key] = copy.deepcopy(value)

    async def astream(self, payload: Mapping[str, Any], config: Mapping[str, Any] | None = None):
        messages = payload.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, BaseMessage) and _message_text(message).strip() == "__force_graph_failure__":
                    raise RuntimeError("Forced graph failure for fixture runner")

        case_name = str(payload.get("fixture_case") or self.default_case)
        tick_delay_ms = int(payload.get("fixture_tick_delay_ms") or 0)
        case_payload = self._load_case(case_name)

        thread_id = "fixture-thread"
        if isinstance(config, Mapping):
            configurable = config.get("configurable")
            if isinstance(configurable, Mapping) and configurable.get("thread_id"):
                thread_id = str(configurable["thread_id"])

        state = copy.deepcopy(self._states_by_thread.get(thread_id, {}))
        if isinstance(messages, list):
            existing_messages = state.setdefault("messages", [])
            if isinstance(existing_messages, list):
                existing_messages.extend(copy.deepcopy(messages))

        for tick in case_payload.get("ticks", []):
            if tick_delay_ms > 0:
                await asyncio.sleep(tick_delay_ms / 1000)
            node_name = str(tick["node_name"])
            node_output = copy.deepcopy(tick["node_output"])
            if isinstance(node_output, Mapping):
                self._merge_state(state, node_output)
            yield {node_name: node_output}

        self._states_by_thread[thread_id] = state

    def get_state(self, thread_id: str) -> dict[str, Any]:
        return copy.deepcopy(self._states_by_thread.get(thread_id, {}))

    async def aclose(self) -> None:
        return None
