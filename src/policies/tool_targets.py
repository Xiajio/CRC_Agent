"""Shared normalization helpers for pending-step target classification."""

from __future__ import annotations

from typing import Iterable


_TARGET_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("assessment", ("ask_user",)),
    ("case_database", ("database", "case_database", "case")),
    ("tool_executor", ("calculator", "tool_executor")),
    ("rad_agent", ("imaging", "ct", "mri", "tumor", "radiology")),
    ("path_agent", ("pathology", "clam", "biopsy")),
)

_DEFAULT_TARGET = "knowledge"
_ASSIGNEE_ALIASES: dict[str, str] = {
    "assessment": "assessment",
    "ask_user": "assessment",
    "knowledge": "knowledge",
    "web_search": "knowledge",
    "web": "knowledge",
    "case_database": "case_database",
    "database": "case_database",
    "rad_agent": "rad_agent",
    "imaging": "rad_agent",
    "path_agent": "path_agent",
    "pathology": "path_agent",
    "tool_executor": "tool_executor",
    "calculator": "tool_executor",
}


def classify_pending_step_target(tool_name: str, assignee: str = "") -> str:
    assignee_text = str(assignee or "").strip().lower()
    if assignee_text:
        return _ASSIGNEE_ALIASES.get(assignee_text, _DEFAULT_TARGET)

    tool = str(tool_name or "").strip().lower()
    if not tool:
        return ""

    for target, keywords in _TARGET_KEYWORDS:
        if _contains_any(tool, keywords):
            return target
    return _DEFAULT_TARGET


def _contains_any(tool_name: str, keywords: Iterable[str]) -> bool:
    return any(keyword in tool_name for keyword in keywords)


__all__ = ["classify_pending_step_target"]
