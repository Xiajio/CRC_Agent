from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from src.nodes import chat_main_node
from src.state import CRCAgentState


PATIENT_ID_KEY = "\u53d7\u8bd5\u8005\u7f16\u53f7"
GENDER_KEY = "\u6027\u522b"
AGE_KEY = "\u5e74\u9f84\uff08\u5177\u4f53\uff09"


class _UnusedModel:
    def bind_tools(self, _tools):
        return self

    def invoke(self, _payload):
        raise AssertionError("LLM fallback should not run in chat_main database regression tests.")


class _FakeTool:
    def __init__(self, name: str, result) -> None:
        self.name = name
        self._result = result
        self.calls: list[dict[str, object]] = []

    def invoke(self, payload: dict[str, object]):
        self.calls.append(payload)
        if callable(self._result):
            return self._result(payload)
        return self._result


def test_chat_main_uses_patient_lookup_tool_and_sets_pending_confirmation(monkeypatch) -> None:
    existing_case = {
        "patient_id": 93,
        "gender": 1,
        "age": 57,
        "tumor_location": "sigmoid",
    }
    get_patient_case_info = _FakeTool("get_patient_case_info", existing_case)
    upsert_patient_info = _FakeTool("upsert_patient_info", {"status": "success"})
    monkeypatch.setattr(
        chat_main_node,
        "CHAT_MAIN_TOOLS",
        [get_patient_case_info, upsert_patient_info],
    )

    runnable = chat_main_node.node_chat_main(model=_UnusedModel(), show_thinking=False)
    result = runnable(
        CRCAgentState(
            messages=[HumanMessage(content="\u6211\u7684ID\u662f93")],
            findings={},
        )
    )

    assert get_patient_case_info.calls == [{"patient_id": 93}]
    assert result["current_patient_id"] == "093"
    assert result["findings"]["pending_patient_id"] == "93"
    assert result["findings"]["pending_patient_data"] == existing_case


def test_chat_main_skips_pending_confirmation_when_patient_lookup_is_missing(monkeypatch) -> None:
    get_patient_case_info = _FakeTool(
        "get_patient_case_info",
        {"error": "not found", "patient_id": 93},
    )
    upsert_patient_info = _FakeTool("upsert_patient_info", {"status": "success"})
    monkeypatch.setattr(
        chat_main_node,
        "CHAT_MAIN_TOOLS",
        [get_patient_case_info, upsert_patient_info],
    )

    runnable = chat_main_node.node_chat_main(model=_UnusedModel(), show_thinking=False)
    result = runnable(
        CRCAgentState(
            messages=[HumanMessage(content="\u60a3\u800593")],
            findings={},
        )
    )

    assert get_patient_case_info.calls == [{"patient_id": 93}]
    assert result["current_patient_id"] == "093"
    assert "pending_patient_id" not in result["findings"]
    assert "pending_patient_data" not in result["findings"]
    assert result["findings"]["patient_record"][PATIENT_ID_KEY] == 93
    assert result["findings"]["active_field"] == GENDER_KEY


def test_chat_main_auto_saves_collected_fields_when_patient_id_is_known(monkeypatch) -> None:
    upsert_patient_info = _FakeTool("upsert_patient_info", {"status": "success"})
    monkeypatch.setattr(chat_main_node, "CHAT_MAIN_TOOLS", [upsert_patient_info])

    runnable = chat_main_node.node_chat_main(model=_UnusedModel(), show_thinking=False)
    result = runnable(
        CRCAgentState(
            messages=[HumanMessage(content="57")],
            findings={
                "active_field": AGE_KEY,
                "patient_record": {
                    PATIENT_ID_KEY: 93,
                    GENDER_KEY: 0,
                    AGE_KEY: 0,
                },
            },
        )
    )

    assert len(upsert_patient_info.calls) == 1
    saved_payload = json.loads(upsert_patient_info.calls[0]["json_data"])
    assert saved_payload[PATIENT_ID_KEY] == 93
    assert saved_payload[AGE_KEY] == 57
    assert result["findings"]["patient_record"][AGE_KEY] == 57
