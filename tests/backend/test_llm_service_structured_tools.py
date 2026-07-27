from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.services.llm_service import create_compatible_chat_openai


def test_compatible_chat_model_disables_thinking_for_tool_calls() -> None:
    model = create_compatible_chat_openai(
        model="deepseek-v4-flash",
        api_key="test-only-key",
        base_url="https://example.invalid/v1",
        provider_hint="deepseek",
    )
    tool = {
        "type": "function",
        "function": {
            "name": "record_candidate",
            "description": "Record a structured candidate.",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    payload = model._get_request_payload(  # type: ignore[attr-defined]
        [HumanMessage(content="Return a structured candidate")],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "record_candidate"}},
    )

    assert payload["extra_body"]["thinking"] == {"type": "disabled"}
    assert "enable_thinking" not in payload["extra_body"]
    assert payload["tools"] == [tool]


def test_openai_tool_calls_do_not_receive_vendor_thinking_parameters() -> None:
    model = create_compatible_chat_openai(
        model="gpt-4o-mini",
        api_key="test-only-key",
        base_url="https://api.openai.com/v1",
        provider_hint="openai",
    )
    tool = {
        "type": "function",
        "function": {
            "name": "record_candidate",
            "description": "Record a structured candidate.",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    payload = model._get_request_payload(  # type: ignore[attr-defined]
        [HumanMessage(content="Return a structured candidate")],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "record_candidate"}},
    )

    extra_body = payload.get("extra_body", {})
    assert "thinking" not in extra_body
    assert "enable_thinking" not in extra_body
    assert "thinking_budget" not in extra_body
    assert payload["tools"] == [tool]
