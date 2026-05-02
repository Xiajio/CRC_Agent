from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src.nodes.node_utils import _invoke_with_streaming, _truncate_message_history, clear_stream_callback, set_stream_callback


class _StreamingChain:
    def stream(self, context: dict):
        yield AIMessage(content="Hello ")
        yield AIMessage(content="world")


class _ReasoningStreamingChain:
    def stream(self, context: dict):
        yield AIMessage(content="根据系统提示，我应该先分析。\n\n")
        yield AIMessage(content="最终回复：您好")


class _TaggedReasoningStreamingChain:
    def stream(self, context: dict):
        yield AIMessage(content="<think>\nThe user has typed a string of question marks.")
        yield AIMessage(content="\n\nI should keep the reply short.\n\n")
        yield AIMessage(content="【长期记忆（摘要）】您好！")
        yield AIMessage(content="我专注于结直肠癌诊疗相关问题。")


def test_invoke_with_streaming_emits_stable_message_id_and_matching_final_message() -> None:
    callback_events: list[dict[str, object]] = []
    token = set_stream_callback(callback_events.append)
    try:
        result = _invoke_with_streaming(
            _StreamingChain(),
            {},
            streaming=True,
            show_thinking=False,
        )
    finally:
        clear_stream_callback(token)

    assert [event["type"] for event in callback_events] == ["start", "raw_first_chunk", "delta", "delta", "end"]

    message_ids = {event.get("message_id") for event in callback_events}
    assert len(message_ids) == 1
    message_id = next(iter(message_ids))
    assert isinstance(message_id, str) and message_id

    assert [event.get("delta") for event in callback_events if event["type"] == "delta"] == [
        "Hello ",
        "world",
    ]
    assert result.id == message_id
    assert result.content == "Hello world"


def test_invoke_with_streaming_only_emits_visible_response_deltas_and_preserves_thinking() -> None:
    callback_events: list[dict[str, object]] = []
    token = set_stream_callback(callback_events.append)
    try:
        result = _invoke_with_streaming(
            _ReasoningStreamingChain(),
            {},
            streaming=True,
            show_thinking=False,
        )
    finally:
        clear_stream_callback(token)

    assert [event["type"] for event in callback_events] == ["start", "raw_first_chunk", "delta", "end"]
    assert [event.get("delta") for event in callback_events if event["type"] == "delta"] == ["您好"]
    assert result.content == "您好"
    assert result.additional_kwargs.get("thinking_content") == "根据系统提示，我应该先分析。"


def test_invoke_with_streaming_strips_thinking_prefix_before_summary_marker() -> None:
    callback_events: list[dict[str, object]] = []
    token = set_stream_callback(callback_events.append)
    try:
        result = _invoke_with_streaming(
            _TaggedReasoningStreamingChain(),
            {},
            streaming=True,
            show_thinking=False,
        )
    finally:
        clear_stream_callback(token)

    assert [event["type"] for event in callback_events] == ["start", "raw_first_chunk", "delta", "delta", "end"]
    assert [event.get("delta") for event in callback_events if event["type"] == "delta"] == [
        "您好！",
        "我专注于结直肠癌诊疗相关问题。",
    ]
    assert result.content == "您好！我专注于结直肠癌诊疗相关问题。"
    assert "question marks" in result.additional_kwargs.get("thinking_content", "")
    assert "长期记忆" not in result.content


def test_truncate_message_history_accepts_legacy_keyword_aliases() -> None:
    messages = [
        HumanMessage(content="first message " * 20),
        HumanMessage(content="second message " * 20),
        HumanMessage(content="third message " * 20),
    ]

    truncated = _truncate_message_history(
        messages,
        max_tokens=1,
        reserve_tokens=0,
        keep_last_n=2,
        max_chars_per_message=12,
    )

    assert len(truncated) == 2
    assert truncated[0].content == "second me..."
    assert truncated[1].content == "third mes..."
