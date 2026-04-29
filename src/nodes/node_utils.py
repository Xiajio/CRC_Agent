from __future__ import annotations

import json
import math
import re
from contextvars import ContextVar, Token
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from src.rag.evidence import evidence_to_references, extract_evidence_block, strip_evidence_block


_STREAM_CALLBACK: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar(
    "node_stream_callback",
    default=None,
)

_THINKING_TAG_PATTERN = re.compile(r"<think(?:ing)?>([\s\S]*?)</think(?:ing)?>", re.IGNORECASE)
_OPEN_THINKING_TAG_PATTERN = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
_VISIBLE_RESPONSE_MARKER_PATTERN = re.compile(
    r"(?:【?\s*长期记忆（摘要）\s*】?)|(?:【?\s*最终回复\s*】?)|(?:最终回复|最终答案|用户可见内容)\s*[:：]",
    re.IGNORECASE,
)
_FINAL_ANSWER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:final\s+answer|final\s+response|最终回复|最终答案|用户可见内容)\s*[:：]",
    re.IGNORECASE,
)
_INTERNAL_LINE_PATTERNS = (
    re.compile(r"^\s*\[Router\].*$", re.MULTILINE),
    re.compile(r"^\s*\[Intent\].*$", re.MULTILINE),
    re.compile(r"^\s*\[Planner\].*$", re.MULTILINE),
    re.compile(r"^\s*\[General Chat\].*$", re.MULTILINE),
    re.compile(r"^\s*\[Staging Router\].*$", re.MULTILINE),
    re.compile(r"^\s*\[WARNING\].*$", re.MULTILINE),
    re.compile(r"^\s*思考过程[:：].*$", re.MULTILINE),
)
_REASONING_PREFIXES = (
    "reasoning:",
    "analysis:",
    "thought:",
    "thinking:",
    "让我先分析",
    "我应该先",
    "需要先",
    "先分析一下",
    "根据系统提示",
)
_POSTOP_MARKERS = ("术后", "post-op", "postop", "anastomosis", "stoma")
_RETRIEVED_METADATA_PATTERN = re.compile(
    r"<retrieved_metadata>(.*?)</retrieved_metadata>",
    re.DOTALL | re.IGNORECASE,
)


class ThinkingColors:
    THINKING_HEADER = ""
    THINKING = ""
    RESPONSE = ""
    RESET = ""


@dataclass(frozen=True)
class ThinkingResult:
    thinking: str = ""
    response: str = ""


def set_stream_callback(callback: Callable[[dict[str, Any]], None]) -> Token:
    return _STREAM_CALLBACK.set(callback)


def clear_stream_callback(token: Token) -> None:
    _STREAM_CALLBACK.reset(token)


def _extract_text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseMessage):
        return _extract_text_content(value.content)
    if isinstance(value, Mapping):
        if "content" in value:
            return _extract_text_content(value.get("content"))
        if "text" in value:
            return _extract_text_content(value.get("text"))
        return json.dumps(dict(value), ensure_ascii=False, default=str)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "".join(_extract_text_content(item) for item in value)
    return str(value)


def _clean_json_string(value: str) -> str:
    text = (value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _clean_and_validate_json(value: str) -> Any | None:
    text = _clean_json_string(value)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_first_json_object(value: str) -> Any | None:
    text = value or ""
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        while start != -1:
            depth = 0
            for index in range(start, len(text)):
                char = text[index]
                if char == start_char:
                    depth += 1
                elif char == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : index + 1]
                        parsed = _clean_and_validate_json(candidate)
                        if parsed is not None:
                            return parsed
                        break
            start = text.find(start_char, start + 1)
    return None


def _unwrap_nested_json(payload: Any, required_keys: Iterable[str] | None = None) -> Any:
    required = {str(key) for key in (required_keys or [])}
    current = payload

    while isinstance(current, Mapping):
        if not required:
            if len(current) == 1:
                only_value = next(iter(current.values()))
                if isinstance(only_value, Mapping):
                    current = only_value
                    continue
            return dict(current)

        if required.issubset(set(current.keys())):
            return dict(current)

        next_mapping = None
        for value in current.values():
            if isinstance(value, Mapping) and required.intersection(set(value.keys())):
                next_mapping = value
                break
        if next_mapping is None:
            return dict(current)
        current = next_mapping

    return current


def _parse_thinking_tags(content: str) -> tuple[str, str]:
    text = content or ""
    thinking_parts: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        captured = match.group(1).strip()
        if captured:
            thinking_parts.append(captured)
        return ""

    visible = _THINKING_TAG_PATTERN.sub(_replace, text).strip("\n\r\t")
    thinking = "\n\n".join(part for part in thinking_parts if part)
    return thinking, visible


def _looks_like_reasoning(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    return any(stripped.startswith(prefix) for prefix in _REASONING_PREFIXES) or any(
        lowered.startswith(prefix)
        for prefix in ("reasoning:", "analysis:", "thought:", "thinking:")
    )


def _split_inline_thinking(content: str) -> tuple[str, str]:
    text = (content or "").strip("\n\r\t")
    if not text:
        return "", ""

    marker_match = _VISIBLE_RESPONSE_MARKER_PATTERN.search(text)
    if marker_match:
        thinking = _OPEN_THINKING_TAG_PATTERN.sub("", text[: marker_match.start()]).strip()
        response = text[marker_match.end() :].lstrip(":： \t\r\n")
        return thinking, response

    final_match = _FINAL_ANSWER_PATTERN.search(text)
    if final_match:
        thinking = text[: final_match.start()].strip()
        response = text[final_match.end() :].lstrip(":： \t\r\n")
        return thinking, response

    if _OPEN_THINKING_TAG_PATTERN.search(text):
        thinking = _OPEN_THINKING_TAG_PATTERN.sub("", text).strip()
        return thinking, ""

    if _looks_like_reasoning(text):
        parts = re.split(r"\n\s*\n", text, maxsplit=1)
        if len(parts) == 2:
            head, tail = parts[0].strip(), parts[1].strip()
            if head and tail and not _looks_like_reasoning(tail):
                return head, tail
        return text, ""

    return "", text


def _sanitize_visible_response(content: str) -> str:
    text = content or ""
    _, text = _parse_thinking_tags(text)
    for pattern in _INTERNAL_LINE_PATTERNS:
        text = pattern.sub("", text)
    text = text.strip("\n\r\t")

    parsed = _clean_and_validate_json(text)
    if isinstance(parsed, (dict, list)):
        return ""

    return re.sub(r"\n{3,}", "\n\n", text)


def _sanitize_message_parts(content: str) -> ThinkingResult:
    explicit_thinking, visible = _parse_thinking_tags(content or "")
    visible = _sanitize_visible_response(visible)
    inline_thinking, response = _split_inline_thinking(visible)
    thinking_parts = [part.strip() for part in (explicit_thinking, inline_thinking) if part and part.strip()]
    return ThinkingResult(
        thinking="\n\n".join(dict.fromkeys(thinking_parts)),
        response=response,
    )


def _extract_thinking_from_chunk(chunk: Any) -> tuple[str, str]:
    parts = _sanitize_message_parts(_extract_text_content(chunk))
    return parts.thinking, parts.response


def _visible_delta_suffix(previous_visible: str, current_visible: str) -> str:
    if not current_visible or current_visible == previous_visible:
        return ""
    if current_visible.startswith(previous_visible):
        return current_visible[len(previous_visible) :]
    prefix_len = 0
    max_prefix = min(len(previous_visible), len(current_visible))
    while prefix_len < max_prefix and previous_visible[prefix_len] == current_visible[prefix_len]:
        prefix_len += 1
    return current_visible[prefix_len:]


def _coerce_message_additional_kwargs(message: AIMessage) -> dict[str, Any]:
    kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(kwargs, Mapping):
        return dict(kwargs)
    return {}


def _ensure_message(message: Any) -> AIMessage:
    if isinstance(message, AIMessage):
        if isinstance(message.content, str):
            parts = _sanitize_message_parts(message.content)
            kwargs = _coerce_message_additional_kwargs(message)
            if parts.thinking:
                existing = (
                    kwargs.get("thinking_content")
                    or kwargs.get("reasoning_content")
                    or kwargs.get("thinking")
                    or ""
                )
                kwargs["thinking_content"] = "\n\n".join(
                    [value for value in (str(existing).strip(), parts.thinking) if value]
                ).strip()
            message.content = parts.response
            message.additional_kwargs = kwargs
        return message

    content = _extract_text_content(message)
    parts = _sanitize_message_parts(content)
    additional_kwargs = {"thinking_content": parts.thinking} if parts.thinking else {}
    return AIMessage(content=parts.response, additional_kwargs=additional_kwargs)


def _invoke_with_streaming(
    chain: Runnable,
    context: dict,
    streaming: bool = False,
    show_thinking: bool = True,
    node_name: str | None = None,
) -> AIMessage:
    del show_thinking

    if not streaming:
        if hasattr(chain, "invoke"):
            return _ensure_message(chain.invoke(context))
        return _ensure_message(chain(context))

    message_id = f"msg_{uuid4().hex}"
    callback = _STREAM_CALLBACK.get()
    if callback is not None:
        callback({"type": "start", "message_id": message_id, "node": node_name})

    raw_accumulated = ""
    visible_response = ""
    tool_calls: list[dict[str, Any]] = []
    first_raw_chunk_emitted = False

    try:
        stream_iterable = chain.stream(context)
        for chunk in stream_iterable:
            if callback is not None and not first_raw_chunk_emitted:
                callback({"type": "raw_first_chunk", "message_id": message_id, "node": node_name})
                first_raw_chunk_emitted = True
            raw_accumulated += _extract_text_content(chunk)
            parts = _sanitize_message_parts(raw_accumulated)
            delta = _visible_delta_suffix(visible_response, parts.response)
            visible_response = parts.response

            chunk_tool_calls = getattr(chunk, "tool_calls", None)
            if isinstance(chunk_tool_calls, list):
                tool_calls.extend(
                    [tool_call for tool_call in chunk_tool_calls if isinstance(tool_call, Mapping)]
                )

            if callback is not None and delta:
                callback(
                    {
                        "type": "delta",
                        "message_id": message_id,
                        "node": node_name,
                        "delta": delta,
                    }
                )
    except Exception:
        fallback = _ensure_message(chain.invoke(context) if hasattr(chain, "invoke") else chain(context))
        fallback.id = message_id
        if callback is not None:
            callback({"type": "end", "message_id": message_id, "node": node_name})
        return fallback

    parts = _sanitize_message_parts(raw_accumulated)
    additional_kwargs: dict[str, Any] = {}
    if parts.thinking:
        additional_kwargs["thinking_content"] = parts.thinking

    if callback is not None:
        callback({"type": "end", "message_id": message_id, "node": node_name})

    return AIMessage(
        content=parts.response,
        id=message_id,
        tool_calls=[
            dict(tool_call)
            for tool_call in tool_calls
            if tool_call.get("name") and tool_call.get("id")
        ],
        additional_kwargs=additional_kwargs,
    )


def _extract_and_update_references(content: str) -> tuple[str, list[dict[str, Any]]]:
    text = content or ""
    pattern = re.compile(r"\[(\d+)\]\s*source=([^\n]+)\n([\s\S]*?)(?=\n\[\d+\]\s*source=|\Z)")
    references: list[dict[str, Any]] = []
    for index_text, source, snippet in pattern.findall(text):
        references.append(
            {
                "index": int(index_text),
                "source": source.strip(),
                "snippet": snippet.strip(),
            }
        )
    cleaned = pattern.sub("", text).strip()
    return cleaned, references


def _extract_structured_evidence(content: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    text = content or ""
    evidence = extract_evidence_block(text)
    cleaned = strip_evidence_block(text)

    metadata_refs: list[dict[str, Any]] = []
    metadata_match = _RETRIEVED_METADATA_PATTERN.search(cleaned)
    if metadata_match:
        try:
            payload = json.loads(metadata_match.group(1))
            if isinstance(payload, list):
                metadata_refs = [item for item in payload if isinstance(item, dict)]
        except json.JSONDecodeError:
            metadata_refs = []
        cleaned = _RETRIEVED_METADATA_PATTERN.sub("", cleaned).strip()

    if evidence:
        return cleaned, evidence_to_references(evidence), evidence
    if metadata_refs:
        return cleaned, metadata_refs, []

    legacy_cleaned, legacy_refs = _extract_and_update_references(cleaned)
    return legacy_cleaned, legacy_refs, []


def _calculate_text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(a=left or "", b=right or "").ratio()


def _normalize_tool_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _build_tools_map(tools: Sequence[BaseTool] | Mapping[str, BaseTool]) -> dict[str, BaseTool]:
    if isinstance(tools, Mapping):
        return {str(name): tool for name, tool in tools.items() if isinstance(tool, BaseTool)}
    return {
        str(getattr(tool, "name", "")): tool
        for tool in tools
        if isinstance(tool, BaseTool) and getattr(tool, "name", None)
    }


def _select_tools(tools: Sequence[BaseTool], requested_names: Sequence[str]) -> list[BaseTool]:
    requested = {_normalize_tool_name(name) for name in requested_names if str(name).strip()}
    if not requested:
        return list(tools)
    selected: list[BaseTool] = []
    for tool in tools:
        name = _normalize_tool_name(getattr(tool, "name", ""))
        if any(request in name or name in request for request in requested):
            selected.append(tool)
    return selected


def _execute_tool_calls(
    tool_calls: Sequence[Mapping[str, Any]],
    tools_map: Mapping[str, BaseTool] | Sequence[BaseTool],
) -> list[ToolMessage]:
    normalized_tools = _build_tools_map(tools_map)
    results: list[ToolMessage] = []

    for tool_call in tool_calls:
        name = str(tool_call.get("name") or "").strip()
        tool = normalized_tools.get(name)
        tool_call_id = str(tool_call.get("id") or "tool_call")
        if tool is None:
            results.append(
                ToolMessage(
                    content=f"Tool not found: {name}",
                    tool_call_id=tool_call_id,
                    name=name,
                )
            )
            continue

        args = tool_call.get("args") or {}
        try:
            output = tool.invoke(args)
        except Exception as exc:
            output = f"Tool execution failed: {exc}"

        results.append(
            ToolMessage(
                content=_extract_text_content(output),
                tool_call_id=tool_call_id,
                name=name,
            )
        )

    return results


def _execute_tool_calls_robust(
    tool_calls: Sequence[Mapping[str, Any]],
    tools_map: Mapping[str, BaseTool] | Sequence[BaseTool],
    max_retries: int = 2,
) -> list[ToolMessage]:
    normalized_tools = _build_tools_map(tools_map)
    results: list[ToolMessage] = []
    for tool_call in tool_calls:
        remaining = max_retries + 1
        last_error = "Tool execution failed."
        while remaining > 0:
            remaining -= 1
            try:
                results.extend(_execute_tool_calls([tool_call], normalized_tools))
                break
            except Exception as exc:
                last_error = str(exc)
        else:
            results.append(
                ToolMessage(
                    content=last_error,
                    tool_call_id=str(tool_call.get("id") or "tool_call"),
                    name=str(tool_call.get("name") or ""),
                )
            )
    return results


def _build_fallback_search_query(text: str, max_terms: int = 6) -> str:
    tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", text or "")
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(token)
        if len(deduped) >= max_terms:
            break
    return " ".join(deduped)


def _user_text(value: Any) -> str:
    if isinstance(value, BaseMessage):
        return _extract_text_content(value.content)
    if isinstance(value, Mapping):
        return _extract_text_content(value.get("content") or value.get("text") or "")
    return _extract_text_content(value)


def _latest_user_text(state: Any) -> str:
    messages = getattr(state, "messages", None) or []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _extract_text_content(message.content).strip()
        if isinstance(message, Mapping) and str(message.get("type") or "").lower() == "human":
            return _extract_text_content(message.get("content")).strip()
    return ""


def _estimate_tokens(value: Any) -> int:
    text = _extract_text_content(value)
    if not text:
        return 0
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    other_chars = max(0, len(text) - chinese_chars)
    return max(1, math.ceil(chinese_chars * 0.6 + other_chars / 4))


def _truncate_single_message(message: BaseMessage, max_chars: int = 1200) -> BaseMessage:
    content = _extract_text_content(message.content)
    if len(content) <= max_chars:
        return message
    trimmed = content[: max_chars - 3].rstrip() + "..."
    cloned = message.model_copy(deep=True)
    cloned.content = trimmed
    return cloned


def _truncate_message_history(
    messages: Sequence[BaseMessage],
    max_tokens: int = 6000,
    reserve_tokens: int = 500,
    keep_last: int = 8,
    keep_last_n: int | None = None,
    max_chars_per_message: int | None = None,
) -> list[BaseMessage]:
    if keep_last_n is not None:
        keep_last = keep_last_n

    budget = max(0, max_tokens - reserve_tokens)
    if budget <= 0:
        selected = list(messages[-keep_last:])
        if max_chars_per_message and max_chars_per_message > 0:
            return [_truncate_single_message(message, max_chars_per_message) for message in selected]
        return selected

    selected: list[BaseMessage] = []
    total = 0
    for message in reversed(list(messages)):
        token_cost = _estimate_tokens(getattr(message, "content", ""))
        if selected and total + token_cost > budget and len(selected) >= keep_last:
            break
        selected.append(message)
        total += token_cost
    result = list(reversed(selected))
    if max_chars_per_message and max_chars_per_message > 0:
        return [_truncate_single_message(message, max_chars_per_message) for message in result]
    return result


def _compress_rag_context(rag_context: Any, max_chars: int = 4000) -> str:
    text = _extract_text_content(rag_context).strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2].rstrip()
    tail = text[-max_chars // 2 :].lstrip()
    return f"{head}\n...\n{tail}"


def _create_rag_digest(
    *,
    rag_context: Any = "",
    references: Sequence[Mapping[str, Any]] | None = None,
    queries: Sequence[str] | None = None,
    max_digest_chars: int = 500,
) -> str:
    sections: list[str] = []
    if queries:
        sections.append(f"Queries: {', '.join(query for query in queries if query)}")
    if references:
        ref_preview = []
        for item in references[:3]:
            source = str(item.get("source") or item.get("title") or "").strip()
            if source:
                ref_preview.append(source)
        if ref_preview:
            sections.append(f"References: {', '.join(ref_preview)}")
    context_text = _compress_rag_context(rag_context, max_chars=max_digest_chars)
    if context_text:
        sections.append(f"Context: {context_text}")
    digest = "\n".join(section for section in sections if section).strip()
    return digest[:max_digest_chars]


def _create_tool_result_digest(results: Any, max_chars: int = 500) -> str:
    return _compress_rag_context(results, max_chars=max_chars)


def _profile_as_text(profile: Any) -> str:
    if profile is None:
        return ""
    if hasattr(profile, "model_dump"):
        profile = profile.model_dump()
    if isinstance(profile, Mapping):
        parts = []
        for key, value in profile.items():
            if value in (None, "", [], {}):
                continue
            parts.append(f"{key}: {value}")
        return "\n".join(parts)
    return str(profile)


def _build_pinned_context(state: Any) -> str:
    sections: list[str] = []
    patient_id = getattr(state, "current_patient_id", None)
    if patient_id:
        sections.append(f"patient_id: {patient_id}")
    profile_text = _profile_as_text(getattr(state, "patient_profile", None))
    if profile_text:
        sections.append(f"patient_profile:\n{profile_text}")
    findings = getattr(state, "findings", None)
    if isinstance(findings, Mapping):
        important = {
            key: value
            for key, value in dict(findings).items()
            if key
            in {
                "user_intent",
                "encounter_track",
                "clinical_entry_reason",
                "triage_current_field",
                "inquiry_message",
                "pathology_confirmed",
                "tumor_location",
            }
            and value not in (None, "", [], {})
        }
        if important:
            sections.append(f"findings: {json.dumps(important, ensure_ascii=False)}")
    return "\n\n".join(section for section in sections if section)


def _build_summary_memory(state: Any) -> str:
    summary_memory = getattr(state, "summary_memory", None)
    if isinstance(summary_memory, str) and summary_memory.strip():
        return summary_memory.strip()
    structured = getattr(state, "structured_summary", None)
    if hasattr(structured, "text_summary") and getattr(structured, "text_summary"):
        return str(structured.text_summary).strip()
    if isinstance(structured, Mapping):
        text_summary = structured.get("text_summary")
        if isinstance(text_summary, str):
            return text_summary.strip()
    return ""


def _build_profile_change_entry(
    previous_profile: Any,
    new_profile: Any,
    *,
    source: str = "",
) -> dict[str, Any] | None:
    before = previous_profile.model_dump() if hasattr(previous_profile, "model_dump") else previous_profile
    after = new_profile.model_dump() if hasattr(new_profile, "model_dump") else new_profile
    if not isinstance(before, Mapping):
        before = {}
    if not isinstance(after, Mapping):
        after = {}

    changes = {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in set(before.keys()) | set(after.keys())
        if before.get(key) != after.get(key)
    }
    if not changes:
        return None
    return {"source": source, "changes": changes}


def _invoke_structured_with_recovery(
    *,
    prompt: BasePromptTemplate | Runnable | None,
    model: Any,
    schema: Any,
    payload: Mapping[str, Any],
    log_prefix: str = "",
    fallback_factory: Callable[[Mapping[str, Any], Exception], Any] | None = None,
) -> Any:
    del log_prefix

    try:
        structured_model = model.with_structured_output(schema)
        chain = (prompt | structured_model) if prompt is not None else structured_model
        return chain.invoke(dict(payload))
    except Exception as exc:
        try:
            raw_chain = (prompt | model) if prompt is not None else model
            raw_response = raw_chain.invoke(dict(payload))
            parsed = _clean_and_validate_json(_extract_text_content(raw_response))
            if parsed is None:
                parsed = _extract_first_json_object(_extract_text_content(raw_response))
            if parsed is not None:
                normalized = _unwrap_nested_json(parsed, getattr(schema, "model_fields", {}).keys())
                if hasattr(schema, "model_validate"):
                    return schema.model_validate(normalized)
                return schema(**normalized)
        except Exception:
            pass

        if fallback_factory is not None:
            return fallback_factory(payload, exc)

        if hasattr(schema, "model_validate"):
            return schema.model_validate({})
        return schema()


def _is_postop_context(value: Any) -> bool:
    text = _extract_text_content(value)
    if not text and hasattr(value, "messages"):
        text = "\n".join(_extract_text_content(message.content) for message in getattr(value, "messages", []))
    lowered = text.lower()
    return any(marker in lowered for marker in _POSTOP_MARKERS)


def _generate_fallback_plan(state: Any) -> dict[str, Any]:
    return {
        "strategy": "fallback",
        "summary": "Insufficient structured evidence; recommend clinician review.",
        "patient_id": getattr(state, "current_patient_id", None),
    }


def _needs_full_decision(state: Any) -> bool:
    findings = getattr(state, "findings", None)
    if isinstance(findings, Mapping) and str(findings.get("user_intent") or "") == "treatment_decision":
        return True
    return not bool(getattr(state, "decision_json", None))


def _is_repeated_rejection(history: Sequence[Any], feedback: str) -> bool:
    normalized_feedback = (feedback or "").strip().lower()
    if not normalized_feedback:
        return False
    recent = [_extract_text_content(item).strip().lower() for item in history[-3:]]
    return sum(1 for item in recent if item and normalized_feedback in item) >= 2


def _format_messages_for_summary(messages: Sequence[BaseMessage], max_chars: int = 1000) -> str:
    lines: list[str] = []
    total = 0
    for message in messages:
        role = getattr(message, "type", message.__class__.__name__).lower()
        content = _extract_text_content(getattr(message, "content", ""))
        line = f"{role}: {content}".strip()
        total += len(line)
        if total > max_chars:
            remaining = max(0, max_chars - (total - len(line)))
            if remaining > 3:
                lines.append(line[: remaining - 3] + "...")
            break
        lines.append(line)
    return "\n".join(lines)


def auto_update_roadmap_from_state(state: Any) -> list[Any]:
    roadmap = getattr(state, "roadmap", None)
    return list(roadmap or [])


def _calculate_improvement(previous_text: str, current_text: str) -> float:
    previous = (previous_text or "").strip()
    current = (current_text or "").strip()
    if not current:
        return 0.0
    if not previous:
        return 1.0
    return max(0.0, 1.0 - _calculate_text_similarity(previous, current))


def _extract_ct_text(state: Any) -> str:
    findings = getattr(state, "findings", None)
    if isinstance(findings, Mapping):
        for key in ("ct_report", "ct_text", "ct_summary"):
            value = findings.get(key)
            if value:
                return _extract_text_content(value)
    return ""


def _extract_mri_text(state: Any) -> str:
    findings = getattr(state, "findings", None)
    if isinstance(findings, Mapping):
        for key in ("mri_report", "mri_text", "mri_summary"):
            value = findings.get(key)
            if value:
                return _extract_text_content(value)
    return ""


def _extract_pathology_text(state: Any) -> str:
    findings = getattr(state, "findings", None)
    if isinstance(findings, Mapping):
        for key in ("pathology_report", "pathology_text", "pathology_summary"):
            value = findings.get(key)
            if value:
                return _extract_text_content(value)
    return ""


__all__ = [
    "ThinkingColors",
    "ThinkingResult",
    "set_stream_callback",
    "clear_stream_callback",
    "_clean_json_string",
    "_clean_and_validate_json",
    "_unwrap_nested_json",
    "_extract_and_update_references",
    "_extract_structured_evidence",
    "_calculate_text_similarity",
    "_parse_thinking_tags",
    "_split_inline_thinking",
    "_sanitize_visible_response",
    "_extract_text_content",
    "_extract_first_json_object",
    "_extract_thinking_from_chunk",
    "_invoke_with_streaming",
    "_ensure_message",
    "_select_tools",
    "_execute_tool_calls",
    "_execute_tool_calls_robust",
    "_build_fallback_search_query",
    "_user_text",
    "_latest_user_text",
    "_estimate_tokens",
    "_truncate_single_message",
    "_truncate_message_history",
    "_compress_rag_context",
    "_create_rag_digest",
    "_create_tool_result_digest",
    "_build_pinned_context",
    "_build_summary_memory",
    "_build_profile_change_entry",
    "_invoke_structured_with_recovery",
    "_is_postop_context",
    "_generate_fallback_plan",
    "_needs_full_decision",
    "_is_repeated_rejection",
    "_format_messages_for_summary",
    "auto_update_roadmap_from_state",
    "_calculate_improvement",
    "_extract_ct_text",
    "_extract_mri_text",
    "_extract_pathology_text",
]


