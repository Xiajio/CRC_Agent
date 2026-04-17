from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.nodes.node_utils import _sanitize_visible_response, _split_inline_thinking


def sanitize_user_visible_content(content: Any) -> Any:
    if isinstance(content, str):
        return _sanitize_visible_response(content)

    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        sanitized: list[Any] = []
        for item in content:
            if isinstance(item, str):
                sanitized.append(_sanitize_visible_response(item))
                continue

            if isinstance(item, Mapping):
                payload = dict(item)
                text = payload.get("text")
                if isinstance(text, str):
                    payload["text"] = _sanitize_visible_response(text)
                sanitized.append(payload)
                continue

            sanitized.append(item)

        return sanitized

    return content
