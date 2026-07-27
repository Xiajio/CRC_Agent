from __future__ import annotations

import re


_APPARENT_IDENTIFIER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"),
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(
        r"(?i)(?:patient[ _-]?id|medical[ _-]?record|mrn|"
        r"患者(?:id|编号)|姓名|身份证(?:号)?|病历号|住院号|手机号|电话|电子邮箱)"
        r"\s*[:：=]\s*[^\s,，;；]+"
    ),
)


def validate_deidentified_text(field_name: str, value: str) -> str:
    """Reject common direct identifiers before text can leave the process."""

    for pattern in _APPARENT_IDENTIFIER_PATTERNS:
        if pattern.search(value):
            raise ValueError(
                f"{field_name} must not contain apparent patient identifiers "
                "or contact details"
            )
    return value


__all__ = ["validate_deidentified_text"]
