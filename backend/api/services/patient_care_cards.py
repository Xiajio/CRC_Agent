from __future__ import annotations

import json
from typing import Any, Mapping


DEFAULT_CARE_CARDS = {
    "focusMetrics": [
        "如出现大量便血、黑便、持续腹痛或停止排气排便，请及时线下就医。",
    ],
    "periodicChecks": [
        "完成专项问诊或上传报告后，系统会生成更具体的检查安排。",
    ],
    "dailyActions": [
        "记录症状出现时间、频率、颜色和诱因，便于复诊时说明。",
    ],
}

SIGNAL_LABELS = {
    "bleeding": "留意便血或黑便是否加重",
    "rectal_bleeding": "留意便血或黑便是否加重",
    "bowel_change": "记录排便习惯、次数或性状变化",
    "weight_loss": "关注体重下降、乏力或贫血表现",
    "fever": "观察发热、呕吐或腹痛加重",
    "obstruction": "如停止排气排便或腹胀加重，请及时线下就医",
    "massive_bleeding": "大量便血时请及时线下就医",
}


PATIENT_MESSAGE_PERIODIC_CHECKS = {
    "seek_emergency_care": "\u5982\u51fa\u73b0\u6301\u7eed\u52a0\u91cd\u8179\u75db\u3001\u505c\u6b62\u6392\u6c14\u6392\u4fbf\u6216\u53cd\u590d\u5455\u5410\uff0c\u8bf7\u7acb\u5373\u7ebf\u4e0b\u6025\u8bca\u8bc4\u4f30",
}


def _load_payload(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return parsed
    return {}


def _latest_crc_triage_payload(records: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for record in records:
        if record.get("record_type") != "crc_triage_assessment":
            continue
        payload = _load_payload(record.get("normalized_payload_json"))
        if payload:
            return payload
    return None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        candidate = item.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    return result


def build_patient_care_cards(records: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    payload = _latest_crc_triage_payload(records)
    if payload is None:
        return {key: list(value) for key, value in DEFAULT_CARE_CARDS.items()}

    signals = payload.get("known_crc_signals")
    if not isinstance(signals, Mapping):
        signals = {}
    red_flags = payload.get("red_flags")
    if not isinstance(red_flags, list):
        red_flags = []

    focus_metrics = [
        SIGNAL_LABELS.get(str(key), f"关注：{key}")
        for key, value in signals.items()
        if value is True
    ]
    focus_metrics.extend(SIGNAL_LABELS.get(str(flag), f"关注：{flag}") for flag in red_flags)

    missing_information = payload.get("missing_information")
    if isinstance(missing_information, list):
        focus_metrics.extend(f"补充{item}" for item in missing_information if isinstance(item, str) and item.strip())

    disposition = str(payload.get("disposition") or payload.get("next_step") or "")
    patient_message_key = str(payload.get("patient_message_key") or "")
    periodic_checks: list[str] = []
    if patient_message_key in PATIENT_MESSAGE_PERIODIC_CHECKS:
        periodic_checks.append(PATIENT_MESSAGE_PERIODIC_CHECKS[patient_message_key])
    if disposition in {"urgent_gi_clinic", "emergency", "mdt_or_specialist"}:
        periodic_checks.append("尽快预约消化专科门诊")

    suggested_tests = payload.get("suggested_tests")
    if isinstance(suggested_tests, list):
        periodic_checks.extend(
            f"准备或完成：{item}"
            for item in suggested_tests
            if isinstance(item, str) and item.strip()
        )

    daily_actions = [
        "记录便血颜色、次数和伴随症状",
        "复诊或上传报告时携带既往检查结果",
        "若症状明显加重，优先线下就医",
    ]

    return {
        "focusMetrics": _dedupe(focus_metrics) or list(DEFAULT_CARE_CARDS["focusMetrics"]),
        "periodicChecks": _dedupe(periodic_checks) or list(DEFAULT_CARE_CARDS["periodicChecks"]),
        "dailyActions": _dedupe(daily_actions),
    }


__all__ = ["build_patient_care_cards"]
