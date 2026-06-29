from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Literal, Mapping

from src.services.clinical_safety_policy import (
    evaluate_clinical_safety_policy,
    merge_policy_disposition,
)


CrcTriageStage = Literal[
    "identity",
    "vitals",
    "red_flags",
    "symptom_cluster",
    "differential",
    "tests",
    "final",
]

CRC_TRIAGE_STAGE_SEQUENCE: tuple[CrcTriageStage, ...] = (
    "identity",
    "vitals",
    "red_flags",
    "symptom_cluster",
    "differential",
    "tests",
    "final",
)


@dataclass(frozen=True)
class CrcTriageAnswer:
    question_id: str
    answer_text: str


@dataclass(frozen=True)
class _CrcTriageQuestion:
    id: str
    stage: CrcTriageStage
    text: str
    options: tuple[str, ...]
    askable: bool
    terminal: bool


_CRC_TRIAGE_QUESTION_DEFINITIONS: tuple[_CrcTriageQuestion, ...] = (
    _CrcTriageQuestion(
        id="vitals_shock_or_consciousness",
        stage="vitals",
        text="最近有没有出现头晕、眼前发黑、意识模糊，或者突然出冷汗、面色苍白的情况？",
        options=("没有", "有", "不清楚"),
        askable=True,
        terminal=False,
    ),
    _CrcTriageQuestion(
        id="vitals_heart_or_breathing",
        stage="vitals",
        text="最近有没有明显心慌、胸闷、喘不上气，或者呼吸比平时费力？",
        options=("没有", "有", "不清楚"),
        askable=True,
        terminal=False,
    ),
    _CrcTriageQuestion(
        id="red_flags_weight_or_bleeding",
        stage="red_flags",
        text="最近有没有便血、黑便、原因不明的体重下降，或者明显贫血？",
        options=("没有", "有", "不清楚"),
        askable=True,
        terminal=False,
    ),
    _CrcTriageQuestion(
        id="red_flags_pain_or_obstruction",
        stage="red_flags",
        text="有没有持续加重的腹痛、停止排气排便，或反复呕吐？",
        options=("没有", "有", "不清楚"),
        askable=True,
        terminal=False,
    ),
    _CrcTriageQuestion(
        id="symptom_cluster_chief",
        stage="symptom_cluster",
        text="这次最主要的不舒服是什么？比如腹痛、腹泻、便秘、便血、恶心呕吐等。",
        options=("腹痛腹胀", "排便改变", "出血或黑便", "恶心呕吐", "其他"),
        askable=True,
        terminal=False,
    ),
    _CrcTriageQuestion(
        id="differential_duration",
        stage="differential",
        text="这些症状持续多久了？整体是在好转、稳定，还是加重？",
        options=("正在好转", "基本稳定", "逐渐加重"),
        askable=True,
        terminal=False,
    ),
    _CrcTriageQuestion(
        id="tests_recent_exam",
        stage="tests",
        text="最近做过哪些相关检查？比如血常规、粪便潜血、肠镜、腹部CT或肿瘤标志物。",
        options=("没有做过", "做过但结果不清楚", "做过且有结果"),
        askable=True,
        terminal=False,
    ),
    _CrcTriageQuestion(
        id="final_ready",
        stage="final",
        text="我已经收集到关键情况，接下来会整理风险提示和建议就诊方向，可以吗？",
        options=("可以", "我还要补充", "暂时结束"),
        askable=False,
        terminal=True,
    ),
)

CRC_TRIAGE_QUESTIONS: tuple[Mapping[str, Any], ...] = tuple(
    MappingProxyType(
        {
            "id": question.id,
            "stage": question.stage,
            "text": question.text,
            "options": question.options,
            "askable": question.askable,
            "terminal": question.terminal,
        }
    )
    for question in _CRC_TRIAGE_QUESTION_DEFINITIONS
)


def _question_payload(question: _CrcTriageQuestion) -> dict[str, Any]:
    return {
        "id": question.id,
        "stage": question.stage,
        "text": question.text,
        "options": question.options,
        "askable": question.askable,
        "terminal": question.terminal,
    }


def _next_question(question_id: str) -> _CrcTriageQuestion | None:
    for index, question in enumerate(_CRC_TRIAGE_QUESTION_DEFINITIONS):
        if question.id == question_id:
            next_index = index + 1
            if next_index < len(_CRC_TRIAGE_QUESTION_DEFINITIONS):
                return _CRC_TRIAGE_QUESTION_DEFINITIONS[next_index]
            return None
    return None


def _vitals_result(qa_summary: list[dict[str, Any]]) -> dict[str, str]:
    abnormal = any(
        item["answer"] in {"有", "不清楚"}
        for item in qa_summary
        if item["stage"] == "vitals"
    )
    if abnormal:
        return {
            "stage": "vitals",
            "title": "节点1：生命体征评估",
            "risk_level": "需进一步确认生命体征风险",
            "summary": "患者报告或无法排除生命体征异常，需要继续筛查危险信号。",
            "next_step": "进入节点2：全系统危险信号筛查。",
        }
    return {
        "stage": "vitals",
        "title": "节点1：生命体征评估",
        "risk_level": "生命体征平稳",
        "summary": "未识别到意识异常、休克表现、明显心率或呼吸异常。",
        "next_step": "进入节点2：全系统危险信号筛查。",
    }


def _is_no_recent_test_answer(answer: Any) -> bool:
    normalized = str(answer or "").strip().replace(" ", "")
    if normalized in {
        "没有",
        "没有做过",
        "没有检查",
        "没做",
        "没做过",
        "未做",
        "未检查",
    }:
        return True
    return normalized.startswith(("没有做过", "没做过", "未做过"))


def _combined_answer_text(qa_summary: list[dict[str, Any]]) -> str:
    return "\n".join(str(item.get("answer") or "") for item in qa_summary)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    normalized = text.lower()
    return any(needle.lower() in normalized for needle in needles)


def _extract_age(text: str) -> int | None:
    for pattern in (
        r"\u5e74\u9f84\s*(\d{1,3})\s*\u5c81",
        r"age\s*(\d{1,3})",
        r"(\d{1,3})\s*\u5c81",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _answer_for_question(
    qa_summary: list[dict[str, Any]],
    question_id: str,
) -> str:
    for item in qa_summary:
        if item.get("question_id") == question_id:
            return str(item.get("answer") or "")
    return ""


def _is_affirmative_answer(answer: Any) -> bool:
    normalized = str(answer or "").strip().replace(" ", "")
    return normalized.startswith("有")


def _derive_safety_policy_input(qa_summary: list[dict[str, Any]]) -> dict[str, Any]:
    answer_text = _combined_answer_text(qa_summary)
    red_flags_pain_or_obstruction_answer = _answer_for_question(
        qa_summary,
        "red_flags_pain_or_obstruction",
    )
    symptom_cluster_chief_answer = _answer_for_question(
        qa_summary,
        "symptom_cluster_chief",
    )
    differential_duration_answer = _answer_for_question(
        qa_summary,
        "differential_duration",
    )
    tests_recent_exam_answer = next(
        (
            item.get("answer")
            for item in qa_summary
            if item.get("question_id") == "tests_recent_exam"
        ),
        None,
    )
    has_recent_tests = tests_recent_exam_answer is not None and not _is_no_recent_test_answer(
        tests_recent_exam_answer
    )
    pain_or_obstruction_positive = _is_affirmative_answer(
        red_flags_pain_or_obstruction_answer
    )
    worsening_abdominal_pain = _contains_any(
        f"{symptom_cluster_chief_answer}\n{differential_duration_answer}",
        ("腹痛", "腹胀", "逐渐加重"),
    )

    return {
        "age": _extract_age(answer_text),
        "rectal_bleeding": _contains_any(
            answer_text,
            ("便血", "出血", "rectal bleeding"),
        ),
        "weight_loss": _contains_any(
            answer_text,
            ("体重下降", "消瘦", "weight loss"),
        ),
        "vomiting": _contains_any(answer_text, ("呕吐", "vomiting")),
        "obstipation": pain_or_obstruction_positive
        or _contains_any(
            answer_text,
            ("停止排气排便", "停止排便", "obstipation"),
        ),
        "severe_abdominal_pain": pain_or_obstruction_positive
        or worsening_abdominal_pain
        or _contains_any(
            answer_text,
            ("剧烈腹痛", "持续加重的腹痛", "严重腹痛"),
        ),
        "user_explanation": answer_text,
        "endoscopy_status": "available" if has_recent_tests else None,
        "fecal_occult_blood_test": "available" if has_recent_tests else None,
    }


def _risk_level_after_disposition(current_risk_level: str, disposition: str) -> str:
    if disposition in {"emergency", "urgent_gi_clinic", "urgent"}:
        return "high"
    return current_risk_level


def _default_assessment_id(assessment: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in dict(assessment).items()
        if key != "assessment_id"
    }
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return f"crc_assessment_{sha256(payload_json.encode('utf-8')).hexdigest()[:12]}"


def _build_final_assessment(state: dict[str, Any]) -> dict[str, Any]:
    qa_summary = list(state.get("qa_summary") or [])
    chief_complaint = next(
        (
            item["answer"]
            for item in qa_summary
            if item.get("question_id") == "symptom_cluster_chief"
        ),
        "患者完成 CRC 专项预问诊",
    )
    has_tests = any(
        item.get("question_id") == "tests_recent_exam"
        and not _is_no_recent_test_answer(item.get("answer"))
        for item in qa_summary
    )
    assessment = {
        "record_type": "crc_triage_assessment",
        "chief_complaint": chief_complaint,
        "symptom_group": "CRC相关门诊分诊",
        "risk_level": "medium" if not has_tests else "high",
        "disposition": "complete_basic_tests" if not has_tests else "urgent_gi_clinic",
        "red_flags": [],
        "known_crc_signals": {},
        "suggested_tests": ["血常规", "粪便潜血", "肠镜或结肠镜相关检查"],
        "missing_information": [] if has_tests else ["内镜或粪便潜血等辅助检查结果"],
        "qa_summary": qa_summary,
        "patient_summary": "已完成 CRC 专项预问诊，建议结合辅助检查结果进一步判断。",
        "next_step": "上传或携带近期检查结果，必要时预约消化专科门诊。",
        "source_session_id": "",
        "source_subflow": "crc_triage",
        "node_results": list(state.get("node_results") or []),
    }
    policy_result = evaluate_clinical_safety_policy(
        _derive_safety_policy_input(qa_summary)
    )
    assessment["disposition"] = merge_policy_disposition(
        assessment["disposition"],
        policy_result["disposition"],
    )
    assessment["risk_level"] = _risk_level_after_disposition(
        assessment["risk_level"],
        assessment["disposition"],
    )
    assessment["safety_policy_version"] = policy_result["safety_policy_version"]
    assessment["matched_rules"] = list(policy_result["matched_rules"])
    assessment["hard_fail_flags"] = list(policy_result["hard_fail_flags"])
    assessment["patient_message_key"] = policy_result["patient_message_key"]
    assessment["assessment_id"] = _default_assessment_id(assessment)
    return assessment


def start_crc_triage_state(registry_patient_id: int) -> dict[str, Any]:
    first_question = _question_payload(_CRC_TRIAGE_QUESTION_DEFINITIONS[0])
    return {
        "stage": first_question["stage"],
        "identity": {
            "source": "langg_registry",
            "registry_patient_id": registry_patient_id,
            "crc_client_local_id": None,
        },
        "current_question": first_question,
        "active_inquiry": True,
        "qa_summary": [],
        "node_results": [],
        "miss_count": 0,
    }


def advance_crc_triage(state: dict[str, Any], answer: CrcTriageAnswer) -> dict[str, Any]:
    next_state = deepcopy(state)
    current_question = next_state.get("current_question")

    if not current_question or current_question.get("id") != answer.question_id:
        next_state.setdefault("qa_summary", []).append(
            {
                "stage": next_state.get("stage"),
                "question_id": "free_text",
                "question": None,
                "answer": answer.answer_text,
            }
        )
        return next_state

    qa_summary = next_state.setdefault("qa_summary", [])
    qa_summary.append(
        {
            "stage": current_question["stage"],
            "question_id": current_question["id"],
            "question": current_question["text"],
            "answer": answer.answer_text,
        }
    )

    following_question = _next_question(current_question["id"])
    if following_question is not None and following_question.askable:
        if current_question["id"] == "vitals_heart_or_breathing":
            next_state.setdefault("node_results", []).append(_vitals_result(qa_summary))
        next_state["current_question"] = _question_payload(following_question)
        next_state["active_inquiry"] = True
        next_state["stage"] = following_question.stage
    else:
        next_state["current_question"] = None
        next_state["active_inquiry"] = False
        next_state["stage"] = "final"
        if current_question["id"] == "tests_recent_exam":
            next_state["assessment"] = _build_final_assessment(next_state)

    return next_state


__all__ = [
    "CRC_TRIAGE_QUESTIONS",
    "CRC_TRIAGE_STAGE_SEQUENCE",
    "CrcTriageAnswer",
    "CrcTriageStage",
    "advance_crc_triage",
    "start_crc_triage_state",
]
