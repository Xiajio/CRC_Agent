from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _template_tick(node_name: str, node_output: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_name": node_name,
        "node_output": node_output,
    }


def _assistant_message(message_id: str, content: str) -> dict[str, Any]:
    return {
        "type": "ai",
        "id": message_id,
        "content": content,
    }


def _guideline_references() -> list[dict[str, Any]]:
    return [
        {
            "index": 1,
            "source": "NCCN Rectal Cancer",
            "page": 1,
            "snippet": "For locally advanced nonmetastatic rectal cancer, multidisciplinary planning and neoadjuvant therapy are guideline-supported options.",
        },
        {
            "index": 2,
            "source": "ESMO Colorectal Cancer",
            "page": 2,
            "snippet": "Treatment selection should integrate stage, resectability, performance status, and anticipated toxicity.",
        },
    ]


def _patient_card_payload() -> dict[str, Any]:
    return {
        "patient_id": "093",
        "data": {
            "patient_id": "093",
            "patient_info": {
                "gender": "Male",
                "age": 31,
                "ecog": 1,
                "cea": "8.4 ng/mL",
            },
            "diagnosis_block": {
                "confirmed": "Rectal adenocarcinoma",
                "primary_site": "Rectum",
                "mmr_status": "pMMR",
            },
            "staging_block": {
                "clinical_stage": "Locally advanced CRC",
                "ct_stage": "T3",
                "cn_stage": "N-positive concern",
                "cm_stage": "M0",
            },
        },
    }


def _medical_card_payload() -> dict[str, Any]:
    return {
        "summary": "Uploaded note summary for patient 093 with biopsy-proven rectal adenocarcinoma and locally advanced nonmetastatic disease.",
        "data": {
            "patient_summary": {
                "patient_id": "093",
                "name": "Acceptance Patient 093",
                "chief_complaint": "Uploaded note highlights colorectal cancer findings requiring follow-up planning.",
                "age": 31,
                "gender": "Male",
            },
            "diagnosis_block": {
                "confirmed": "Rectal adenocarcinoma",
            },
            "staging_block": {
                "clinical_stage": "Locally advanced disease",
                "risk_status": "High risk",
            },
            "treatment_draft": [
                {
                    "name": "Next step",
                    "details": "Review the uploaded findings with the CRC team and confirm the next treatment decision.",
                }
            ],
        },
    }


CASE_PRESETS: dict[str, dict[str, Any]] = {
    "database_case": {
        "thread_id": "capture-database-case",
        "prompt": "Please review patient 093 and use the case database for CRC findings, imaging, and pathology when relevant.",
        "state": {
            "patient_profile": {},
            "findings": {},
            "clinical_stage": "Assessment",
            "assessment_draft": None,
            "decision_json": None,
            "medical_card": None,
            "roadmap": [],
            "current_patient_id": "093",
        },
    },
    "decision_case": {
        "thread_id": "capture-decision-case",
        "prompt": "Please draft a treatment decision for patient 093 and explain the CRC plan with evidence.",
        "capture_mode": "template",
        "state": {
            "patient_profile": {},
            "findings": {"pathology_confirmed": True},
            "clinical_stage": "Decision",
            "assessment_draft": "Existing structured assessment draft.",
            "decision_json": None,
            "medical_card": None,
            "roadmap": [],
            "current_patient_id": "093",
        },
        "ticks": [
            _template_tick(
                "assessment",
                {
                    "clinical_stage": "Decision",
                    "current_patient_id": "093",
                    "messages": [
                        _assistant_message(
                            "fixture-decision-case",
                            "For patient 093, this deterministic fixture recommends multidisciplinary review, neoadjuvant therapy, interval restaging, and then a surgery discussion.",
                        )
                    ],
                    "current_plan": [
                        {"step": "Confirm multidisciplinary CRC review", "status": "completed"},
                        {"step": "Proceed with neoadjuvant systemic therapy", "status": "in_progress"},
                        {"step": "Restage and discuss definitive surgery", "status": "pending"},
                    ],
                    "retrieved_references": _guideline_references(),
                    "assessment_draft": "For patient 093, this deterministic fixture recommends multidisciplinary review, neoadjuvant therapy, interval restaging, and then a surgery discussion.",
                    "patient_card": _patient_card_payload(),
                },
            )
        ],
    },
    "safety_case": {
        "thread_id": "capture-safety-case",
        "prompt": "Patient 093 reportedly had a severe prior reaction to oxaliplatin. Please assess safety risks before continuing the CRC treatment discussion.",
        "capture_mode": "template",
        "state": {
            "patient_profile": {},
            "findings": {"safety_violation": "unknown"},
            "clinical_stage": "Decision",
            "assessment_draft": None,
            "decision_json": None,
            "medical_card": None,
            "roadmap": [],
            "current_patient_id": "093",
        },
        "ticks": [
            _template_tick(
                "assessment",
                {
                    "clinical_stage": "Decision",
                    "current_patient_id": "093",
                    "messages": [
                        _assistant_message(
                            "fixture-safety-case",
                            "This fixture blocks further oxaliplatin planning until the prior severe reaction is reviewed and an alternative regimen is considered.",
                        )
                    ],
                    "safety_violation": "High-risk chemotherapy safety alert: do not continue oxaliplatin until the prior severe reaction is clarified and the regimen is reassessed.",
                    "assessment_draft": "This fixture blocks further oxaliplatin planning until the prior severe reaction is reviewed and an alternative regimen is considered.",
                },
            )
        ],
    },
    "knowledge_case": {
        "thread_id": "capture-knowledge-case",
        "prompt": "What does cT3 mean in colorectal cancer staging?",
        "capture_mode": "template",
        "state": {
            "patient_profile": {},
            "findings": {},
            "clinical_stage": "Assessment",
            "assessment_draft": None,
            "decision_json": None,
            "medical_card": None,
            "roadmap": [],
            "current_patient_id": None,
        },
        "ticks": [
            _template_tick(
                "assessment",
                {
                    "clinical_stage": "Knowledge",
                    "messages": [
                        _assistant_message(
                            "fixture-knowledge-case",
                            "cT3 means the colorectal primary tumor extends through the muscular wall into surrounding pericolorectal tissue, so staging workup should confirm local extent and distant spread before treatment planning.",
                        )
                    ],
                    "retrieved_references": _guideline_references(),
                    "assessment_draft": "cT3 means the colorectal primary tumor extends through the muscular wall into surrounding pericolorectal tissue, so staging workup should confirm local extent and distant spread before treatment planning.",
                },
            )
        ],
    },
    "offtopic_date_case": {
        "thread_id": "capture-offtopic-date-case",
        "prompt": "What date is it today?",
        "capture_mode": "template",
        "state": {
            "patient_profile": {},
            "findings": {},
            "clinical_stage": "Assessment",
            "assessment_draft": None,
            "decision_json": None,
            "medical_card": None,
            "roadmap": [],
            "current_patient_id": None,
        },
        "ticks": [
            _template_tick(
                "assessment",
                {
                    "clinical_stage": "General Chat",
                    "messages": [
                        _assistant_message(
                            "fixture-offtopic-date-case",
                            "I cannot verify the real-time date in fixture mode. If you want to continue, ask another colorectal cancer question and I will stay on that topic.",
                        )
                    ],
                    "assessment_draft": "I cannot verify the real-time date in fixture mode. If you want to continue, ask another colorectal cancer question and I will stay on that topic.",
                },
            )
        ],
    },
    "offtopic_date_after_plan_case": {
        "thread_id": "capture-offtopic-after-plan-case",
        "prompt": "Can you check what date it is today?",
        "capture_mode": "template",
        "state": {
            "patient_profile": {},
            "current_patient_id": "093",
            "clinical_stage": "Decision",
            "assessment_draft": "Existing CRC assessment draft.",
            "findings": {"pathology_confirmed": True},
            "decision_json": None,
            "medical_card": None,
            "roadmap": [],
        },
        "ticks": [
            _template_tick(
                "assessment",
                {
                    "clinical_stage": "General Chat",
                    "current_patient_id": "093",
                    "messages": [
                        _assistant_message(
                            "fixture-offtopic-date-after-plan-case",
                            "I cannot verify the live date in this deterministic fixture run. If you want, we can return to patient 093 and continue the CRC treatment discussion.",
                        )
                    ],
                    "assessment_draft": "I cannot verify the live date in this deterministic fixture run. If you want, we can return to patient 093 and continue the CRC treatment discussion.",
                },
            )
        ],
    },
    "upload_followup_case": {
        "thread_id": "capture-upload-followup-case",
        "prompt": "Please use the uploaded note as context and summarize the CRC-relevant findings.",
        "capture_mode": "template",
        "state": {
            "patient_profile": {"patient_id": "093"},
            "findings": {"pathology_confirmed": True},
            "clinical_stage": "Assessment",
            "assessment_draft": None,
            "decision_json": None,
            "medical_card": {
                "type": "medical_visualization_card",
                "data": {
                    "patient_summary": {
                        "patient_id": "093",
                        "name": "Fixture Patient",
                        "primary_diagnosis": "Rectal adenocarcinoma",
                    },
                    "crc_relevant_findings": [
                        "Biopsy-proven colorectal adenocarcinoma",
                        "Rectal mass with cT3 staging context",
                        "Suspicious mesorectal lymph nodes",
                    ],
                },
            },
            "roadmap": [],
            "current_patient_id": "093",
        },
        "ticks": [
            _template_tick(
                "assessment",
                {
                    "clinical_stage": "Assessment",
                    "current_patient_id": "093",
                    "messages": [
                        _assistant_message(
                            "fixture-upload-followup-case",
                            "The uploaded note supports biopsy-proven rectal adenocarcinoma, locally advanced nonmetastatic disease, and the need for coordinated CRC treatment follow-up.",
                        )
                    ],
                    "assessment_draft": "The uploaded note supports biopsy-proven rectal adenocarcinoma, locally advanced nonmetastatic disease, and the need for coordinated CRC treatment follow-up.",
                    "medical_card": _medical_card_payload(),
                },
            )
        ],
    },
}


SENSITIVE_KEY_RE = re.compile(r"(api[_-]?key|token|secret|password|bearer)", re.IGNORECASE)
ABSOLUTE_PATH_RE = re.compile(r"^(?:[A-Za-z]:\\|/|\\\\)")
SENSITIVE_VALUE_PATTERNS = [
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\b(?:api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]+\b"),
]


def _sanitize_string(value: str) -> str:
    for pattern in SENSITIVE_VALUE_PATTERNS:
        if pattern.search(value):
            return "[REDACTED]"
    if ABSOLUTE_PATH_RE.match(value):
        return "<PATH>"
    return value.replace(os.getcwd(), "<WORKSPACE>")


def _serialize_message(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": getattr(message, "type", message.__class__.__name__.lower()),
        "content": getattr(message, "content", ""),
    }
    additional_kwargs = getattr(message, "additional_kwargs", None)
    if additional_kwargs:
        payload["additional_kwargs"] = _serialize_node_output(additional_kwargs)
    response_metadata = getattr(message, "response_metadata", None)
    if response_metadata:
        payload["response_metadata"] = _serialize_node_output(response_metadata)
    return payload


def _serialize_mapping(value: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        key_str = str(key)
        if SENSITIVE_KEY_RE.search(key_str):
            sanitized[key_str] = "[REDACTED]"
        else:
            sanitized[key_str] = _serialize_node_output(item)
    return sanitized


def _serialize_node_output(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _serialize_node_output(value.model_dump(mode="python"))
    if is_dataclass(value):
        return _serialize_node_output(asdict(value))
    if hasattr(value, "content") and hasattr(value, "__class__") and value.__class__.__name__.endswith("Message"):
        return _serialize_message(value)
    if isinstance(value, dict):
        return _serialize_mapping(value)
    if isinstance(value, (list, tuple, set)):
        return [_serialize_node_output(item) for item in value]
    if isinstance(value, Path):
        return "<PATH>" if value.is_absolute() else str(value)
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _sanitize_string(repr(value))


def _build_case_messages(case: dict[str, Any]) -> list[HumanMessage]:
    messages = [HumanMessage(content=case["prompt"])]
    for extra_message in case.get("messages", []):
        if extra_message.get("type") != "human":
            raise ValueError("Only human extra messages are supported in fixture presets")
        messages.insert(-1, HumanMessage(content=extra_message["content"]))
    return messages


async def _capture_live_case(case_name: str) -> dict[str, Any]:
    from src.config import Settings
    from src.graph_builder import build_graph

    case = CASE_PRESETS[case_name]
    settings = Settings()
    graph = build_graph(settings)
    initial_state = {
        **case["state"],
        "messages": _build_case_messages(case),
    }

    ticks: list[dict[str, Any]] = []
    async for update in graph.astream(
        initial_state,
        config={"configurable": {"thread_id": case["thread_id"]}},
        stream_mode="updates",
    ):
        if isinstance(update, dict):
            for node_name, node_output in update.items():
                ticks.append(
                    {
                        "node_name": node_name,
                        "node_output": _serialize_node_output(node_output),
                    }
                )

    return {
        "case_name": case_name,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "capture_mode": "live",
        "prompt": case["prompt"],
        "ticks": ticks,
    }


def _build_template_case(case_name: str) -> dict[str, Any]:
    case = CASE_PRESETS[case_name]
    ticks = case.get("ticks")
    if not isinstance(ticks, list):
        raise ValueError(f"Template fixture case is missing ticks: {case_name}")

    return {
        "case_name": case_name,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "capture_mode": "template",
        "prompt": case["prompt"],
        "ticks": ticks,
    }


async def _build_case(case_name: str) -> dict[str, Any]:
    case = CASE_PRESETS[case_name]
    if str(case.get("capture_mode") or "").lower() == "template":
        return _build_template_case(case_name)
    return await _capture_live_case(case_name)


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_serialize_node_output(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture sanitized graph tick fixtures.")
    parser.add_argument("--case", choices=sorted(CASE_PRESETS))
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--list-cases", action="store_true", help="Print supported case names")
    args = parser.parse_args()

    if args.list_cases:
        print("\n".join(sorted(CASE_PRESETS)))
        return 0

    if not args.case or not args.output:
        parser.error("--case and --output are required unless --list-cases is used")

    result = asyncio.run(_build_case(args.case))
    _write_output(Path(args.output), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
