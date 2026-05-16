from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.nodes import pathology_nodes, radiology_nodes
from src.state import CRCAgentState


class _UnusedModel:
    def invoke(self, _payload):
        raise AssertionError("Model should not run in media split identity tests.")


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


def _radiology_result(_payload: dict[str, object]) -> dict[str, object]:
    return {
        "has_tumor": False,
        "total_images": 2,
        "images_with_tumor": 0,
        "tumor_detection_rate": "0%",
        "max_confidence": 0.0,
        "total_detections": 0,
        "sample_images_with_tumor": [],
        "all_results": [],
        "processing_timestamp": "2026-05-09T00:00:00",
    }


def _pathology_result(payload: dict[str, object]) -> dict[str, object]:
    return {
        "success": True,
        "patient_id": payload["patient_id"],
        "slides_analyzed": 1,
        "tumor_slides": 0,
        "normal_slides": 1,
        "overall_diagnosis": "NEGATIVE",
        "results": [],
    }


def _pathology_slide_result(confidence: float) -> dict[str, object]:
    return {
        "success": True,
        "prediction": "tumor",
        "tumor_probability": 0.93,
        "confidence": confidence,
    }


def _radiology_result_with_confidence(max_confidence: float) -> dict[str, object]:
    result = _radiology_result({})
    result["max_confidence"] = max_confidence
    return result


def test_radiology_uses_registry_identity_before_stale_legacy_patient_id() -> None:
    tumor_tool = _FakeTool("perform_comprehensive_tumor_check", _radiology_result)
    runnable = radiology_nodes.node_rad_agent([tumor_tool], model=_UnusedModel(), show_thinking=False)

    result = runnable(
        CRCAgentState(
            messages=[HumanMessage(content="show CT imaging")],
            registry_patient_id=7,
            current_patient_id="093",
            findings={"current_patient_id": "093"},
        )
    )

    assert tumor_tool.calls == [{"patient_id": "007"}]
    assert result["registry_patient_id"] == 7
    assert result["case_database_patient_id"] == "007"
    assert result["findings"]["registry_patient_id"] == 7
    assert result["findings"]["case_database_patient_id"] == "007"


def test_pathology_uses_registry_identity_before_stale_legacy_patient_id() -> None:
    pathology_tool = _FakeTool("perform_comprehensive_pathology_analysis", _pathology_result)
    runnable = pathology_nodes.node_pathology_agent([pathology_tool], model=_UnusedModel(), show_thinking=False)

    result = runnable(
        CRCAgentState(
            messages=[HumanMessage(content="run pathology analysis")],
            registry_patient_id=7,
            current_patient_id="093",
            findings={"current_patient_id": "093"},
        )
    )

    assert pathology_tool.calls == [{"patient_id": "007"}]
    assert result["registry_patient_id"] == 7
    assert result["case_database_patient_id"] == "007"
    assert result["findings"]["registry_patient_id"] == 7
    assert result["findings"]["case_database_patient_id"] == "007"


def test_radiology_uses_case_sample_identity_before_legacy_patient_id() -> None:
    tumor_tool = _FakeTool("perform_comprehensive_tumor_check", _radiology_result)
    runnable = radiology_nodes.node_rad_agent([tumor_tool], model=_UnusedModel(), show_thinking=False)

    result = runnable(
        CRCAgentState(
            messages=[HumanMessage(content="show CT imaging")],
            case_database_patient_id="094",
            current_patient_id="093",
            findings={"current_patient_id": "093"},
        )
    )

    assert tumor_tool.calls == [{"patient_id": "094"}]
    assert result["case_database_patient_id"] == "094"
    assert result["findings"]["case_database_patient_id"] == "094"
    assert "registry_patient_id" not in result


def test_pathology_uses_legacy_patient_id_only_after_split_ids_are_absent() -> None:
    pathology_tool = _FakeTool("perform_comprehensive_pathology_analysis", _pathology_result)
    runnable = pathology_nodes.node_pathology_agent([pathology_tool], model=_UnusedModel(), show_thinking=False)

    result = runnable(
        CRCAgentState(
            messages=[HumanMessage(content="run pathology analysis")],
            current_patient_id="093",
            findings={"current_patient_id": "093"},
        )
    )

    assert pathology_tool.calls == [{"patient_id": "093"}]
    assert result["case_database_patient_id"] == "093"
    assert result["findings"]["case_database_patient_id"] == "093"
    assert "registry_patient_id" not in result


def test_pathology_slide_marks_review_when_confidence_is_below_threshold() -> None:
    pathology_tool = _FakeTool("pathology_slide_classify", _pathology_slide_result(0.79))
    runnable = pathology_nodes.node_pathology_agent([pathology_tool], model=_UnusedModel(), show_thinking=False)

    result = runnable(
        CRCAgentState(
            messages=[HumanMessage(content=r"analyze C:\slides\case001.svs")],
        )
    )

    pathology_card_data = result["findings"]["pathology_card"]["data"]
    pathology_report = result["findings"]["pathology_report"]
    assert pathology_card_data["confidence_threshold"] == 0.8
    assert pathology_card_data["needs_review"] is True
    assert pathology_report["confidence_threshold"] == 0.8
    assert pathology_report["needs_review"] is True


def test_pathology_slide_does_not_mark_review_at_or_above_threshold() -> None:
    pathology_tool = _FakeTool("pathology_slide_classify", _pathology_slide_result(0.81))
    runnable = pathology_nodes.node_pathology_agent([pathology_tool], model=_UnusedModel(), show_thinking=False)

    result = runnable(
        CRCAgentState(
            messages=[HumanMessage(content=r"analyze C:\slides\case002.svs")],
        )
    )

    pathology_card_data = result["findings"]["pathology_card"]["data"]
    assert pathology_card_data["confidence_threshold"] == 0.8
    assert pathology_card_data["needs_review"] is False


def test_radiology_detection_marks_review_when_confidence_is_below_threshold() -> None:
    tumor_tool = _FakeTool("perform_comprehensive_tumor_check", _radiology_result_with_confidence(0.49))
    runnable = radiology_nodes.node_rad_agent([tumor_tool], model=_UnusedModel(), show_thinking=False)

    result = runnable(
        CRCAgentState(
            messages=[HumanMessage(content="show CT imaging")],
            current_patient_id="093",
            findings={"current_patient_id": "093"},
        )
    )

    tumor_card_data = result["findings"]["tumor_detection_card"]["data"]
    assert tumor_card_data["confidence_threshold"] == 0.5
    assert tumor_card_data["needs_review"] is True


def test_radiology_detection_does_not_mark_review_at_threshold() -> None:
    tumor_tool = _FakeTool("perform_comprehensive_tumor_check", _radiology_result_with_confidence(0.5))
    runnable = radiology_nodes.node_rad_agent([tumor_tool], model=_UnusedModel(), show_thinking=False)

    result = runnable(
        CRCAgentState(
            messages=[HumanMessage(content="show CT imaging")],
            current_patient_id="093",
            findings={"current_patient_id": "093"},
        )
    )

    tumor_card_data = result["findings"]["tumor_detection_card"]["data"]
    assert tumor_card_data["confidence_threshold"] == 0.5
    assert tumor_card_data["needs_review"] is False
