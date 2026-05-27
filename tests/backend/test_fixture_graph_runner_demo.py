import pytest
from langchain_core.messages import HumanMessage

from backend.api.services.fixture_graph_runner import FixtureGraphRunner


async def collect_fixture_ticks(case_name: str, message: str, **payload):
    runner = FixtureGraphRunner(default_case=case_name)
    ticks = []
    async for tick in runner.astream(
        {
            "messages": [HumanMessage(content=message)],
            "fixture_case": case_name,
            **payload,
        },
        config={"configurable": {"thread_id": f"demo-{case_name}"}},
    ):
        ticks.append(tick)
    return ticks


@pytest.mark.asyncio
async def test_demo_patient_triage_question_fixture():
    ticks = await collect_fixture_ticks("demo_patient_triage_question", "最近两个月大便带血")

    assert [next(iter(tick.keys())) for tick in ticks] == ["intent_router", "outpatient_triage"]
    triage = ticks[-1]["outpatient_triage"]["findings"]["triage_question_card"]
    assert triage["type"] == "triage_question_card"
    assert triage["field_key"] == "duration"


@pytest.mark.asyncio
async def test_demo_patient_triage_final_fixture():
    ticks = await collect_fixture_ticks("demo_patient_triage_final", "持续时间超过1个月")

    assert [next(iter(tick.keys())) for tick in ticks] == ["outpatient_triage"]
    triage = ticks[-1]["outpatient_triage"]["findings"]["triage_card"]
    assert triage["type"] == "triage_card"
    assert triage["data"]["risk_level"] in {"medium", "high"}


@pytest.mark.asyncio
async def test_demo_doctor_decision_fixture():
    ticks = await collect_fixture_ticks(
        "demo_doctor_decision",
        "请基于当前患者信息生成临床评估、证据依据和治疗建议。",
        case_database_patient_id="093",
    )

    assert [next(iter(tick.keys())) for tick in ticks] == [
        "intent_router",
        "planner",
        "assessment",
        "decision",
        "citation",
        "critic",
        "finalize",
    ]
    final_output = ticks[-1]["finalize"]
    assert final_output["requires_human_review"] is True
    assert "\u9700\u4eba\u5de5\u590d\u6838" in final_output["final_response"]
    assert "HUMAN_REVIEW_REQUIRED" not in final_output["final_response"]
