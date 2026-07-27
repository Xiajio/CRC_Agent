from __future__ import annotations

import json
import threading

import pytest

from backend.api.services.auto_research_store import (
    AutoResearchRunNotFoundError,
    AutoResearchRunStore,
    AutoResearchStoreIntegrityError,
)
from src.contracts.auto_research import (
    AutoResearchRequest,
    AutoResearchRun,
    ResearchStage,
    make_auto_research_request_hash,
    make_auto_research_run_id,
)


def _failed_run() -> AutoResearchRun:
    request = AutoResearchRequest(
        request_id="request_store_001",
        project_id="project_store_001",
        question="A valid research question",
        requested_by="pi_operator",
        idempotency_key="store-001",
    )
    return AutoResearchRun(
        run_id=make_auto_research_run_id(
            request.project_id,
            request.idempotency_key,
        ),
        request_hash=make_auto_research_request_hash(request),
        request=request,
        status="failed_shadow",
        created_at="2026-07-19T08:00:00+00:00",
        completed_at="2026-07-19T08:00:01+00:00",
        stages=[
            ResearchStage(
                name="literature_search",
                status="failed",
                started_at="2026-07-19T08:00:00+00:00",
                completed_at="2026-07-19T08:00:01+00:00",
                summary="Failed closed.",
                error="No verified sources.",
            )
        ],
        sources=[],
        hypotheses=[],
        study_plans=[],
        report_markdown="",
        iteration_count=0,
        provenance={
            "pipeline_version": "shadow_auto_research_v1",
            "retriever": "fake",
            "reasoner": "fake",
        },
    )


def test_empty_store_read_does_not_create_directories(tmp_path: Path) -> None:
    root = tmp_path / "reports" / "auto_research"
    store = AutoResearchRunStore(root)

    state = store.read_state()

    assert state.runs == []
    assert state.integrity == {"status": "verified", "warnings": []}
    assert not root.exists()


def test_store_writes_once_and_round_trips_run(tmp_path: Path) -> None:
    store = AutoResearchRunStore(tmp_path / "reports" / "auto_research")
    run = _failed_run()

    store.write_run(run)

    assert store.get_run(run.run_id) == run
    assert store.read_state().runs == [run]
    with pytest.raises(FileExistsError):
        store.write_run(run)


def test_store_revalidates_mutable_contract_fields_before_writing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reports" / "auto_research"
    store = AutoResearchRunStore(root)
    run = _failed_run()
    run.stages.append(
        ResearchStage(
            name="report_synthesis",
            status="completed",
            started_at="2026-07-19T08:00:01+00:00",
            completed_at="2026-07-19T08:00:02+00:00",
            summary="This stage cannot follow a failed literature search.",
        )
    )

    with pytest.raises(ValueError, match="failed_shadow must contain only"):
        store.write_run(run)

    assert not root.exists()


def test_store_reports_malformed_artifact_as_integrity_warning(tmp_path: Path) -> None:
    root = tmp_path / "reports" / "auto_research"
    runs = root / "runs"
    runs.mkdir(parents=True)
    (runs / "broken.json").write_text("{not json", encoding="utf-8")

    state = AutoResearchRunStore(root).read_state()

    assert state.runs == []
    assert state.integrity["status"] == "warning"
    assert "not valid JSON" in state.integrity["warnings"][0]


def test_store_reports_contract_missing_fields_as_integrity_warning(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reports" / "auto_research"
    runs = root / "runs"
    runs.mkdir(parents=True)
    run = _failed_run()
    payload = run.to_dict()
    payload.pop("request")
    path = runs / f"{run.run_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    store = AutoResearchRunStore(root)

    state = store.read_state()

    assert state.runs == []
    assert state.integrity["status"] == "warning"
    assert "does not match the auto-research contract" in state.integrity["warnings"][0]
    with pytest.raises(
        AutoResearchStoreIntegrityError,
        match="does not match the auto-research contract",
    ):
        store.get_run(run.run_id)


def test_get_run_distinguishes_missing_from_invalid(tmp_path: Path) -> None:
    store = AutoResearchRunStore(tmp_path / "reports" / "auto_research")

    with pytest.raises(AutoResearchRunNotFoundError):
        store.get_run("auto_research_run_missing")
    with pytest.raises(AutoResearchStoreIntegrityError, match="safe filename"):
        store.get_run("../escape")


def test_store_rejects_filename_that_does_not_match_run_id(tmp_path: Path) -> None:
    root = tmp_path / "reports" / "auto_research"
    runs = root / "runs"
    runs.mkdir(parents=True)
    run = _failed_run()
    import json

    (runs / "auto_research_run_wrong.json").write_text(
        json.dumps(run.to_dict()),
        encoding="utf-8",
    )

    state = AutoResearchRunStore(root).read_state()

    assert state.runs == []
    assert "filename must match" in state.integrity["warnings"][0]
    assert state.integrity["affected_artifacts"] == [
        {
            "code": "filename_run_id_mismatch",
            "artifact_path": "runs/auto_research_run_wrong.json",
            "filename_run_id": "auto_research_run_wrong",
            "persisted_run_id": run.run_id,
            "message": state.integrity["warnings"][0],
            "excluded_from_runs": True,
        }
    ]
    assert [
        action["code"] for action in state.integrity["recovery_actions"]
    ] == ["rerun_with_new_idempotency_key", "manual_quarantine"]
    assert all(
        action["overwrites_existing_artifact"] is False
        and action["clinical_data_mutated"] is False
        for action in state.integrity["recovery_actions"]
    )
    assert (runs / "auto_research_run_wrong.json").read_text(
        encoding="utf-8"
    ) == json.dumps(run.to_dict())


def test_store_keeps_valid_runs_visible_while_excluding_mismatched_copy(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reports" / "auto_research"
    store = AutoResearchRunStore(root)
    run = _failed_run()
    store.write_run(run)
    copied_path = root / "runs" / "auto_research_run_validation_copy.json"
    copied_path.write_text(json.dumps(run.to_dict()), encoding="utf-8")

    state = store.read_state()

    assert state.runs == [run]
    assert state.integrity["status"] == "warning"
    assert state.integrity["affected_artifacts"][0]["artifact_path"] == (
        "runs/auto_research_run_validation_copy.json"
    )
    assert state.integrity["affected_artifacts"][0]["excluded_from_runs"] is True
    assert copied_path.is_file()


def test_store_removes_partial_file_when_serialization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reports" / "auto_research"
    store = AutoResearchRunStore(root)
    run = _failed_run()

    def fail_dump(*args: object, **kwargs: object) -> None:
        raise OSError("simulated serialization failure")

    monkeypatch.setattr("backend.api.services.auto_research_store.json.dump", fail_dump)

    with pytest.raises(OSError, match="simulated serialization failure"):
        store.write_run(run)

    assert not (root / "runs" / f"{run.run_id}.json").exists()
    assert list((root / "runs").glob("*.tmp")) == []


def test_store_does_not_publish_a_partially_written_json_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reports" / "auto_research"
    store = AutoResearchRunStore(root)
    run = _failed_run()
    started = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    original_dump = json.dump

    def slow_dump(payload: object, handle: object, **kwargs: object) -> None:
        handle.write('{"partial":')  # type: ignore[attr-defined]
        handle.flush()  # type: ignore[attr-defined]
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release the writer")
        handle.seek(0)  # type: ignore[attr-defined]
        handle.truncate()  # type: ignore[attr-defined]
        original_dump(payload, handle, **kwargs)

    monkeypatch.setattr("backend.api.services.auto_research_store.json.dump", slow_dump)

    def write() -> None:
        try:
            store.write_run(run)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    writer = threading.Thread(target=write)
    writer.start()
    assert started.wait(timeout=5)

    during_write = store.read_state()
    assert during_write.runs == []
    assert during_write.integrity == {"status": "verified", "warnings": []}

    release.set()
    writer.join(timeout=5)
    assert not writer.is_alive()
    assert errors == []
    assert store.get_run(run.run_id) == run
    assert list((root / "runs").glob("*.tmp")) == []
