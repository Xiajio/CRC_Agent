from __future__ import annotations

from pathlib import Path

from src.contracts.diagram import DiagramCompileRequest, DiagramSpec
from src.services.diagram_service import DiagramService


class _Reasoner:
    def generate_spec(self, request: DiagramCompileRequest) -> DiagramSpec:
        return DiagramSpec.model_validate(
            {
                "metadata": {
                    "title": "只读实验流程",
                    "diagram_type": request.diagram_type,
                },
                "layout": {"direction": request.direction},
                "nodes": [
                    {"id": "start", "label": "开始", "ports": ["out"]},
                    {"id": "finish", "label": "结束", "ports": ["in"]},
                ],
                "edges": [
                    {
                        "id": "flow",
                        "source": "start.out",
                        "target": "finish.in",
                        "type": "control_flow",
                    }
                ],
            }
        )


def _write_sentinel(path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{label}: original\n", encoding="utf-8")


def _snapshot(paths: dict[str, Path]) -> dict[str, str]:
    return {label: path.read_text(encoding="utf-8") for label, path in paths.items()}


def _all_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_diagram_shadow_compile_is_stateless_and_writes_nothing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protected_paths = {
        "safety_policy": tmp_path / "config" / "clinical_safety_policy.yaml",
        "prompt": tmp_path / "src" / "prompts" / "crc_prompt.md",
        "route": tmp_path / "src" / "routes" / "crc_routes.json",
        "rag": tmp_path / "reports" / "literature" / "rag_index.json",
        "patient_state": tmp_path / "runtime" / "patient_state" / "current.json",
        "doctor_state": tmp_path / "runtime" / "doctor_state" / "current.json",
        "session_state": tmp_path / "runtime" / "sessions" / "current.json",
        "training_data": tmp_path / "training_data" / "diagram.jsonl",
        "model": tmp_path / "models" / "diagram.bin",
    }
    for label, path in protected_paths.items():
        _write_sentinel(path, label)
    before_snapshot = _snapshot(protected_paths)
    before_files = _all_files(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = DiagramService(reasoner=_Reasoner()).compile(
        DiagramCompileRequest(
            prompt="开始后进入结束节点。",
            requested_by="admin_operator",
            idempotency_key="diagram-non-mutation-001",
            diagram_type="flowchart",
            direction="LR",
            deidentified=True,
        )
    )

    assert result.runtime.persisted is False
    assert result.runtime.applies_automatically is False
    assert result.runtime.clinical_state_mutated is False
    assert _snapshot(protected_paths) == before_snapshot
    assert _all_files(tmp_path) == before_files
