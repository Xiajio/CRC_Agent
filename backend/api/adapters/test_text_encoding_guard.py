from __future__ import annotations

import importlib.util
import shutil
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load_encoding_checker():
    checker_path = ROOT / "scripts" / "check_text_encoding.py"
    spec = importlib.util.spec_from_file_location("check_text_encoding", checker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tracked_text_files_do_not_contain_common_chinese_mojibake() -> None:
    checker = _load_encoding_checker()

    issues = checker.find_text_encoding_issues(ROOT)

    assert issues == []


def test_detects_private_use_mojibake_fragments() -> None:
    checker = _load_encoding_checker()
    sample_root = Path("sample-root")
    sample = sample_root / "sample.py"
    mojibake_fragment = "".join(chr(codepoint) for codepoint in [0x74BA, 0xE21C, 0x6571, 0x95AB, 0x660F, 0x7DEB])

    issues = list(checker._scan_text(sample, f"# === 2. {mojibake_fragment} (Routing Logic) ===\n", sample_root))

    assert any(issue.kind == "private_use_or_replacement" for issue in issues)


def test_detects_short_gbk_mojibake_ui_labels() -> None:
    checker = _load_encoding_checker()
    sample_root = Path("sample-root")
    sample = sample_root / "uploads-panel.tsx"
    mojibake_label = "\u8d44\u6599\u4e0a\u4f20".encode("utf-8").decode("gbk")

    issues = list(checker._scan_text(sample, f"<h2>{mojibake_label}</h2>\n", sample_root))

    assert any(issue.kind == "gbk_mojibake" for issue in issues)


def test_scans_ignored_superpowers_specs(monkeypatch) -> None:
    checker = _load_encoding_checker()
    temp_root = ROOT / "_tmp" / f"text-encoding-guard-{uuid.uuid4().hex}"
    try:
        temp_root.mkdir(parents=True)
        spec_path = temp_root / "docs" / "superpowers" / "specs" / "damaged-design.md"
        spec_path.parent.mkdir(parents=True)
        mojibake_title = "\u8d44\u6599\u4e0a\u4f20".encode("utf-8").decode("gbk")
        spec_path.write_text(f"# {mojibake_title}\n", encoding="utf-8")
        monkeypatch.setattr(checker, "_tracked_files", lambda root: [])

        issues = checker.find_text_encoding_issues(temp_root)

        assert any(
            issue.path == "docs/superpowers/specs/damaged-design.md" and issue.kind == "gbk_mojibake"
            for issue in issues
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
