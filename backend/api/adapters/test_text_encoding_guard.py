from __future__ import annotations

import importlib.util
import sys
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
