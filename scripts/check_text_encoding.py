from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


QUESTION_RUN = re.compile(r"\?{3,}")
PRIVATE_USE_OR_REPLACEMENT = re.compile(r"[\uE000-\uF8FF\uFFFD]")
COMMON_CHINESE_CODEPOINTS = {
    0x4E2D,
    0x4E34,
    0x4F18,
    0x4F30,
    0x4F53,
    0x4F7F,
    0x4FDD,
    0x5173,
    0x5177,
    0x5206,
    0x5212,
    0x5361,
    0x53D6,
    0x56DE,
    0x5757,
    0x590D,
    0x5B58,
    0x5F15,
    0x5F84,
    0x5DE5,
    0x60A3,
    0x6267,
    0x636E,
    0x652F,
    0x6570,
    0x6587,
    0x65B0,
    0x662F,
    0x667A,
    0x670D,
    0x67E5,
    0x68C0,
    0x6A21,
    0x70B9,
    0x7406,
    0x7528,
    0x75C5,
    0x7D22,
    0x7ED3,
    0x7EDC,
    0x7F51,
    0x8282,
    0x884C,
    0x89C4,
    0x8BCD,
    0x8DEF,
    0x8BEF,
    0x8BC4,
    0x8C03,
    0x8FD4,
    0x95EE,
    0x9519,
    0x952E,
}
LATIN_MOJIBAKE_CODEPOINTS = set(range(0x00C0, 0x0100)) | {
    0x0152,
    0x0153,
    0x0160,
    0x0161,
    0x0178,
    0x017D,
    0x017E,
    0x0192,
    0x2020,
    0x2021,
    0x2030,
    0x2039,
    0x203A,
    0x20AC,
    0x2122,
    0xFFFD,
}


@dataclass(frozen=True)
class EncodingIssue:
    path: str
    line: int
    kind: str
    preview: str


def _tracked_files(root: Path) -> list[Path]:
    try:
        output = subprocess.check_output(
            ["git", "ls-files"],
            cwd=root,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return [
            path
            for path in root.rglob("*")
            if path.is_file() and ".git" not in path.parts and "node_modules" not in path.parts
        ]
    return [root / line for line in output.splitlines() if line.strip()]


def _looks_binary(data: bytes) -> bool:
    return b"\x00" in data[:4096]


def _count_cjk(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def _count_common_chinese(text: str) -> int:
    return sum(1 for char in text if ord(char) in COMMON_CHINESE_CODEPOINTS)


def _recover_gbk_mojibake(line: str) -> str:
    return line.encode("gbk", errors="ignore").decode("utf-8", errors="ignore")


def _count_latin_mojibake(text: str) -> int:
    return sum(1 for char in text if ord(char) in LATIN_MOJIBAKE_CODEPOINTS)


def _scan_text(path: Path, text: str, root: Path) -> Iterable[EncodingIssue]:
    relative = path.relative_to(root).as_posix()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if PRIVATE_USE_OR_REPLACEMENT.search(line):
            yield EncodingIssue(relative, line_number, "private_use_or_replacement", line.strip()[:120])
        recovered = _recover_gbk_mojibake(line)
        if (
            recovered != line
            and _count_cjk(recovered) >= 3
            and _count_common_chinese(recovered) >= 2
        ):
            yield EncodingIssue(relative, line_number, "gbk_mojibake", line.strip()[:120])
        if _count_latin_mojibake(line) >= 3 and _count_cjk(line) < 2:
            yield EncodingIssue(relative, line_number, "latin_mojibake", line.strip()[:120])
        if QUESTION_RUN.search(line):
            yield EncodingIssue(relative, line_number, "question_mark_loss", line.strip()[:120])


def find_text_encoding_issues(root: Path) -> list[EncodingIssue]:
    issues: list[EncodingIssue] = []
    for path in _tracked_files(root):
        if not path.exists() or not path.is_file():
            continue
        data = path.read_bytes()
        if _looks_binary(data):
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            issues.append(
                EncodingIssue(
                    path.relative_to(root).as_posix(),
                    exc.start + 1,
                    "invalid_utf8",
                    str(exc),
                )
            )
            continue
        issues.extend(_scan_text(path, text, root))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Check tracked text files for common Chinese mojibake.")
    parser.add_argument("root", nargs="?", default=".", help="Repository root to scan.")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    issues = find_text_encoding_issues(root)
    for issue in issues:
        preview = issue.preview.encode("unicode_escape", errors="backslashreplace").decode("ascii")
        print(f"{issue.path}:{issue.line}: {issue.kind}: {preview}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
