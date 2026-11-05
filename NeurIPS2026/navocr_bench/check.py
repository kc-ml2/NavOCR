"""Check that every image a manifest names exists and is non-empty.

PaddleOCR reports an unreadable file as ``!buf.empty()`` with no file name, so
check up front. Names that exist only under another Unicode form (NFC/NFD) are
listed separately.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Iterable

from navocr_bench.manifest import Entry

Problem = tuple[str, str, str]


def _reason(root: Path, filename: str) -> str | None:
    path = root / filename
    if not path.exists():
        for form in ("NFC", "NFD"):
            if (root / unicodedata.normalize(form, filename)).exists():
                return "missing (unicode normalisation)"
        return "missing"
    if not path.is_file():
        return "not a file"
    if path.stat().st_size == 0:
        return "empty"
    return None


def audit(entries: Iterable[Entry], image_dir: Path | str) -> list[Problem]:
    root = Path(image_dir)
    problems = []
    for entry in entries:
        reason = _reason(root, entry.filename)
        if reason:
            problems.append((entry.id, entry.filename, reason))
    return problems
