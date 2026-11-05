"""Map the LLM's selected strings back onto the OCR boxes they came from.

Matching tolerates formatting only: Unicode form, case, whitespace,
punctuation, and several OCR fragments joined into one name. No edit distance.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

from navocr_bench.llm import FilterResult
from navocr_bench.ocr import OcrItem, OcrResult

_NON_WORD = re.compile(r"[^\w]", re.UNICODE)


@dataclass(frozen=True)
class RemapResult:
    id: str
    items: list[OcrItem]
    unmatched: list[str]


def _key(text: str) -> str:
    """NFKC, casefold, no whitespace or punctuation."""
    return _NON_WORD.sub("", unicodedata.normalize("NFKC", text).casefold())


def _find(items: Sequence[OcrItem], text: str) -> list[int]:
    hits = [i for i, item in enumerate(items) if item.text == text]
    if not hits:
        wanted = _key(text)
        if wanted:
            hits = [i for i, item in enumerate(items) if _key(item.text) == wanted]
    return hits


def remap_selection(
    items: Sequence[OcrItem], selected: Iterable[str]
) -> tuple[list[OcrItem], list[str]]:
    """Exact match, then normalised match, then per-token match.

    A string matching several items keeps every box; one matching nothing goes
    to ``unmatched``.
    """
    kept: dict[int, OcrItem] = {}
    unmatched = []
    for text in selected:
        hits = _find(items, text)
        if not hits:
            tokens = text.split()
            if len(tokens) > 1:
                hits = [i for token in tokens for i in _find(items, token)]
        if not hits:
            unmatched.append(text)
        for i in hits:
            kept.setdefault(i, items[i])
    return [kept[i] for i in sorted(kept)], unmatched


def remap(
    ocr_results: Iterable[OcrResult], filtered: Iterable[FilterResult]
) -> list[RemapResult]:
    selections = {r.id: r.selected for r in filtered}
    out = []
    for result in ocr_results:
        items, unmatched = remap_selection(
            result.items, selections.get(result.id, [])
        )
        out.append(RemapResult(id=result.id, items=items, unmatched=unmatched))
    return out
