"""Recover the selected text list from a small model's raw completion.

Handles JSON in code fences, arrays inside prose, truncated arrays and bullet lists.
"""
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
# Cut off by max tokens: opening fence, no closing one.
_FENCE_MARKER = re.compile(r"```(?:json)?")
_ARRAY = re.compile(r"\[.*?\]", re.DOTALL)
# Greedy first: names can contain brackets ("[유:운]").
_ARRAY_GREEDY = re.compile(r"\[.*\]", re.DOTALL)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_EMPTY_SENTINELS = {"none", "n/a", "null", "-"}
# After a one-shot prompt (p4) the model may invent more "Input:" examples;
# cut there. Not at "Output:", which can prefix the real answer.
_CONTINUATION = re.compile(r"\n\s*Input\s*:", re.IGNORECASE)
# Line fallback: longer lines are explanations, not names. Store names in the
# ground truth are 1-7 tokens.
_MAX_NAME_TOKENS = 7


def _is_prose(line: str) -> bool:
    return (
        "**" in line
        or line.endswith((":", "."))
        or len(line.split()) > _MAX_NAME_TOKENS
    )


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _strings(value) -> list[str] | None:
    if isinstance(value, dict):
        for key in ("text", "texts", "labels", "result"):
            if key in value:
                return _strings(value[key])
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v.strip() for v in value if isinstance(v, str)]
    return None


def _from_json(blob: str) -> list[str] | None:
    try:
        return _strings(json.loads(blob))
    except json.JSONDecodeError:
        return None


def _clean_piece(piece: str) -> str:
    """One comma-separated slice of a malformed array back to a bare name."""
    piece = piece.strip().rstrip(",").strip()
    # Trailing array-closers that do not belong to the name itself.
    while piece.endswith("]") and piece.count("]") > piece.count("["):
        piece = piece[:-1].strip().rstrip(",").strip()
    if len(piece) >= 2 and piece[0] == piece[-1] and piece[0] in "\"'":
        piece = piece[1:-1]
    elif piece[:1] in {'"', "'"}:  # string cut off by max-tokens
        piece = piece[1:]
    return piece.strip()


def _salvage_array(text: str) -> list[str]:
    """Names out of an array JSON refused: truncated, or a trailing comma."""
    start = text.find("[")
    if start < 0:
        return []
    pieces = (_clean_piece(p) for p in text[start + 1 :].split(","))
    return [
        p
        for p in pieces
        if p and p.casefold() not in _EMPTY_SENTINELS and not _is_prose(p)
    ]


def parse_selection(raw: str) -> list[str]:
    text = raw.strip()
    if not text or text.casefold() in _EMPTY_SENTINELS:
        return []

    cut = _CONTINUATION.search(text)
    if cut:
        text = text[: cut.start()].strip()

    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    else:
        text = _FENCE_MARKER.sub(" ", text).strip()

    candidates = (
        text,
        *(m.group(0) for m in _ARRAY_GREEDY.finditer(text)),
        *(m.group(0) for m in _ARRAY.finditer(text)),
    )
    for candidate in candidates:
        items = _from_json(candidate)
        if items is not None:
            return _dedupe(items)

    if "[" in text:
        salvaged = _salvage_array(text)
        if salvaged:
            return _dedupe(salvaged)

    lines = (_BULLET.sub("", line).strip() for line in text.splitlines())
    return _dedupe(
        [
            line
            for line in lines
            if line.strip("[]{},")
            and line.casefold() not in _EMPTY_SENTINELS
            and not _is_prose(line)
        ]
    )
