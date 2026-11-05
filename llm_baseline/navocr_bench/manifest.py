"""Manifest: the image list of a split, with an ``img_NNNN`` id per image.

Later stages key their outputs by this id and check them against the list.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from navocr_bench.paths import prepared

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Entry:
    id: str
    filename: str


def build_manifest(image_dir: Path | str) -> list[Entry]:
    paths = sorted(
        p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    return [Entry(id=f"img_{i:04d}", filename=p.name) for i, p in enumerate(paths, 1)]


def write_manifest(entries: list[Entry], path: Path | str) -> None:
    payload = [asdict(e) for e in entries]
    prepared(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_manifest(path: Path | str) -> list[Entry]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Entry(id=row["id"], filename=row["filename"]) for row in payload]
