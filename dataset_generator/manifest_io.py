from __future__ import annotations

import csv
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional


@dataclass
class ManifestRow:
    image_filename: str
    region: str
    category: str
    store_name: str
    language: str
    image_url: str
    attribution: str
    crawled_at: str
    status: str
    clip_score: Optional[float] = None
    ocr_box_count: Optional[int] = None
    top_ocr_text: Optional[str] = None
    best_levenshtein_score: Optional[float] = None
    num_label_boxes: Optional[int] = None
    split: Optional[str] = None


@dataclass
class DetectionRow:
    image_filename: str
    ocr_text: str
    ocr_text_normalized: str
    ocr_confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float
    x4: float
    y4: float
    levenshtein_score: Optional[float] = None
    ocr_filter_type: Optional[str] = None
    is_prominent_sign: Optional[bool] = None


@dataclass
class PipelineConfig:
    run_id: str
    raw_root: str
    work_root: str
    output_root: str
    clip_filter_threshold: float = 0.7
    similarity_threshold: float = 0.5
    device: str = "cpu"


_MANIFEST_FIELDS = [f.name for f in fields(ManifestRow)]
_DETECTION_FIELDS = [f.name for f in fields(DetectionRow)]


def _opt_float(v: str) -> Optional[float]:
    return float(v) if v not in ("", None) else None


def _opt_int(v: str) -> Optional[int]:
    return int(v) if v not in ("", None) else None


def _opt_bool(v: str) -> Optional[bool]:
    if v in ("", None):
        return None
    return v.lower() in ("true", "1", "yes")


def _row_to_manifest(row: dict[str, str]) -> ManifestRow:
    return ManifestRow(
        image_filename=row["image_filename"],
        region=row["region"],
        category=row["category"],
        store_name=row["store_name"],
        language=row["language"],
        image_url=row["image_url"],
        attribution=row["attribution"],
        crawled_at=row["crawled_at"],
        status=row["status"],
        clip_score=_opt_float(row.get("clip_score", "")),
        ocr_box_count=_opt_int(row.get("ocr_box_count", "")),
        top_ocr_text=row.get("top_ocr_text") or None,
        best_levenshtein_score=_opt_float(row.get("best_levenshtein_score", "")),
        num_label_boxes=_opt_int(row.get("num_label_boxes", "")),
        split=row.get("split") or None,
    )


def _row_to_detection(row: dict[str, str]) -> DetectionRow:
    return DetectionRow(
        image_filename=row["image_filename"],
        ocr_text=row["ocr_text"],
        ocr_text_normalized=row["ocr_text_normalized"],
        ocr_confidence=float(row["ocr_confidence"]),
        x1=float(row["x1"]),
        y1=float(row["y1"]),
        x2=float(row["x2"]),
        y2=float(row["y2"]),
        x3=float(row["x3"]),
        y3=float(row["y3"]),
        x4=float(row["x4"]),
        y4=float(row["y4"]),
        levenshtein_score=_opt_float(row.get("levenshtein_score", "")),
        ocr_filter_type=row.get("ocr_filter_type") or None,
        is_prominent_sign=_opt_bool(row.get("is_prominent_sign", "")),
    )


def _to_dict(row: Any, fieldnames: list[str]) -> dict[str, Any]:
    d = asdict(row)
    return {k: ("" if d[k] is None else d[k]) for k in fieldnames}


class _CsvIO:
    """Common locked CSV reader/writer with auto-init header."""

    _fields: list[str] = []

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._fields).writeheader()


class ManifestIO(_CsvIO):
    _fields = _MANIFEST_FIELDS

    def read_all(self) -> list[ManifestRow]:
        with self._lock, open(self._path, newline="", encoding="utf-8") as f:
            return [_row_to_manifest(row) for row in csv.DictReader(f)]

    def update_row(self, image_filename: str, **fields_: Any) -> None:
        with self._lock:
            with open(self._path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                if row["image_filename"] == image_filename:
                    for k, v in fields_.items():
                        row[k] = "" if v is None else str(v)
            with open(self._path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._fields)
                w.writeheader()
                w.writerows(rows)

    def write_all(self, rows: list[ManifestRow]) -> None:
        with self._lock, open(self._path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            w.writeheader()
            w.writerows([_to_dict(r, self._fields) for r in rows])


class DetectionIO(_CsvIO):
    _fields = _DETECTION_FIELDS

    def read_all(self) -> list[DetectionRow]:
        with self._lock, open(self._path, newline="", encoding="utf-8") as f:
            return [_row_to_detection(row) for row in csv.DictReader(f)]

    def append_rows(self, rows: list[DetectionRow]) -> None:
        with self._lock, open(self._path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            w.writerows([_to_dict(r, self._fields) for r in rows])

    def write_all(self, rows: list[DetectionRow]) -> None:
        with self._lock, open(self._path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            w.writeheader()
            w.writerows([_to_dict(r, self._fields) for r in rows])

    def update_rows(self, image_filename: str, **fields_: Any) -> None:
        with self._lock:
            with open(self._path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                if row["image_filename"] == image_filename:
                    for k, v in fields_.items():
                        row[k] = "" if v is None else str(v)
            with open(self._path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._fields)
                w.writeheader()
                w.writerows(rows)
