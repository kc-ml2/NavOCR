"""Average precision over boxes, for Table 3.

Kept boxes are ranked by PP-OCRv5's recognition confidence, since the LLM gives
no score. All-point interpolation (VOC2010); ``compare_ap_pycocotools.py``
compares it with pycocotools' 101-point version.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from navocr_bench.llm import read_filtered
from navocr_bench.manifest import read_manifest
from navocr_bench.ocr import BBox, read_dump
from navocr_bench.remap import remap

# IoU thresholds Table 3 reports.
IOU_THRESHOLDS = (0.25, 0.5)


def iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / (aw * ah + bw * bh - inter)


@dataclass(frozen=True)
class Detection:
    """One predicted box. ``image`` keys into the ground truth mapping."""

    image: str
    bbox: BBox
    score: float


def _matches(
    gt: Mapping[str, Sequence[BBox]],
    preds: Sequence[Detection],
    iou_threshold: float,
) -> list[bool]:
    """Hit or miss per prediction in score order; each GT box is claimed once.

    Predictions on images absent from the ground truth are ignored.
    """
    ranked = sorted(
        (p for p in preds if p.image in gt), key=lambda p: p.score, reverse=True
    )
    claimed: dict[str, set[int]] = {image: set() for image in gt}
    hits = []
    for pred in ranked:
        boxes = gt[pred.image]
        taken = claimed[pred.image]
        best, best_iou = None, iou_threshold
        for i, box in enumerate(boxes):
            if i in taken:
                continue
            overlap = iou(pred.bbox, box)
            if overlap >= best_iou:
                best, best_iou = i, overlap
        if best is None:
            hits.append(False)
        else:
            taken.add(best)
            hits.append(True)
    return hits


def average_precision(
    gt: Mapping[str, Sequence[BBox]],
    preds: Sequence[Detection],
    iou_threshold: float,
) -> float | None:
    """Area under the interpolated precision-recall curve; None if GT is empty."""
    n_gt = sum(len(boxes) for boxes in gt.values())
    if n_gt == 0:
        return None
    hits = _matches(gt, preds, iou_threshold)
    points = [(0.0, 1.0)]
    tp = 0
    for i, hit in enumerate(hits, start=1):
        tp += hit
        points.append((tp / n_gt, tp / i))
    # Precision envelope: best precision at this recall or higher.
    best = 0.0
    envelope = []
    for recall, precision in reversed(points):
        best = max(best, precision)
        envelope.append((recall, best))
    envelope.reverse()
    area = 0.0
    for (r0, _), (r1, p1) in zip(envelope, envelope[1:]):
        area += (r1 - r0) * p1
    return area


def read_coco_gt(path: Path | str) -> dict[str, list[BBox]]:
    """COCO instances file to ``{file_name: [box, ...]}`` in pixel ``x y w h``.

    Images without annotations get an empty list, so predictions on them are
    false positives.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    names = {image["id"]: image["file_name"] for image in payload["images"]}
    boxes: dict[str, list[BBox]] = {name: [] for name in names.values()}
    for ann in payload["annotations"]:
        x, y, w, h = ann["bbox"]
        boxes[names[ann["image_id"]]].append((float(x), float(y), float(w), float(h)))
    return boxes


def detections(
    results: Iterable[object], names: Mapping[str, str]
) -> list[Detection]:
    """Flatten results carrying ``id`` and ``items`` into detections.

    ``names`` maps manifest ids to file names, which key the ground truth.
    """
    out = []
    for result in results:
        filename = names.get(result.id)
        if filename is None:
            continue
        for item in result.items:
            out.append(Detection(filename, item.bbox, item.score))
    return out


def evaluate(
    gt: Mapping[str, Sequence[BBox]],
    preds: Sequence[Detection],
    thresholds: Iterable[float] = IOU_THRESHOLDS,
) -> dict[str, float | None]:
    return {f"AP@{t:.2f}": average_precision(gt, preds, t) for t in thresholds}


def _is_selection(path: Path) -> bool:
    """True for an LLM selection file, False for an OCR dump (boxes)."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                return "selected" in json.loads(line)
    raise SystemExit(f"{path} is empty")


def boxed_results(path: Path, ocr_results):
    """Prediction file as results carrying boxes, remapping LLM selections."""
    if not _is_selection(path):
        return read_dump(path), "dump"
    if ocr_results is None:
        raise SystemExit(
            f"{path} holds LLM selections, not boxes; pass --ocr so they can be "
            "mapped back onto the OCR boxes they came from"
        )
    return remap(ocr_results, read_filtered(path)), "filtered"


def scored_ground_truth(manifest: Path, gt: dict):
    """Manifest id -> file name, and the ground truth restricted to the manifest."""
    names = {e.id: e.filename for e in read_manifest(manifest)}
    known = set(names.values())
    scored = {name: boxes for name, boxes in gt.items() if name in known}
    missing = len(gt) - len(scored)
    if missing:
        print(f"warning: {missing} annotated image(s) absent from the manifest")
    return names, scored
