"""Stage 1: dump every text region PP-OCRv5 finds, before any filtering.

Only ``build_paddle_engine`` imports PaddleOCR.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image, ImageOps

from navocr_bench.manifest import Entry
from navocr_bench.paths import prepared

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class OcrItem:
    text: str
    bbox: BBox
    score: float


@dataclass(frozen=True)
class OcrResult:
    id: str
    items: list[OcrItem]
    elapsed_s: float
    error: str | None = None


Engine = Callable[[Path], Sequence[OcrItem]]


def run_ocr(
    entries: Iterable[Entry],
    image_dir: Path | str,
    engine: Engine,
    clock: Callable[[], float] = time.perf_counter,
    abort_after: int = 10,
) -> Iterator[OcrResult]:
    """Run the engine over every entry, recording per-image failures.

    Stops after ``abort_after`` failures in a row (0 disables this).
    """
    root = Path(image_dir)
    consecutive = 0
    for entry in entries:
        start = clock()
        try:
            items = list(engine(root / entry.filename))
            error = None
            consecutive = 0
        except Exception as exc:
            items, error = [], str(exc)
            consecutive += 1
        yield OcrResult(
            id=entry.id, items=items, elapsed_s=clock() - start, error=error
        )
        if abort_after and consecutive >= abort_after:
            raise RuntimeError(
                f"aborting after {consecutive} consecutive failures; "
                f"last was {entry.id}: {error}"
            )


def write_dump(results: Iterable[OcrResult], path: Path | str) -> None:
    with prepared(path).open("w", encoding="utf-8") as fh:
        for result in results:
            row = {
                "id": result.id,
                "elapsed_s": result.elapsed_s,
                "error": result.error,
                "items": [
                    {"text": i.text, "bbox": list(i.bbox), "score": i.score}
                    for i in result.items
                ],
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_dump(path: Path | str) -> list[OcrResult]:
    results = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        results.append(
            OcrResult(
                id=row["id"],
                items=[
                    OcrItem(text=i["text"], bbox=tuple(i["bbox"]), score=i["score"])
                    for i in row["items"]
                ],
                elapsed_s=row["elapsed_s"],
                error=row.get("error"),
            )
        )
    return results


DET_MODEL = "PP-OCRv5_mobile_det"
REC_MODEL = "korean_PP-OCRv5_mobile_rec"
# PaddleOCR default.
REC_BATCH_SIZE = 6
# Long side capped at this; smaller images are never enlarged.
MAX_SIDE = 1080


def paddle_ocr_kwargs(
    device: str = "gpu", det_model: str = DET_MODEL, max_side: int = MAX_SIDE
) -> dict:
    """PaddleOCR arguments, with the document-preprocessing models switched off.

    ``limit_type="max"`` keeps the detector from enlarging images.
    """
    return {
        "text_detection_model_name": det_model,
        "text_recognition_model_name": REC_MODEL,
        "text_recognition_batch_size": REC_BATCH_SIZE,
        "text_det_limit_side_len": max_side,
        "text_det_limit_type": "max",
        "device": device,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }


def _downscaled(path: Path, max_side: int):
    """``(image, scale)`` with the long side capped at ``max_side``.

    The path and 1.0 if no resize is needed, otherwise a Lanczos-resized BGR
    array and the factor back to original pixels.
    """
    with Image.open(path) as image:
        long_side = max(image.size)
        if long_side <= max_side:
            return str(path), 1.0
        image = ImageOps.exif_transpose(image).convert("RGB")
        ratio = max_side / long_side
        size = (round(image.width * ratio), round(image.height * ratio))
        resized = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(resized)[:, :, ::-1].copy(), long_side / max_side


def build_paddle_engine(
    device: str = "gpu", det_model: str = DET_MODEL, max_side: int = MAX_SIDE
) -> Engine:
    """PP-OCRv5 detector + Korean mobile recogniser, input capped at ``max_side``.

    The Korean dictionary also covers Latin and digits. Boxes are in original
    image pixels, like the ground truth.
    """
    from paddleocr import PaddleOCR  # installed only in navocr-ocr

    reader = PaddleOCR(
        **paddle_ocr_kwargs(device=device, det_model=det_model, max_side=max_side)
    )

    def engine(path: Path) -> list[OcrItem]:
        image, scale = _downscaled(path, max_side)
        items = []
        for page in reader.predict(image):
            texts = page["rec_texts"]
            scores = page["rec_scores"]
            boxes = page["rec_boxes"]
            for text, score, box in zip(texts, scores, boxes):
                x0, y0, x1, y1 = (float(v) * scale for v in box)
                items.append(
                    OcrItem(
                        text=text,
                        bbox=(x0, y0, x1 - x0, y1 - y0),
                        score=float(score),
                    )
                )
        return items

    return engine


def ocr_settings(det_model: str = DET_MODEL, max_side: int = MAX_SIDE) -> dict:
    """Models, sizes and package versions of an OCR run."""
    return {
        "detection_model": det_model,
        "recognition_model": REC_MODEL,
        "recognition_batch_size": REC_BATCH_SIZE,
        "max_side": max_side,
        "det_limit_type": "max",
        "versions": package_versions("paddleocr", "paddlex", "paddlepaddle-gpu", "paddlepaddle"),
    }


def package_versions(*packages: str) -> dict[str, str]:
    out = {}
    for name in packages:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            pass
    return out
