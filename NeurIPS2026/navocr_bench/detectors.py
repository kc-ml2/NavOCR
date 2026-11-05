"""Adapters for fine-tuned PaddleDetection and PaddleOCR detectors."""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from navocr_bench.manifest import Entry
from navocr_bench.ocr import OcrItem, OcrResult, run_ocr


def _box_from_polygon(polygon) -> tuple[float, float, float, float] | None:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) < 4:
        return None
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    if x1 <= x0 or y1 <= y0:
        return None
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def build_ppocr_detector(
    model_dir: Path | str,
    device: str = "gpu:0",
    max_side: int | None = 1080,
):
    """Build a PP-OCRv5 detector that returns boxes and detector confidence."""
    # A local export does not need PaddleX's official model-host probe.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import TextDetection

    kwargs = {
        "model_name": "PP-OCRv5_mobile_det",
        "model_dir": str(Path(model_dir).resolve()),
        "device": device,
    }
    if device == "cpu":
        kwargs["enable_mkldnn"] = False
    if max_side is not None:
        kwargs.update(limit_side_len=max_side, limit_type="max")
    detector = TextDetection(**kwargs)

    def engine(path: Path) -> list[OcrItem]:
        predictions = list(detector.predict(str(path)))
        if len(predictions) != 1:
            raise RuntimeError(f"expected one detection result, got {len(predictions)}")
        result = predictions[0]
        polygons = result["dt_polys"]
        scores = result.get("dt_scores")
        if scores is None:
            scores = np.ones(len(polygons), dtype=np.float32)

        items = []
        for polygon, score in zip(polygons, scores):
            box = _box_from_polygon(polygon)
            if box is not None:
                items.append(OcrItem(text="", bbox=box, score=float(score)))
        return items

    return engine


def run_ppocr_detector(
    entries: Iterable[Entry],
    image_dir: Path | str,
    model_dir: Path | str,
    device: str = "gpu:0",
    max_side: int | None = 1080,
):
    engine = build_ppocr_detector(model_dir, device=device, max_side=max_side)
    return run_ocr(entries, image_dir, engine)


def run_paddledet_detector(
    entries: Iterable[Entry],
    image_dir: Path | str,
    paddledet_root: Path | str,
    config: Path | str,
    weights: Path | str,
    device: str = "gpu:0",
) -> list[OcrResult]:
    """Run a PaddleDetection checkpoint and convert its output to OCR dumps."""
    root = Path(paddledet_root).resolve()
    if not (root / "ppdet").is_dir():
        raise FileNotFoundError(f"PaddleDetection package not found under {root}")
    sys.path.insert(0, str(root))

    import paddle
    from ppdet.core.workspace import load_config, merge_config
    from ppdet.engine import Trainer

    rows = list(entries)
    paths = [str(Path(image_dir).resolve() / row.filename) for row in rows]
    paddle.set_device(device)

    load_config(str(Path(config).resolve()))
    cfg = merge_config(
        {
            "num_classes": 1,
            "TestDataset": {"dataset_dir": None, "anno_path": None},
        }
    )
    cfg.weights = str(Path(weights).resolve())
    cfg.use_gpu = device.startswith("gpu")
    cfg.use_xpu = cfg.use_npu = cfg.use_mlu = False
    trainer = Trainer(cfg, mode="test")
    trainer.load_weights(cfg.weights)

    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="navocr-paddledet-") as output_dir:
        batches = trainer.predict(
            paths,
            output_dir=output_dir,
            visualize=False,
            save_results=False,
        )
    elapsed_s = time.perf_counter() - start
    mean_elapsed_s = elapsed_s / len(rows) if rows else 0.0

    by_index: dict[int, list[OcrItem]] = {i: [] for i in range(len(rows))}
    seen_indices: set[int] = set()
    for batch in batches:
        image_ids = np.asarray(batch["im_id"]).reshape(-1)
        box_counts = np.asarray(batch["bbox_num"]).reshape(-1)
        boxes = np.asarray(batch["bbox"])
        offset = 0
        for image_id, count in zip(image_ids, box_counts):
            index = int(image_id)
            if index not in by_index:
                raise RuntimeError(f"PaddleDetection returned unknown image id {index}")
            if index in seen_indices:
                raise RuntimeError(f"PaddleDetection returned image id {index} twice")
            seen_indices.add(index)
            for class_id, score, x0, y0, x1, y1 in boxes[offset : offset + int(count)]:
                if int(class_id) != 0 or x1 <= x0 or y1 <= y0:
                    continue
                by_index[index].append(
                    OcrItem(
                        text="",
                        bbox=(float(x0), float(y0), float(x1 - x0), float(y1 - y0)),
                        score=float(score),
                    )
                )
            offset += int(count)
        if offset != len(boxes):
            raise RuntimeError(
                f"PaddleDetection returned {len(boxes)} boxes but counted {offset}"
            )

    missing_indices = set(by_index) - seen_indices
    if missing_indices:
        first = min(missing_indices)
        raise RuntimeError(
            f"PaddleDetection omitted {len(missing_indices)} image(s), first index {first}"
        )

    return [
        OcrResult(
            id=row.id,
            items=by_index[index],
            elapsed_s=mean_elapsed_s,
            error=None,
        )
        for index, row in enumerate(rows)
    ]
