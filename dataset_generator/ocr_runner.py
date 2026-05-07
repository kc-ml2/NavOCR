from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .manifest_io import DetectionIO, DetectionRow, ManifestIO, ManifestRow, PipelineConfig

logger = logging.getLogger(__name__)


class OCRRunner:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._ocr: Any = None

    def _load_ocr(self) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "paddleocr is not installed. "
                "Install it manually with the appropriate PaddlePaddle variant for your hardware "
                "(CPU: `pip install paddlepaddle paddleocr`, "
                "GPU: `pip install paddlepaddle-gpu paddleocr`). "
                "See https://github.com/PaddlePaddle/PaddleOCR for details."
            ) from exc
        self._ocr = PaddleOCR(use_textline_orientation=True, lang="en")

    def run_image(self, image_path: Path) -> list[DetectionRow]:
        if self._ocr is None:
            self._load_ocr()

        results = self._ocr.predict(str(image_path))
        if not results:
            return []

        detections: list[DetectionRow] = []
        for res in results:
            polys = res.get("rec_polys") or res.get("dt_polys") or []
            texts = res.get("rec_texts") or []
            scores = res.get("rec_scores") or []
            for poly, text, score in zip(polys, texts, scores):
                if poly is None or text is None or not str(text).strip():
                    continue
                pts = list(poly)
                if len(pts) != 4:
                    continue
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts
                ocr_text = str(text)
                ocr_text_normalized = ocr_text.replace(" ", "").lower()
                detections.append(
                    DetectionRow(
                        image_filename=image_path.name,
                        ocr_text=ocr_text,
                        ocr_text_normalized=ocr_text_normalized,
                        ocr_confidence=float(score),
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        x3=float(x3),
                        y3=float(y3),
                        x4=float(x4),
                        y4=float(y4),
                    )
                )

        return detections

    def run(
        self,
        rows: list[ManifestRow],
        image_dir: Path,
        manifest_io: ManifestIO,
        detection_io: DetectionIO,
    ) -> None:
        for row in rows:
            if row.status != "clip_filter_pass":
                continue

            image_path = image_dir / row.image_filename
            try:
                detections = self.run_image(image_path)
                if detections:
                    detection_io.append_rows(detections)
                manifest_io.update_row(
                    row.image_filename,
                    ocr_box_count=len(detections),
                    status="ocr_done",
                )
            except Exception:
                logger.exception("OCR failed for %s", row.image_filename)
                manifest_io.update_row(row.image_filename, status="error")
