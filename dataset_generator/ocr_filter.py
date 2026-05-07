from __future__ import annotations

import logging
from typing import Optional

import textdistance as td

from .manifest_io import DetectionIO, DetectionRow, ManifestIO, ManifestRow, PipelineConfig

logger = logging.getLogger(__name__)


class OCRFilter:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def _compute_score(self, ocr_norm: str, store_name: str) -> float:
        ocr_lower = ocr_norm.lower()
        scores: list[float] = []
        for word in store_name.split():
            scores.append(td.levenshtein.normalized_similarity(word.lower(), ocr_lower))
        combined = store_name.replace(" ", "").lower()
        scores.append(td.levenshtein.normalized_similarity(combined, ocr_lower))
        return max(scores) if scores else 0.0

    def best_score(
        self,
        ocr_norm: str,
        store_name: str,
    ) -> tuple[float, str]:
        return self._compute_score(ocr_norm, store_name), "original"

    def run(
        self,
        rows: list[ManifestRow],
        manifest_io: ManifestIO,
        detection_io: DetectionIO,
    ) -> None:
        ocr_done = [r for r in rows if r.status == "ocr_done"]
        if not ocr_done:
            return

        all_detections = detection_io.read_all()
        detections_by_image: dict[str, list[DetectionRow]] = {}
        for det in all_detections:
            detections_by_image.setdefault(det.image_filename, []).append(det)

        for row in ocr_done:
            try:
                image_dets = detections_by_image.get(row.image_filename, [])

                for det in image_dets:
                    score, mtype = self.best_score(
                        det.ocr_text_normalized,
                        row.store_name,
                    )
                    det.levenshtein_score = score
                    det.ocr_filter_type = mtype
                    det.is_prominent_sign = score >= self._config.similarity_threshold

                prominent = [d for d in image_dets if d.is_prominent_sign]
                num_label_boxes = len(prominent)

                top_ocr_text: Optional[str] = None
                best_lev_score: Optional[float] = None
                if prominent:
                    top_det = max(
                        prominent,
                        key=lambda d: d.levenshtein_score or 0.0,
                    )
                    top_ocr_text = top_det.ocr_text
                    best_lev_score = top_det.levenshtein_score

                manifest_io.update_row(
                    row.image_filename,
                    status="ocr_filtered",
                    num_label_boxes=num_label_boxes,
                    top_ocr_text=top_ocr_text,
                    best_levenshtein_score=best_lev_score,
                )

            except Exception as exc:
                logger.exception("OCR filter error for image %r: %s", row.image_filename, exc)
                manifest_io.update_row(row.image_filename, status="error")

        if ocr_done:
            detection_io.write_all(all_detections)
