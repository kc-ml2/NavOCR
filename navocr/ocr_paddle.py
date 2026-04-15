from __future__ import annotations

import cv2

from navocr.ocr_base import BaseOCR, PaddleOCRConfig


class PaddleOCRRecognizer(BaseOCR):
    def __init__(self, config: PaddleOCRConfig):
        super().__init__(config)
        from paddleocr import PaddleOCR

        self.ocr = self._create_ocr(PaddleOCR)

    def _create_ocr(self, paddle_ocr_cls):
        try:
            return paddle_ocr_cls(
                lang=self.config.language,
                use_angle_cls=True,
                det_db_thresh=0.25,
                det_db_box_thresh=0.4,
                rec_batch_num=32,
                enable_mkldnn=False,
            )
        except Exception:
            return paddle_ocr_cls(lang=self.config.language)

    def recognize(self, image_crop) -> str:
        if image_crop.size == 0:
            return self.NO_TEXT

        try:
            h, w = image_crop.shape[:2]
            if self.max_resize and max(h, w) > self.max_resize:
                scale = self.max_resize / max(h, w)
                image_crop = cv2.resize(image_crop, (int(w * scale), int(h * scale)))

            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
            image_crop = cv2.equalizeHist(image_crop)
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_GRAY2BGR)

            results = self.ocr.ocr(image_crop)
            recognized_text = self._extract_text(results)
            return recognized_text if recognized_text else self.NO_TEXT
        except Exception:
            return self.ERROR

    def _extract_text(self, results) -> str | None:
        if not results:
            return None

        result = results[0]
        if isinstance(result, dict) and 'rec_texts' in result and 'rec_scores' in result:
            texts = []
            for text, conf in zip(result['rec_texts'], result['rec_scores']):
                if conf > self.confidence_threshold:
                    text = text.strip()
                    if text:
                        texts.append(text)
            if texts:
                return ' '.join(' '.join(texts).split())

        if isinstance(result, list) and len(result) > 0:
            texts = []
            for line in result:
                if len(line) > 1:
                    text = line[1][0].strip()
                    conf = line[1][1]
                    if conf > self.confidence_threshold and text:
                        texts.append(text)
            if texts:
                return ' '.join(' '.join(texts).split())

        return None
