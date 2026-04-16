from __future__ import annotations

import cv2
from paddleocr import TextRecognition

from navocr.ocr_base import BaseOCR, PaddleOCRConfig


class PaddleOCRRecognizer(BaseOCR):
    def __init__(self, config: PaddleOCRConfig):
        super().__init__(config)

        self.ocr = self._create_ocr(TextRecognition)

    def _create_ocr(self, text_recognition_cls):
        kwargs = {}
        if self.config.model_path:
            kwargs['model_dir'] = self.config.model_path
        else:
            kwargs['model_name'] = self._resolve_recognition_model_name(self.config.language)
        if self.config.device:
            kwargs['device'] = str(self.config.device).lower()
        return text_recognition_cls(**kwargs)

    @staticmethod
    def _resolve_recognition_model_name(language: str | None) -> str:
        lang = (language or 'en').lower()
        if lang in {'ch', 'chinese_cht', 'japan'}:
            return 'PP-OCRv5_server_rec'
        if lang == 'en':
            return 'en_PP-OCRv5_mobile_rec'
        if lang == 'korean':
            return 'korean_PP-OCRv5_mobile_rec'
        if lang == 'th':
            return 'th_PP-OCRv5_mobile_rec'
        if lang == 'el':
            return 'el_PP-OCRv5_mobile_rec'
        if lang == 'te':
            return 'te_PP-OCRv5_mobile_rec'
        if lang == 'ta':
            return 'ta_PP-OCRv5_mobile_rec'
        return 'latin_PP-OCRv5_mobile_rec'

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

            results = self.ocr.predict(image_crop)
            recognized_text = self._extract_text(results)
            return recognized_text if recognized_text else self.NO_TEXT
        except Exception:
            return self.ERROR

    def _extract_text(self, results) -> str | None:
        if not results:
            return None

        result = results[0]
        if isinstance(result, dict) and 'rec_text' in result and 'rec_score' in result:
            return self._join_texts(result['rec_text'], result['rec_score'])

        if isinstance(result, dict) and 'rec_texts' in result and 'rec_scores' in result:
            return self._join_texts(result['rec_texts'], result['rec_scores'])

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

    def _join_texts(self, texts, scores) -> str | None:
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(scores, (list, tuple)):
            scores = [scores]

        picked = []
        for text, conf in zip(texts, scores):
            if float(conf) > self.confidence_threshold:
                text = str(text).strip()
                if text:
                    picked.append(text)
        if picked:
            return ' '.join(' '.join(picked).split())
        return None
