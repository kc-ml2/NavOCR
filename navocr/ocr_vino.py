from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from navocr.ocr_base import BaseOCR, OpenVINOOCRConfig


class OpenVINOOCRRecognizer(BaseOCR):
    def __init__(self, config: OpenVINOOCRConfig):
        super().__init__(config)
        import openvino as ov

        if not self.config.model_path:
            raise ValueError('OpenVINO OCR model path is required')
        if not self.config.dict_path:
            raise ValueError('OpenVINO OCR dictionary path is required')
        if self.config.rec_h is None or self.config.rec_img_w is None or self.config.rec_max_w is None:
            raise ValueError('OpenVINO OCR preprocessing dimensions are required')

        core = ov.Core()
        compile_config = {'PERFORMANCE_HINT': 'LATENCY'}
        if str(self.config.device or 'CPU').upper().startswith('GPU'):
            compile_config['INFERENCE_PRECISION_HINT'] = 'f32'

        self.rec_model = core.compile_model(
            core.read_model(self.config.model_path),
            self.config.device or 'CPU',
            config=compile_config,
        )
        self.char_list = self._load_char_list(self.config.dict_path)

    @staticmethod
    def _load_char_list(dict_path: str) -> list[str]:
        path = Path(dict_path)
        if not path.exists():
            raise FileNotFoundError(f'Character dictionary not found: {dict_path}')

        if path.suffix.lower() in ('.yml', '.yaml'):
            import yaml

            with open(path, encoding='utf-8') as handle:
                cfg = yaml.safe_load(handle)
            chars = cfg['PostProcess']['character_dict']
            return [''] + [str(char) for char in chars] + [' ']

        with open(path, encoding='utf-8') as handle:
            chars = [line.rstrip('\n') for line in handle]
        return [''] + chars + [' ']

    def recognize(self, image_crop) -> str:
        if image_crop.size == 0:
            return self.NO_TEXT

        try:
            rec_in = self._preprocess(image_crop)
            rec_out = list(self.rec_model.infer_new_request({0: rec_in}).values())[0]
            text, conf = self._ctc_decode_with_conf(rec_out[0], self.char_list)
            text = ' '.join(text.strip().split())
            if text and conf >= self.confidence_threshold:
                return text
            return self.NO_TEXT
        except Exception:
            return self.ERROR

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        h, w = crop_bgr.shape[:2]
        max_wh = max(self.config.rec_img_w / self.config.rec_h, w / h)
        img_w = min(int(self.config.rec_h * max_wh), self.config.rec_max_w)
        resized_w = min(img_w, int(math.ceil(self.config.rec_h * w / h)))
        resized = cv2.resize(crop_bgr, (resized_w, self.config.rec_h))
        img = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        img = (img - 0.5) / 0.5
        canvas = np.zeros((3, self.config.rec_h, img_w), dtype=np.float32)
        canvas[:, :, :resized_w] = img
        return canvas[np.newaxis]

    @staticmethod
    def _ctc_decode_with_conf(probs: np.ndarray, char_list: list[str]) -> tuple[str, float]:
        indices = np.argmax(probs, axis=-1)
        best = np.max(probs, axis=-1)
        result = []
        confs = []
        prev = -1
        for idx, score in zip(indices, best):
            idx = int(idx)
            if idx != prev and idx != 0:
                result.append(char_list[idx])
                confs.append(float(score))
            prev = idx
        text = ''.join(result)
        conf = float(np.mean(confs)) if confs else 0.0
        return text, conf
