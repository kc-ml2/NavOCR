from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch

from navocr.ocr_base import BaseOCR, PyTorchOCRConfig


class PyTorchOCRRecognizer(BaseOCR):
    def __init__(self, config: PyTorchOCRConfig):
        super().__init__(config)

        if not self.config.model_path:
            raise ValueError('PyTorch OCR model path is required')
        if not Path(self.config.model_path).is_file():
            raise FileNotFoundError(f'PyTorch OCR model not found: {self.config.model_path}')
        if not self.config.dict_path:
            raise ValueError('PyTorch OCR dictionary path is required')
        if self.config.rec_h is None or self.config.rec_img_w is None or self.config.rec_max_w is None:
            raise ValueError('PyTorch OCR preprocessing dimensions are required')

        self.device = self._resolve_device(self.config.device)
        self.model = torch.load(
            self.config.model_path, map_location='cpu', weights_only=False,
        ).to(self.device).eval()
        self.char_list = self._load_char_list(self.config.dict_path)

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        requested = (device or 'cpu').lower()
        if requested.startswith('cuda') and not torch.cuda.is_available():
            return 'cpu'
        return requested

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
            rec_in = torch.from_numpy(self._preprocess(image_crop)).to(self.device)
            with torch.inference_mode():
                rec_out = self.model(rec_in).detach().cpu().numpy()
            if rec_out.ndim != 3 or rec_out.shape[0] != 1 or rec_out.shape[2] != len(self.char_list):
                raise RuntimeError(f'Unexpected OCR output shape: {rec_out.shape}')
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
