from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np
import onnxruntime as ort

from navocr.detector_base import BaseDetector, ONNXDetectorConfig


class ONNXDetector(BaseDetector):
    def __init__(self, config: ONNXDetectorConfig):
        super().__init__(config)
        if not self.config.model_path:
            raise ValueError('ONNX detector model path is required')
        if not os.path.isfile(self.config.model_path):
            raise FileNotFoundError(f'ONNX detector model not found: {self.config.model_path}')
        if self.config.imgsz is None:
            raise ValueError('ONNX detector imgsz is required')

        self.imgsz = int(self.config.imgsz)
        providers = self._resolve_providers(self.config.device)
        self.session = ort.InferenceSession(self.config.model_path, providers=providers)

        input_names = {i.name for i in self.session.get_inputs()}
        if input_names != {'image', 'scale_factor'}:
            raise RuntimeError(
                f'Expected inputs ["image", "scale_factor"], got {sorted(input_names)}'
            )
        output_names = [o.name for o in self.session.get_outputs()]
        for required in ('fetch_name_0', 'fetch_name_1'):
            if required not in output_names:
                raise RuntimeError(
                    f'Expected outputs to include "{required}", got {output_names}'
                )

    @staticmethod
    def _resolve_providers(device: str | None) -> list[str]:
        requested = (device or 'cpu').lower()
        available = ort.get_available_providers()
        if requested.startswith('cuda') and 'CUDAExecutionProvider' in available:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']

    def _preprocess(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        ratio = self.imgsz / max(h, w)
        new_h, new_w = int(h * ratio + 0.5), int(w * ratio + 0.5)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, None, fx=new_w / w, fy=new_h / h,
            interpolation=cv2.INTER_CUBIC,
        )

        padded = np.ones((self.imgsz, self.imgsz, 3), dtype=np.float32)
        padded *= np.array([123.675, 116.28, 103.53], dtype=np.float32)
        padded[:new_h, :new_w] = resized
        padded *= 1.0 / 255.0
        padded -= np.array([0.485, 0.456, 0.406])[None, None, :]
        padded /= np.array([0.229, 0.224, 0.225])[None, None, :]

        return {
            'image': np.ascontiguousarray(padded.transpose(2, 0, 1)[None]),
            'scale_factor': np.array([[new_h / h, new_w / w]], dtype=np.float32),
        }

    def infer(self, image_list):
        results = []
        for image_path in image_list:
            bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f'Cannot read image: {image_path}')
            results.append(self._infer_bgr(bgr))
        return results

    def infer_loaded_images(self, image_list: List[np.ndarray]):
        return [self._infer_bgr(bgr) for bgr in image_list]

    def _infer_bgr(self, bgr: np.ndarray):
        if bgr is None:
            raise ValueError('Input image is None')
        if not isinstance(bgr, np.ndarray):
            raise TypeError(f'Expected np.ndarray, got {type(bgr)!r}')
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f'Expected BGR image with shape [H, W, 3], got {bgr.shape}')

        boxes, bbox_num = self.session.run(
            ['fetch_name_0', 'fetch_name_1'],
            self._preprocess(bgr),
        )
        boxes = np.asarray(boxes, dtype=np.float32)
        count = bbox_num.item()
        if boxes.shape != (count, 6):
            raise RuntimeError(f'Expected bbox shape ({count}, 6), got {boxes.shape}')

        keep = (boxes[:, 0] >= 0) & (boxes[:, 1] >= self.detection_threshold)
        return {'bbox': boxes[keep]}
