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
        if 'images' not in input_names or 'orig_target_sizes' not in input_names:
            raise RuntimeError(
                f'Expected inputs ["images", "orig_target_sizes"], got {sorted(input_names)}'
            )
        output_names = [o.name for o in self.session.get_outputs()]
        for required in ('labels', 'boxes', 'scores'):
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

    @staticmethod
    def _letterbox(image: np.ndarray, size: int):
        h, w = image.shape[:2]
        ratio = min(size / w, size / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_w = (size - new_w) // 2
        pad_h = (size - new_h) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        return padded, ratio, pad_w, pad_h

    def _preprocess(self, bgr: np.ndarray):
        padded, ratio, pad_w, pad_h = self._letterbox(bgr, self.imgsz)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = rgb.transpose(2, 0, 1)[np.newaxis]
        orig_size = np.array([[self.imgsz, self.imgsz]], dtype=np.int64)
        return blob, orig_size, ratio, pad_w, pad_h

    @staticmethod
    def _map_boxes_to_original(boxes: np.ndarray, ratio: float, pad_w: int, pad_h: int,
                               orig_w: int, orig_h: int) -> np.ndarray:
        boxes = boxes.copy()
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes /= ratio
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)
        return boxes

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

        orig_h, orig_w = bgr.shape[:2]
        blob, orig_size, ratio, pad_w, pad_h = self._preprocess(bgr)
        labels, boxes, scores = self.session.run(
            ['labels', 'boxes', 'scores'],
            {'images': blob, 'orig_target_sizes': orig_size},
        )

        labels = labels[0]
        boxes = boxes[0]
        scores = scores[0]
        boxes = self._map_boxes_to_original(boxes, ratio, pad_w, pad_h, orig_w, orig_h)

        keep = scores > self.detection_threshold
        packed = []
        for lbl, score, box in zip(labels[keep], scores[keep], boxes[keep]):
            x1, y1, x2, y2 = box.astype(np.float32).tolist()
            packed.append([float(lbl), float(score), x1, y1, x2, y2])

        return {'bbox': np.asarray(packed, dtype=np.float32)}
