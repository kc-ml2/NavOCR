from __future__ import annotations

import os
from typing import List

import numpy as np
import onnxruntime as ort

from navocr.detector_base import BaseDetector, ONNXDetectorConfig
from navocr.detector_preprocess import preprocess_ppyoloe, validate_preprocess_input_shape
from navocr.pipeline_utils import load_image


class ONNXDetector(BaseDetector):
    DETECTIONS_OUTPUT_NAME = 'fetch_name_0'
    DETECTION_COUNT_OUTPUT_NAME = 'fetch_name_1'

    def __init__(self, config: ONNXDetectorConfig):
        super().__init__(config)
        if not self.config.model_path:
            raise ValueError('ONNX detector model path is required')
        if not os.path.isfile(self.config.model_path):
            raise FileNotFoundError(f'ONNX detector model not found: {self.config.model_path}')
        if self.config.preprocess is None:
            raise ValueError('ONNX detector preprocessing config is required')

        providers = self._resolve_providers(self.config.device)
        self.session = ort.InferenceSession(self.config.model_path, providers=providers)

        inputs = {i.name: i for i in self.session.get_inputs()}
        if set(inputs) != {'image', 'scale_factor'}:
            raise RuntimeError(
                f'Expected inputs ["image", "scale_factor"], got {sorted(inputs)}'
            )
        validate_preprocess_input_shape(inputs['image'].shape, self.config.preprocess)
        output_names = [o.name for o in self.session.get_outputs()]
        for required in (
            self.DETECTIONS_OUTPUT_NAME,
            self.DETECTION_COUNT_OUTPUT_NAME,
        ):
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
        return preprocess_ppyoloe(bgr, self.config.preprocess)

    def infer(self, image_list):
        results = []
        for image_path in image_list:
            results.append(self._infer_bgr(load_image(image_path)))
        return results

    def infer_loaded_images(self, image_list: List[np.ndarray]):
        return [self._infer_bgr(bgr) for bgr in image_list]

    def _infer_bgr(self, bgr: np.ndarray):
        boxes, bbox_num = self.session.run(
            [self.DETECTIONS_OUTPUT_NAME, self.DETECTION_COUNT_OUTPUT_NAME],
            self._preprocess(bgr),
        )
        boxes = np.asarray(boxes, dtype=np.float32)
        count = bbox_num.item()
        if boxes.shape != (count, 6):
            raise RuntimeError(f'Expected bbox shape ({count}, 6), got {boxes.shape}')

        keep = (boxes[:, 0] >= 0) & (boxes[:, 1] >= self.detection_threshold)
        return {'bbox': boxes[keep]}
