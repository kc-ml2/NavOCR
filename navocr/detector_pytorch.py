from __future__ import annotations

import os
from typing import List

import numpy as np
import torch

from navocr.detector_base import BaseDetector, PyTorchDetectorConfig
from navocr.detector_preprocess import preprocess_ppyoloe
from navocr.pipeline_utils import load_image


class PyTorchDetector(BaseDetector):
    def __init__(self, config: PyTorchDetectorConfig):
        super().__init__(config)
        if not self.config.model_path:
            raise ValueError('PyTorch detector model path is required')
        if not os.path.isfile(self.config.model_path):
            raise FileNotFoundError(f'PyTorch detector model not found: {self.config.model_path}')
        if self.config.preprocess is None:
            raise ValueError('PyTorch detector preprocessing config is required')

        self.device = self._resolve_device(self.config.device)
        self.model = torch.load(
            self.config.model_path, map_location='cpu', weights_only=False,
        ).to(self.device).eval()

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        requested = (device or 'cuda').lower()
        if requested.startswith('cuda') and not torch.cuda.is_available():
            return 'cpu'
        return requested

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
        feed = self._preprocess(bgr)
        scale_factor = torch.from_numpy(feed['scale_factor']).to(self.device)
        image = torch.from_numpy(feed['image']).to(self.device)

        with torch.inference_mode():
            detections, detection_counts = self.model(scale_factor, image)

        detections = np.asarray(detections.detach().cpu().numpy(), dtype=np.float32)
        detection_count = detection_counts.item()
        if detections.shape != (detection_count, 6):
            raise RuntimeError(
                f'Expected detections shape ({detection_count}, 6), got {detections.shape}'
            )

        keep = (detections[:, 0] >= 0) & (detections[:, 1] >= self.detection_threshold)
        return {'bbox': detections[keep]}
