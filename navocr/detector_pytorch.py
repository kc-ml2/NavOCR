from __future__ import annotations

import os
import sys
from typing import List

import cv2
import numpy as np

from navocr.detector_base import BaseDetector, PyTorchDetectorConfig


class PyTorchDetector(BaseDetector):
    def __init__(self, config: PyTorchDetectorConfig):
        super().__init__(config)
        if not self.config.model_path:
            raise ValueError('PyTorch detector model path is required')
        if not os.path.isfile(self.config.model_path):
            raise FileNotFoundError(f'PyTorch detector checkpoint not found: {self.config.model_path}')
        if not self.config.pytorch_config_path:
            raise ValueError('PyTorch detector pytorch_config_path is required')
        if not os.path.isfile(self.config.pytorch_config_path):
            raise FileNotFoundError(f'PyTorch detector config not found: {self.config.pytorch_config_path}')
        if self.config.imgsz is None:
            raise ValueError('PyTorch detector imgsz is required')

        self.imgsz = int(self.config.imgsz)
        self.device = self._resolve_device(self.config.device)

        engine_root = self._resolve_engine_root(self.config.engine_root)
        if engine_root and engine_root not in sys.path:
            sys.path.insert(0, engine_root)

        try:
            from engine.core import YAMLConfig
        except ImportError as exc:
            raise ImportError(
                'Failed to import engine.core. Set detector_engine_root in the YAML '
                'params to the directory that contains the "engine/" package.'
            ) from exc

        import torch
        import torch.nn as nn

        self._torch = torch

        cfg = YAMLConfig(self.config.pytorch_config_path, resume=self.config.model_path)
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

        ckpt = torch.load(self.config.model_path, map_location='cpu')
        state = (ckpt.get('ema') or {}).get('module') or ckpt.get('model') or ckpt
        cfg.model.load_state_dict(state)

        class _DeployModel(nn.Module):
            def __init__(self, model_cfg):
                super().__init__()
                self.model = model_cfg.model.deploy()
                self.post = model_cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                return self.post(self.model(images), orig_target_sizes)

        self.model = _DeployModel(cfg).to(self.device).eval()

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        import torch
        requested = (device or 'cuda').lower()
        if requested.startswith('cuda') and not torch.cuda.is_available():
            return 'cpu'
        return requested

    @staticmethod
    def _resolve_engine_root(configured: str | None) -> str | None:
        if configured:
            root = os.path.abspath(configured)
            if not os.path.isdir(os.path.join(root, 'engine')):
                raise FileNotFoundError(
                    f'detector_engine_root does not contain an "engine/" package: {root}'
                )
            return root

        # Default: NavOCR repo root (parent of the navocr package directory),
        # which holds the vendored engine/ tree.
        navocr_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        if os.path.isdir(os.path.join(navocr_root, 'engine')):
            return navocr_root
        return None

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
        return blob, ratio, pad_w, pad_h

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

        torch = self._torch
        orig_h, orig_w = bgr.shape[:2]
        blob, ratio, pad_w, pad_h = self._preprocess(bgr)

        with torch.no_grad():
            blob_t = torch.from_numpy(blob).to(self.device)
            orig_size = torch.tensor([[self.imgsz, self.imgsz]], dtype=torch.long).to(self.device)
            labels_t, boxes_t, scores_t = self.model(blob_t, orig_size)

        labels = labels_t[0].cpu().numpy()
        boxes = boxes_t[0].cpu().numpy()
        scores = scores_t[0].cpu().numpy()
        boxes = self._map_boxes_to_original(boxes, ratio, pad_w, pad_h, orig_w, orig_h)

        keep = scores > self.detection_threshold
        packed = []
        for lbl, score, box in zip(labels[keep], scores[keep], boxes[keep]):
            x1, y1, x2, y2 = box.astype(np.float32).tolist()
            packed.append([float(lbl), float(score), x1, y1, x2, y2])

        return {'bbox': np.asarray(packed, dtype=np.float32)}
