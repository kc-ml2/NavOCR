from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from navocr.detector_base import PPYOLOEPreprocessConfig


def preprocess_ppyoloe(bgr: np.ndarray, config: PPYOLOEPreprocessConfig) -> dict[str, np.ndarray]:
    h, w = bgr.shape[:2]
    if config.keep_ratio:
        # Match PaddleDetection Resize, including its short/long-side semantics.
        ratio = min(min(config.target_size) / min(h, w), max(config.target_size) / max(h, w))
        new_h, new_w = int(h * ratio + 0.5), int(w * ratio + 0.5)
    else:
        new_h, new_w = config.target_size

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=config.interp)
    padded = np.full((*config.pad_size, 3), config.fill_value, dtype=np.float32)
    padded[:new_h, :new_w] = resized
    if config.is_scale:
        padded *= 1.0 / 255.0
    # In-place normalization preserves the native reader's float32 results.
    padded -= config.mean
    padded /= config.std
    return {
        'image': np.ascontiguousarray(padded.transpose(2, 0, 1)[None]),
        'scale_factor': np.array([[new_h / h, new_w / w]], dtype=np.float32),
    }


def validate_preprocess_input_shape(shape: Sequence, config: PPYOLOEPreprocessConfig) -> None:
    expected = (1, 3, *config.pad_size)
    if len(shape) != len(expected) or any(
        isinstance(actual, int) and actual != wanted for actual, wanted in zip(shape, expected)
    ):
        raise ValueError(f'Preprocessing produces {expected}, but model image input has shape {tuple(shape)}')
