from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class OCRConfig:
    backend: str | None = None
    language: str = 'en'
    confidence_threshold: float = 0.6
    max_resize: int | None = None
    device: str | None = None
    model_path: str | None = None


@dataclass
class PaddleOCRConfig(OCRConfig):
    pass


@dataclass
class OpenVINOOCRConfig(OCRConfig):
    dict_path: str | None = None
    rec_h: int | None = None
    rec_img_w: int | None = None
    rec_max_w: int | None = None


class BaseOCR(ABC):
    NO_TEXT = 'no_text_detected'
    ERROR = 'ocr_error'

    def __init__(self, config: OCRConfig):
        self.config = config
        self.confidence_threshold = self.config.confidence_threshold
        self.max_resize = self.config.max_resize

    @abstractmethod
    def recognize(self, image_crop) -> str:
        raise NotImplementedError
