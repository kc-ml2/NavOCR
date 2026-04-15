from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class DetectorConfig:
    backend: str | None = None
    model_path: str | None = None
    detection_threshold: float = 0.5
    output_dir: str = ""
    device: str | None = None


@dataclass
class PaddleDetectorConfig(DetectorConfig):
    detector_config_path: str | None = None


@dataclass
class OpenVINODetectorConfig(DetectorConfig):
    imgsz: int | None = None


class BaseDetector(ABC):
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.detection_threshold = self.config.detection_threshold
        self.output_dir = self.config.output_dir

    @staticmethod
    def normalize_image_list(image_list: Sequence[str] | Iterable[str]) -> list[str]:
        if isinstance(image_list, list):
            return image_list
        return list(image_list)

    @abstractmethod
    def infer(self, image_list, visualize: bool = False, save_results: bool = False):
        raise NotImplementedError
