from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


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

    @abstractmethod
    def infer(self, image_list, visualize: bool = False, save_results: bool = False):
        raise NotImplementedError
