from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DetectorConfig:
    backend: str | None = None
    model_path: str | None = None
    detection_threshold: float = 0.5
    device: str | None = None
    output_dir: str = ""
    # Kept on detector config for compatibility because PaddleDetection's
    # Trainer.predict() still expects an output_dir at inference time, and the
    # same setting is also consumed as the runtime output directory in ROS and
    # standalone entry points.


@dataclass
class PaddleDetectorConfig(DetectorConfig):
    detector_config_path: str | None = None
    quiet_logs: bool = False


@dataclass
class OpenVINODetectorConfig(DetectorConfig):
    imgsz: int | None = None


@dataclass
class ONNXDetectorConfig(DetectorConfig):
    imgsz: int | None = None


class BaseDetector(ABC):
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.detection_threshold = self.config.detection_threshold
        self.output_dir = self.config.output_dir

    @abstractmethod
    def infer(self, image_list):
        raise NotImplementedError

    def infer_loaded_images(self, image_list):
        return None
