from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DetectorConfig:
    backend: str | None = None
    model_path: str | None = None
    detector_config_path: str | None = None
    class_names: dict[int, str] = field(default_factory=dict)
    detection_threshold: float = 0.5
    device: str | None = None
    output_dir: str = ""
    # Kept on detector config for compatibility because PaddleDetection's
    # Trainer.predict() still expects an output_dir at inference time, and the
    # same setting is also consumed as the runtime output directory in ROS and
    # standalone entry points.
    preprocess: dict | None = None


@dataclass
class PaddleDetectorConfig(DetectorConfig):
    quiet_logs: bool = False


@dataclass
class OpenVINODetectorConfig(DetectorConfig):
    pass


@dataclass
class PyTorchDetectorConfig(DetectorConfig):
    pass


@dataclass
class ONNXDetectorConfig(DetectorConfig):
    pass


class BaseDetector(ABC):
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.class_names = config.class_names
        self.detection_threshold = self.config.detection_threshold
        self.output_dir = self.config.output_dir

    def get_class_name(self, class_id: int) -> str:
        return self.class_names.get(class_id, f"class_{class_id}")

    @abstractmethod
    def infer(self, image_list):
        raise NotImplementedError

    def infer_loaded_images(self, image_list):  # pylint: disable=unused-argument
        return None
