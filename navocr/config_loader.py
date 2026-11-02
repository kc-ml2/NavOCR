from __future__ import annotations

import json
from pathlib import Path

import yaml

from navocr.detector_base import (
    DetectorConfig,
    ONNXDetectorConfig,
    OpenVINODetectorConfig,
    PaddleDetectorConfig,
    PyTorchDetectorConfig,
)
from navocr.ocr_base import OCRConfig, ONNXOCRConfig, OpenVINOOCRConfig, PaddleOCRConfig, PyTorchOCRConfig


def load_ros_parameters(path: str | Path, node_name: str = "/**") -> tuple[dict, Path]:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if node_name in payload and "ros__parameters" in payload[node_name]:
        return payload[node_name]["ros__parameters"], config_path

    if "/**" in payload and "ros__parameters" in payload["/**"]:
        return payload["/**"]["ros__parameters"], config_path

    for value in payload.values():
        if isinstance(value, dict) and "ros__parameters" in value:
            return value["ros__parameters"], config_path

    raise ValueError(f"No ros__parameters section found in {config_path}")


def infer_project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for parent in resolved.parents:
        if parent.name == "configs":
            return parent.parent
    return resolved.parent


def resolve_navocr_path(config_path: Path, value: str | None) -> str | None:
    if not value:
        return value

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    project_root = infer_project_root(config_path)
    return str((project_root / candidate).resolve())


def load_detector_config(path: str | Path, node_name: str = "/**") -> "DetectorConfig":
    params, config_path = load_ros_parameters(path, node_name=node_name)
    backend = params.get("detector_backend")
    annotation_path = infer_project_root(config_path) / (
        "configs/paddle/datasets/metadata/annotations/instances_test.json"
    )
    with annotation_path.open(encoding="utf-8") as handle:
        category_names = {category["id"]: category["name"] for category in json.load(handle)["categories"]}
    detector_config_path = resolve_navocr_path(config_path, params.get("detector_config_path"))
    preprocess = None
    if backend in ("onnx", "openvino", "pytorch"):
        if not detector_config_path:
            raise ValueError(f"{backend} detector_config_path is required")
        with Path(detector_config_path).open(encoding="utf-8") as handle:
            reader = yaml.safe_load(handle)["TestReader"]
        transforms = reader["sample_transforms"]
        expected = ("Decode", "Resize", "Pad", "NormalizeImage", "Permute")
        if tuple(name for transform in transforms for name in transform) != expected:
            raise ValueError(f"TestReader.sample_transforms must follow {expected}")
        if reader.get("fuse_normalize", False):
            raise ValueError("External PP-YOLOE preprocessing requires TestReader.fuse_normalize=False")
        stages = {name: options for transform in transforms for name, options in transform.items()}
        resize, pad, normalize = (stages[name] for name in ("Resize", "Pad", "NormalizeImage"))
        if pad["pad_mode"] != 0:
            raise ValueError("PP-YOLOE scale_factor coordinates require Pad.pad_mode=0 (right/bottom padding)")
        preprocess = {
            "target_size": tuple(resize["target_size"]),
            "keep_ratio": resize["keep_ratio"],
            "interp": resize["interp"],
            "pad_size": tuple(pad["size"]),
            "fill_value": tuple(pad["fill_value"]),
            "mean": tuple(normalize["mean"]),
            "std": tuple(normalize["std"]),
            "is_scale": normalize["is_scale"],
        }
    common_kwargs = {
        "backend": backend,
        "model_path": resolve_navocr_path(config_path, params.get("detector_model_path")),
        "detector_config_path": detector_config_path,
        "class_names": {0: category_names[1], 1: category_names[2]},
        "detection_threshold": float(params.get("detector_threshold", 0.5)),
        "output_dir": resolve_navocr_path(config_path, params.get("output_dir")) or "",
        "device": params.get("detector_device") or None,
        "preprocess": preprocess,
    }

    if backend == "paddle":
        return PaddleDetectorConfig(
            **common_kwargs,
            quiet_logs=bool(params.get("detector_quiet_logs", False)),
        )
    if backend in ("onnx", "openvino", "pytorch"):
        config_type = {
            "onnx": ONNXDetectorConfig,
            "openvino": OpenVINODetectorConfig,
            "pytorch": PyTorchDetectorConfig,
        }[backend]
        return config_type(**common_kwargs)
    return DetectorConfig(**common_kwargs)


def load_ocr_config(path: str | Path, node_name: str = "/**") -> "OCRConfig":
    params, config_path = load_ros_parameters(path, node_name=node_name)
    backend = params.get("ocr_backend")
    common_kwargs = {
        "backend": backend,
        "language": params.get("ocr_language", "en"),
        "confidence_threshold": float(params.get("ocr_threshold", 0.6)),
        "max_resize": (int(params["ocr_max_resize"]) if "ocr_max_resize" in params else None),
        "device": params.get("ocr_device") or None,
        "model_path": resolve_navocr_path(config_path, params.get("ocr_model_path")),
    }

    if backend == "paddle":
        return PaddleOCRConfig(**common_kwargs)
    if backend in ("onnx", "openvino", "pytorch"):
        config_type = {
            "onnx": ONNXOCRConfig,
            "openvino": OpenVINOOCRConfig,
            "pytorch": PyTorchOCRConfig,
        }[backend]
        return config_type(
            **common_kwargs,
            dict_path=resolve_navocr_path(config_path, params.get("ocr_dict_path")),
            rec_h=(int(params["ocr_rec_h"]) if "ocr_rec_h" in params else None),
            rec_img_w=(int(params["ocr_rec_img_w"]) if "ocr_rec_img_w" in params else None),
            rec_max_w=(int(params["ocr_rec_max_w"]) if "ocr_rec_max_w" in params else None),
        )
    return OCRConfig(**common_kwargs)
