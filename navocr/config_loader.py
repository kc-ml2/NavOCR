from __future__ import annotations

from pathlib import Path

import yaml
from navocr.detector_base import (
    DetectorConfig,
    ONNXDetectorConfig,
    OpenVINODetectorConfig,
    PaddleDetectorConfig,
    PyTorchDetectorConfig,
)
from navocr.ocr_base import OCRConfig, ONNXOCRConfig, OpenVINOOCRConfig, PaddleOCRConfig


def load_ros_parameters(path: str | Path, node_name: str = '/**') -> tuple[dict, Path]:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, encoding='utf-8') as handle:
        payload = yaml.safe_load(handle) or {}

    if node_name in payload and 'ros__parameters' in payload[node_name]:
        return payload[node_name]['ros__parameters'], config_path

    if '/**' in payload and 'ros__parameters' in payload['/**']:
        return payload['/**']['ros__parameters'], config_path

    for value in payload.values():
        if isinstance(value, dict) and 'ros__parameters' in value:
            return value['ros__parameters'], config_path

    raise ValueError(f'No ros__parameters section found in {config_path}')


def infer_project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for parent in resolved.parents:
        if parent.name == 'configs':
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


def load_detector_config(path: str | Path, node_name: str = '/**') -> 'DetectorConfig':
    params, config_path = load_ros_parameters(path, node_name=node_name)
    backend = params.get('detector_backend')
    common_kwargs = dict(
        backend=backend,
        model_path=resolve_navocr_path(config_path, params.get('detector_model_path')),
        detection_threshold=float(params.get('detector_threshold', 0.5)),
        output_dir=resolve_navocr_path(config_path, params.get('output_dir')) or '',
        device=params.get('detector_device') or None,
    )

    if backend == 'paddle':
        return PaddleDetectorConfig(
            **common_kwargs,
            detector_config_path=resolve_navocr_path(config_path, params.get('detector_config_path')),
            quiet_logs=bool(params.get('detector_quiet_logs', False)),
        )
    if backend == 'openvino':
        return OpenVINODetectorConfig(
            **common_kwargs,
            imgsz=(int(params['detector_imgsz']) if 'detector_imgsz' in params else None),
        )
    if backend == 'pytorch':
        return PyTorchDetectorConfig(
            **common_kwargs,
            pytorch_config_path=resolve_navocr_path(config_path, params.get('detector_pytorch_config_path')),
            engine_root=resolve_navocr_path(config_path, params.get('detector_engine_root')),
            imgsz=(int(params['detector_imgsz']) if 'detector_imgsz' in params else None),
        )
    if backend == 'onnx':
        return ONNXDetectorConfig(
            **common_kwargs,
            imgsz=(int(params['detector_imgsz']) if 'detector_imgsz' in params else None),
        )
    return DetectorConfig(**common_kwargs)


def load_ocr_config(path: str | Path, node_name: str = '/**') -> 'OCRConfig':
    params, config_path = load_ros_parameters(path, node_name=node_name)
    backend = params.get('ocr_backend')
    common_kwargs = dict(
        backend=backend,
        language=params.get('ocr_language', 'en'),
        confidence_threshold=float(params.get('ocr_threshold', 0.6)),
        max_resize=(int(params['ocr_max_resize']) if 'ocr_max_resize' in params else None),
        device=params.get('ocr_device') or None,
        model_path=resolve_navocr_path(config_path, params.get('ocr_model_path')),
    )

    if backend == 'paddle':
        return PaddleOCRConfig(**common_kwargs)
    if backend == 'openvino':
        return OpenVINOOCRConfig(
            **common_kwargs,
            dict_path=resolve_navocr_path(config_path, params.get('ocr_dict_path')),
            rec_h=(int(params['ocr_rec_h']) if 'ocr_rec_h' in params else None),
            rec_img_w=(int(params['ocr_rec_img_w']) if 'ocr_rec_img_w' in params else None),
            rec_max_w=(int(params['ocr_rec_max_w']) if 'ocr_rec_max_w' in params else None),
        )
    if backend == 'onnx':
        return ONNXOCRConfig(
            **common_kwargs,
            dict_path=resolve_navocr_path(config_path, params.get('ocr_dict_path')),
            rec_h=(int(params['ocr_rec_h']) if 'ocr_rec_h' in params else None),
            rec_img_w=(int(params['ocr_rec_img_w']) if 'ocr_rec_img_w' in params else None),
            rec_max_w=(int(params['ocr_rec_max_w']) if 'ocr_rec_max_w' in params else None),
        )
    return OCRConfig(**common_kwargs)
