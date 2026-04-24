from __future__ import annotations

import importlib

from navocr.detector_base import DetectorConfig
from navocr.ocr_base import OCRConfig


DETECTOR_BACKENDS = {
    'paddle': 'navocr.detector_paddle:PaddleDetector',
    'openvino': 'navocr.detector_vino:OpenVINODetector',
    'onnx': 'navocr.detector_onnx:ONNXDetector',
}

OCR_BACKENDS = {
    'paddle': 'navocr.ocr_paddle:PaddleOCRRecognizer',
    'openvino': 'navocr.ocr_vino:OpenVINOOCRRecognizer',
    'onnx': 'navocr.ocr_onnx:ONNXOCRRecognizer',
}


def load_backend_class(registry: dict[str, str], backend_name: str):
    target = registry.get(backend_name)
    if target is None:
        raise ValueError(f'Unsupported backend: {backend_name}')

    module_name, class_name = target.split(':', maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def create_detector(config: DetectorConfig):
    if not config.backend:
        raise ValueError('Detector backend is required')

    detector_cls = load_backend_class(DETECTOR_BACKENDS, config.backend)
    return detector_cls(config)


def create_ocr(config: OCRConfig):
    if not config.backend:
        raise ValueError('OCR backend is required')

    ocr_cls = load_backend_class(OCR_BACKENDS, config.backend)
    return ocr_cls(config)
