"""NavOCR dataset generation pipeline.

This package generates COCO-format training datasets for navigation-relevant
text detection. It consumes images crawled from Maps and produces
labeled bounding boxes for store signboards via a four-stage pipeline:

    Stage 2  CLIPFilter      pending     → clip_filter_pass | clip_filter_fail
    Stage 3  OCRRunner       clip_filter_pass   → ocr_done
    Stage 4  OCRFilter       ocr_done    → ocr_filtered
    Stage 5  COCOExporter    ocr_filtered → exported

Data flows through two CSVs (manifest, detections) defined in manifest_io.
The runner module wires the stages together as a CLI (generate_navocr_dataset).
"""

from dataset_generator.clip_filter import CLIPFilter
from dataset_generator.coco_exporter import COCOExporter
from dataset_generator.manifest_io import (
    DetectionIO,
    DetectionRow,
    ManifestIO,
    ManifestRow,
    PipelineConfig,
)
from dataset_generator.ocr_filter import OCRFilter
from dataset_generator.ocr_runner import OCRRunner

__all__ = [
    "CLIPFilter",
    "COCOExporter",
    "DetectionIO",
    "DetectionRow",
    "ManifestIO",
    "ManifestRow",
    "OCRFilter",
    "OCRRunner",
    "PipelineConfig",
]
