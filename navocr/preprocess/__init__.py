"""NavOCR preprocessing pipeline.

This package generates COCO-format training datasets for navigation-relevant
text detection. It consumes images crawled from Maps and produces
labeled bounding boxes for store signboards via a four-stage pipeline:

    Stage 2  CLIPFilter      pending     → clip_pass | clip_fail
    Stage 3  OCRRunner       clip_pass   → ocr_done
    Stage 4  Matcher         ocr_done    → matched
    Stage 5  COCOExporter    matched     → exported

Data flows through two CSVs (manifest, detections) defined in manifest_io.
The runner module wires the stages together as a CLI (preprocess_navocr).
"""

from navocr.preprocess.clip_filter import CLIPFilter

...
from navocr.preprocess.clip_filter import CLIPFilter
from navocr.preprocess.coco_exporter import COCOExporter
from navocr.preprocess.manifest_io import (
    DetectionIO,
    DetectionRow,
    ManifestIO,
    ManifestRow,
    PipelineConfig,
)
from navocr.preprocess.matcher import Matcher, TranslationCache
from navocr.preprocess.ocr_runner import OCRRunner

__all__ = [
    "CLIPFilter",
    "COCOExporter",
    "DetectionIO",
    "DetectionRow",
    "ManifestIO",
    "ManifestRow",
    "Matcher",
    "OCRRunner",
    "PipelineConfig",
    "TranslationCache",
]
