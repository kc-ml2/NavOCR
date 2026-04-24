"""
Inference-only RT-DETR engine package for NavOCR.

This package intentionally excludes training-only modules so standalone
inference does not require training dependencies.
"""

from . import rtv4

from .backbone import *

from .backbone import (
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
    get_activation,
)
