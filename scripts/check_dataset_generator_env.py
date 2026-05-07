"""NavOCR dataset generator environment check.

Run from project root:
    python scripts/check_dataset_generator_env.py
"""
from __future__ import annotations

import os
import sys
import platform

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_env = Path(".env")
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        _k = _k.strip()
        _v = _v.strip().strip('"').strip("'")
        if _k and _k not in os.environ:
            os.environ[_k] = _v


def ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")


def warn(msg: str) -> None:
    print(f"  \033[33m!\033[0m {msg}")


def fail(msg: str) -> None:
    print(f"  \033[31m✗\033[0m {msg}")


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


errors: list[str] = []
warnings_: list[str] = []


# 1. Python & platform
section("1. Python & platform")
print(f"  python   : {sys.version.split()[0]}")
print(f"  platform : {platform.system()} {platform.machine()}")
py_minor = sys.version_info[:2]
if py_minor < (3, 8):
    fail(f"Python {py_minor[0]}.{py_minor[1]} too old (need 3.8+)")
    errors.append("python_version")
elif py_minor >= (3, 13):
    warn(f"Python {py_minor[0]}.{py_minor[1]} may not have PaddlePaddle wheels")
    warnings_.append("python_version")
else:
    ok(f"Python {py_minor[0]}.{py_minor[1]} compatible")


# 2. Core required dependencies
section("2. Required dependencies")
core_deps = [
    ("PIL", "Pillow"),
    ("textdistance", "textdistance"),
]
for module, package in core_deps:
    try:
        m = __import__(module)
        ver = getattr(m, "__version__", "?")
        ok(f"{package:15s} {ver}")
    except ImportError:
        fail(f"{package:15s} NOT INSTALLED  →  pip install {package}")
        errors.append(package)


# 3. CLIP dependencies (transformers + torch)
section("3. CLIP filter (Stage 2)")
try:
    import torch
    ok(f"torch          {torch.__version__}  (mps={torch.backends.mps.is_available() if hasattr(torch.backends,'mps') else False}, cuda={torch.cuda.is_available()})")
except ImportError:
    fail("torch          NOT INSTALLED  →  pip install torch")
    errors.append("torch")

try:
    import transformers
    ok(f"transformers   {transformers.__version__}")
except ImportError:
    fail("transformers   NOT INSTALLED  →  pip install 'transformers>=4.30'")
    errors.append("transformers")


# 4. PaddleOCR (Stage 3)
section("4. OCR (Stage 3)")
try:
    import paddle
    ok(f"paddle         {paddle.__version__}  (device={paddle.device.get_device()})")
except ImportError:
    fail("paddle         NOT INSTALLED  →  pip install paddlepaddle")
    errors.append("paddle")

try:
    import paddleocr
    ok(f"paddleocr      {paddleocr.__version__}")
except ImportError:
    fail("paddleocr      NOT INSTALLED  →  pip install paddleocr")
    errors.append("paddleocr")


# 5. NavOCR dataset generator modules
section("5. NavOCR dataset generator modules")
modules = [
    "dataset_generator.manifest_io",
    "dataset_generator.ocr_filter",
    "dataset_generator.coco_exporter",
    "dataset_generator.runner",
    "dataset_generator.clip_filter",
    "dataset_generator.ocr_runner",
]
for mod in modules:
    try:
        __import__(mod)
        ok(mod)
    except ImportError as e:
        fail(f"{mod}: {e}")
        errors.append(mod)


# 6. CLI entry point
section("6. CLI entry point")
import shutil
exe = shutil.which("generate_navocr_dataset")
if exe:
    ok(f"generate_navocr_dataset available at {exe}")
else:
    warn("generate_navocr_dataset CLI not on PATH (use `python -m dataset_generator.runner` or `pip install -e .`)")
    warnings_.append("cli")


# 7. Functional smoke test
section("7. Functional smoke test")
try:
    from dataset_generator.manifest_io import PipelineConfig
    from dataset_generator.ocr_filter import OCRFilter
    from dataset_generator.coco_exporter import COCOExporter
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = PipelineConfig("smoke", str(tmp), str(tmp), str(tmp), similarity_threshold=0.5)

        # Levenshtein
        f = OCRFilter(cfg)
        score, ftype = f.best_score("starbucks", "Starbucks")
        assert score >= 0.99, score

        # bbox conversion
        e = COCOExporter(cfg)
        bbox = e.polygon_to_bbox(10, 20, 110, 20, 110, 50, 10, 50)
        assert bbox == [10, 20, 100, 30], bbox

        ok(f"OCR filtering works (score={score:.3f}, type={ftype})")
        ok(f"polygon→bbox conversion works ({bbox})")
except Exception as e:
    fail(f"smoke test failed: {e}")
    errors.append("smoke")


# Summary
section("Summary")
if errors:
    print(f"\n  \033[31m{len(errors)} error(s)\033[0m: {', '.join(errors)}")
    print("  → Pipeline NOT ready. Install missing dependencies above.")
    sys.exit(1)
elif warnings_:
    print(f"\n  \033[33m{len(warnings_)} warning(s)\033[0m: {', '.join(warnings_)}")
    print("  → Pipeline ready with caveats (optional features may be disabled).")
    sys.exit(0)
else:
    print("\n  \033[32mAll checks passed\033[0m — pipeline fully operational.")
    sys.exit(0)
