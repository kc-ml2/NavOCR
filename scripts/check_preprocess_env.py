"""NavOCR preprocessing pipeline environment check.

Run from project root:
    python scripts/check_preprocess_env.py
"""
from __future__ import annotations

import os
import sys
import platform

from pathlib import Path
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


# 5. DeepL (Stage 4, optional)
section("5. DeepL translation (Stage 4, optional)")
try:
    import deepl
    ok(f"deepl          {deepl.__version__}")
except ImportError:
    warn("deepl          not installed (translation will be skipped)")
    warnings_.append("deepl")
except AttributeError:
    ok("deepl          installed (version unknown)")

if os.environ.get("DEEPL_AUTH_KEY"):
    ok("DEEPL_AUTH_KEY env var is set")
else:
    warn("DEEPL_AUTH_KEY not set (translation will be skipped at runtime)")
    warnings_.append("deepl_key")


# 6. NavOCR preprocess modules
section("6. NavOCR preprocess modules")
modules = [
    "navocr.preprocess.manifest_io",
    "navocr.preprocess.matcher",
    "navocr.preprocess.coco_exporter",
    "navocr.preprocess.runner",
    "navocr.preprocess.clip_filter",
    "navocr.preprocess.ocr_runner",
]
for mod in modules:
    try:
        __import__(mod)
        ok(mod)
    except ImportError as e:
        fail(f"{mod}: {e}")
        errors.append(mod)


# 7. CLI entry point
section("7. CLI entry point")
import shutil
exe = shutil.which("preprocess_navocr")
if exe:
    ok(f"preprocess_navocr available at {exe}")
else:
    warn("preprocess_navocr CLI not on PATH (use `python -m navocr.preprocess.runner` or `pip install -e .`)")
    warnings_.append("cli")


# 8. Functional smoke test
section("8. Functional smoke test")
try:
    from navocr.preprocess.manifest_io import ManifestRow, DetectionRow, PipelineConfig
    from navocr.preprocess.matcher import Matcher, TranslationCache
    from navocr.preprocess.coco_exporter import COCOExporter
    import tempfile, json
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = PipelineConfig("smoke", str(tmp), str(tmp), str(tmp), similarity_threshold=0.5)
        cache = TranslationCache(tmp / "cache.json")

        # Levenshtein
        m = Matcher(cfg, cache)
        score, mtype = m.best_score("starbucks", "Starbucks", None, None)
        assert score >= 0.99, score

        # bbox conversion
        e = COCOExporter(cfg)
        bbox = e.polygon_to_bbox(10, 20, 110, 20, 110, 50, 10, 50)
        assert bbox == [10, 20, 100, 30], bbox

        ok(f"Levenshtein matching works (score={score:.3f}, type={mtype})")
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
