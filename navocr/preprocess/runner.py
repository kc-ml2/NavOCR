"""CLI orchestrator for the NavOCR preprocessing pipeline.

Usage:
    preprocess_navocr --run-id <run_id> [options]
    python -m navocr.preprocess.runner --run-id <run_id> [options]

Arguments:
    --run-id           str, REQUIRED
                       Identifies this run. Determines paths:
                         <raw-root>/<run_id>/images/
                         <work-root>/<run_id>/{manifest,detections}.csv
                         <output-root>/<run_id>/{images,annotations}/

    --raw-root         str, default "data/raw"
                       Parent directory containing the crawled image folder.

    --work-root        str, default "data"
                       Parent directory for the run's manifest, detections,
                       and the shared translation cache.

    --output-root      str, default "data/preprocessed"
                       Parent directory for the final COCO dataset output.

    --clip-threshold   float, default 0.7  (override: CLIP_THRESHOLD env)
                       Minimum CLIP score (min across 3-pass cascade) for an
                       image to advance to OCR. Lower → more permissive.

    --sim-threshold    float, default 0.5  (override: SIMILARITY_THRESHOLD env)
                       Minimum normalized Levenshtein similarity between an
                       OCR text and the store name (or its DeepL translation)
                       to label that bbox as a prominent sign.

    --device           str, default "cpu"
                       Device for CLIP inference ("cpu" or "cuda").

    --stages           nargs+, default [clip ocr match export]
                       Subset of stages to run, in order. Useful for partial
                       reruns (e.g. --stages match export).

    --resume           flag
                       Informational only — every stage already skips rows
                       not in its expected input status, so reruns are
                       inherently idempotent.

Environment variables (also loaded from ./.env if present):
    DEEPL_AUTH_KEY        Enables DeepL fallback translation in matcher.
    CLIP_THRESHOLD        Default for --clip-threshold.
    SIMILARITY_THRESHOLD  Default for --sim-threshold.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

from .clip_filter import CLIPFilter
from .coco_exporter import COCOExporter
from .manifest_io import DetectionIO, ManifestIO, ManifestRow, PipelineConfig
from .matcher import Matcher, TranslationCache
from .ocr_runner import OCRRunner

_ALL_STAGES = ["clip", "ocr", "match", "export"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _status_summary(rows: list[ManifestRow]) -> str:
    counts = Counter(r.status for r in rows)
    return "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def run_pipeline(config: PipelineConfig, stages: list[str], resume: bool) -> None:
    image_dir = Path(config.raw_root) / config.run_id / "images"
    work_dir = Path(config.work_root) / config.run_id
    output_root = Path(config.output_root) / config.run_id
    cache_path = Path(config.work_root) / "cache" / "translations.json"

    work_dir.mkdir(parents=True, exist_ok=True)

    manifest_io = ManifestIO(work_dir / "manifest.csv")
    detection_io = DetectionIO(work_dir / "detections.csv")
    cache = TranslationCache(cache_path)

    rows = manifest_io.read_all()
    logger.info("Loaded %d manifest rows", len(rows))

    if resume:
        logger.info("Resume mode: each stage skips rows already past its expected input status")

    if "clip" in stages:
        logger.info("Stage clip starting — %d total rows", len(rows))
        rows = CLIPFilter(config).run(rows, image_dir, manifest_io)
        rows = manifest_io.read_all()
        logger.info("Stage clip done — %s", _status_summary(rows))

    if "ocr" in stages:
        logger.info("Stage ocr starting — %d total rows", len(rows))
        OCRRunner(config).run(rows, image_dir, manifest_io, detection_io)
        rows = manifest_io.read_all()
        logger.info("Stage ocr done — %s", _status_summary(rows))

    if "match" in stages:
        logger.info("Stage match starting — %d total rows", len(rows))
        Matcher(config, cache).run(rows, manifest_io, detection_io)
        rows = manifest_io.read_all()
        logger.info("Stage match done — %s", _status_summary(rows))

    if "export" in stages:
        logger.info("Stage export starting — %d total rows", len(rows))
        detection_rows = detection_io.read_all()
        COCOExporter(config).run(rows, detection_rows, image_dir, output_root, manifest_io)
        rows = manifest_io.read_all()
        logger.info("Stage export done — %s", _status_summary(rows))

    logger.info("Pipeline complete — final status summary: %s", _status_summary(rows))


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(
        prog="preprocess_navocr",
        description="NavOCR preprocessing pipeline",
    )
    parser.add_argument("--run-id", required=True, type=str)
    parser.add_argument("--raw-root", default="data/raw", type=str)
    parser.add_argument("--work-root", default="data", type=str)
    parser.add_argument("--output-root", default="data/preprocessed", type=str)
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=float(os.environ.get("CLIP_THRESHOLD", "0.7")),
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=float(os.environ.get("SIMILARITY_THRESHOLD", "0.5")),
    )
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=_ALL_STAGES,
        default=list(_ALL_STAGES),
    )
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    config = PipelineConfig(
        run_id=args.run_id,
        raw_root=args.raw_root,
        work_root=args.work_root,
        output_root=args.output_root,
        clip_threshold=args.clip_threshold,
        similarity_threshold=args.sim_threshold,
        deepl_auth_key=os.environ.get("DEEPL_AUTH_KEY"),
        device=args.device,
    )

    ordered_stages = [s for s in _ALL_STAGES if s in args.stages]

    try:
        run_pipeline(config, ordered_stages, args.resume)
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
