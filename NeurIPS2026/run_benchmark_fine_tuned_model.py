#!/usr/bin/env python3
"""Table 2: fine-tuned detector AP for NavOCR and LLM training labels."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from navocr_bench.check import audit
from navocr_bench.detect_eval import evaluate_coco_results
from navocr_bench.detectors import run_paddledet_detector, run_ppocr_detector
from navocr_bench.manifest import build_manifest, read_manifest, write_manifest
from navocr_bench.ocr import package_versions, read_dump, write_dump
from navocr_bench.paths import prepared


ROOT = Path(__file__).resolve().parent
DEFAULT_PADDLEDET_ROOT = ROOT / "third_party" / "PaddleDetection"
IOU_THRESHOLDS = (0.25, 0.5, 0.75)
PPOCR_THRESHOLD = 0.25
PPOCR_BOX_THRESHOLD = 0.4
VARIATIONS = {
    "ppyoloe-navocr": {
        "method": "PP-YOLOE (NavOCR)",
        "backend": "paddledet",
        "artifact": ROOT / "weights" / "ppyoloe-navocr" / "model.pdparams",
    },
    "ppyoloe-llm": {
        "method": "PP-YOLOE (LLM)",
        "backend": "paddledet",
        "artifact": ROOT / "weights" / "ppyoloe-llm" / "model.pdparams",
    },
    "ppocrv5-navocr": {
        "method": "PP-OCRv5 mobile_det (NavOCR)",
        "backend": "ppocrv5",
        "artifact": ROOT / "weights" / "ppocrv5-navocr",
    },
    "ppocrv5-llm": {
        "method": "PP-OCRv5 mobile_det (LLM)",
        "backend": "ppocrv5",
        "artifact": ROOT / "weights" / "ppocrv5-llm",
    },
}


def _require_readable(entries, image_dir: Path) -> None:
    problems = audit(entries, image_dir)
    if not problems:
        return
    first = problems[0]
    raise SystemExit(
        f"{len(problems)} unreadable image(s); first: {first[0]} "
        f"{first[2]} ({first[1]})"
    )


def _write_settings(output: Path, settings: dict) -> None:
    path = output.with_suffix(".settings.json")
    prepared(path).write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def _cmd_manifest(args: argparse.Namespace) -> int:
    entries = build_manifest(args.images)
    write_manifest(entries, args.out)
    print(f"{len(entries)} images -> {args.out}")
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    entries = read_manifest(args.manifest)
    _require_readable(entries, args.images)
    spec = VARIATIONS[args.variation]
    artifact = Path(spec["artifact"])
    output = args.out_dir / f"pred.{args.variation}.jsonl"

    if spec["backend"] == "paddledet":
        config = args.config or (
            args.paddledet_root
            / "configs"
            / "ppyoloe"
            / "ppyoloe_crn_s_300e_coco.yml"
        )
        results = run_paddledet_detector(
            entries,
            args.images,
            args.paddledet_root,
            config,
            artifact,
            device=args.device,
        )
        settings = {
            "paddledet_root": str(args.paddledet_root.resolve()),
            "config": str(config.resolve()),
            "versions": package_versions("paddlepaddle-gpu", "paddlepaddle"),
        }
    else:
        max_side = args.max_side or None
        iterator = run_ppocr_detector(
            entries,
            args.images,
            artifact,
            device=args.device,
            max_side=max_side,
            threshold=PPOCR_THRESHOLD,
            box_threshold=PPOCR_BOX_THRESHOLD,
        )
        results = []
        started = time.perf_counter()
        for index, result in enumerate(iterator, start=1):
            results.append(result)
            if index % 100 == 0 or index == len(entries):
                elapsed = time.perf_counter() - started
                rate = index / elapsed if elapsed else 0.0
                print(f"  {index}/{len(entries)} {rate:.1f} img/s", flush=True)
        settings = {
            "model_dir": str(artifact.resolve()),
            "max_side": max_side,
            "threshold": PPOCR_THRESHOLD,
            "box_threshold": PPOCR_BOX_THRESHOLD,
            "versions": package_versions("paddleocr", "paddlex", "paddlepaddle-gpu"),
        }

    write_dump(results, output)
    _write_settings(
        output,
        {
            "variation": args.variation,
            "method": spec["method"],
            "backend": spec["backend"],
            "artifact": str(artifact.resolve()),
            "device": args.device,
            "images": len(entries),
            **settings,
        },
    )
    failures = [result for result in results if result.error]
    boxes = sum(len(result.items) for result in results)
    print(
        f"{args.variation}: {len(results)} images, {boxes} boxes, "
        f"{len(failures)} failed -> {output}"
    )
    return 0


def _validated_results(path: Path, manifest_ids: set[str]):
    results = read_dump(path)
    ids = [result.id for result in results]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{path} contains duplicate image ids")
    missing = manifest_ids - set(ids)
    unknown = set(ids) - manifest_ids
    if missing or unknown:
        raise SystemExit(
            f"{path} does not match the manifest: "
            f"{len(missing)} missing, {len(unknown)} unknown"
        )
    failures = [result for result in results if result.error]
    if failures:
        raise SystemExit(
            f"{path} contains {len(failures)} failed image(s); "
            f"first: {failures[0].id}: {failures[0].error}"
        )
    return results


def _cmd_score(args: argparse.Namespace) -> int:
    entries = read_manifest(args.manifest)
    manifest_ids = {entry.id for entry in entries}
    runs = []
    for variation in args.variations:
        path = args.out_dir / f"pred.{variation}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run detect for {variation}")
        results = _validated_results(path, manifest_ids)
        scores, n_predictions, n_gt_boxes = evaluate_coco_results(
            args.gt, args.manifest, results, IOU_THRESHOLDS
        )
        run = {
            "variation": variation,
            "method": VARIATIONS[variation]["method"],
            "prediction_file": str(path.resolve()),
            "n_images": len(entries),
            "n_gt_boxes": n_gt_boxes,
            "n_predictions": n_predictions,
            "ap": scores,
        }
        runs.append(run)
        print(
            f"{variation}: "
            + " ".join(f"{key}={value:.4f}" for key, value in scores.items())
        )

    report = {
        "ground_truth": str(args.gt.resolve()),
        "manifest": str(args.manifest.resolve()),
        "iou_thresholds": list(IOU_THRESHOLDS),
        "runs": runs,
    }
    output = args.out_dir / "report.table2.json"
    prepared(output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"report -> {output}")
    return 0


def _cmd_table(args: argparse.Namespace) -> int:
    report = json.loads(args.report.read_text(encoding="utf-8"))
    print("| Method | AP@0.25 | AP@0.50 | AP@0.75 |")
    print("| --- | ---: | ---: | ---: |")
    for run in report["runs"]:
        ap = run["ap"]
        print(
            f"| {run['method']} | {ap['AP@0.25']:.3f} | "
            f"{ap['AP@0.50']:.3f} | {ap['AP@0.75']:.3f} |"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_benchmark_fine_tuned_model.py", description=__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("manifest", help="fix the evaluation image list and ids")
    command.add_argument("--images", required=True, type=Path)
    command.add_argument("--out", required=True, type=Path)
    command.set_defaults(func=_cmd_manifest)

    command = sub.add_parser("detect", help="run one fine-tuned detector variation")
    command.add_argument("--variation", required=True, choices=list(VARIATIONS))
    command.add_argument("--manifest", required=True, type=Path)
    command.add_argument("--images", required=True, type=Path)
    command.add_argument("--out-dir", required=True, type=Path)
    command.add_argument("--device", default="gpu:0")
    command.add_argument("--paddledet-root", type=Path, default=DEFAULT_PADDLEDET_ROOT)
    command.add_argument("--config", type=Path)
    command.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="PP-OCRv5 long-side limit; use 0 to keep the original size",
    )
    command.set_defaults(func=_cmd_detect)

    command = sub.add_parser("score", help="score detector dumps at three IoUs")
    command.add_argument("--manifest", required=True, type=Path)
    command.add_argument("--gt", required=True, type=Path)
    command.add_argument("--out-dir", required=True, type=Path)
    command.add_argument(
        "--variations",
        nargs="+",
        choices=list(VARIATIONS),
        default=list(VARIATIONS),
    )
    command.set_defaults(func=_cmd_score)

    command = sub.add_parser("table", help="print the Markdown Table 2 rows")
    command.add_argument("--report", required=True, type=Path)
    command.set_defaults(func=_cmd_table)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
