#!/usr/bin/env python3
"""Table 3: AP and FPS for OCR/LLM and fine-tuned detector variations."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from navocr_bench.check import audit
from navocr_bench.detect_eval import (
    IOU_THRESHOLDS,
    boxed_results,
    detections,
    evaluate,
    read_coco_gt,
    scored_ground_truth,
)
from navocr_bench.detectors import (
    PPOCR_BOX_THRESHOLD,
    PPOCR_MAX_SIDE,
    PPOCR_THRESHOLD,
    PPOCR_UNCLIP_RATIO,
    run_paddledet_detector,
    run_ppocr_detector,
)
from navocr_bench.llm import (
    MAX_TOKENS,
    build_transformers_client,
    read_filtered,
    run_filter,
    write_filtered,
)
from navocr_bench.manifest import build_manifest, read_manifest, write_manifest
from navocr_bench.ocr import (
    DET_MODEL,
    build_paddle_engine,
    ocr_settings,
    package_versions,
    read_dump,
    run_ocr,
    write_dump,
)
from navocr_bench.paths import prepared
from navocr_bench.prompts import PROMPTS


ROOT = Path(__file__).resolve().parent
DEFAULT_PADDLEDET_ROOT = ROOT / "third_party" / "PaddleDetection"
DEFAULT_PPYOLOE_WEIGHTS = ROOT / "weights" / "ppyoloe-navocr" / "model.pdparams"
DEFAULT_PPOCRV5_MODEL_DIR = ROOT / "weights" / "ppocrv5-navocr"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _cmd_manifest(args: argparse.Namespace) -> int:
    entries = build_manifest(args.images)
    write_manifest(entries, args.out)
    print(f"{len(entries)} images -> {args.out}")
    return 0


def _require_readable(entries, image_dir) -> None:
    """Fail before loading OCR models if any image is missing or empty."""
    problems = audit(entries, image_dir)
    if not problems:
        return
    print(f"{len(problems)} unreadable image(s):")
    for id_, filename, reason in problems[:10]:
        print(f"  {id_}  {reason}  ({filename})")
    if len(problems) > 10:
        print(f"  ... and {len(problems) - 10} more")
    raise SystemExit("fix the images, or drop them from the manifest, then rerun")


def progress_line(done: int, total: int, seconds: float) -> str:
    if seconds <= 0:
        return f"  {done}/{total}  -- img/s  eta --"
    rate = done / seconds
    eta = int((total - done) / rate)
    return f"  {done}/{total}  {rate:.1f} img/s  eta {eta}s"


def _with_progress(results, total: int, every: int):
    start = time.perf_counter()
    for i, result in enumerate(results, start=1):
        if every and (i % every == 0 or i == total):
            print(progress_line(i, total, time.perf_counter() - start), flush=True)
        yield result


def summarise_dump(results) -> str:
    """One-line tally of images, text regions and failures."""
    failed = [r for r in results if r.error]
    total = sum(len(r.items) for r in results)
    summary = f"{len(results)} images, {total} text regions, {len(failed)} failed"
    if failed:
        summary += f" (first {failed[0].id}: {failed[0].error})"
    return summary


def _cmd_ocr(args: argparse.Namespace) -> int:
    entries = read_manifest(args.manifest)
    _require_readable(entries, args.images)


    engine = build_paddle_engine(det_model=args.det_model)
    results = list(
        _with_progress(run_ocr(entries, args.images, engine), len(entries), every=100)
    )
    write_dump(results, args.out)
    _write_settings(args.out, {**ocr_settings(det_model=args.det_model), "images": len(entries)})
    print(f"{summarise_dump(results)} -> {args.out}")
    return 0


def _cmd_ppocrv5_detector(args: argparse.Namespace) -> int:
    entries = read_manifest(args.manifest)
    _require_readable(entries, args.images)
    max_side = args.max_side
    results = list(
        _with_progress(
            run_ppocr_detector(
                entries,
                args.images,
                args.model_dir,
                device=args.device,
                max_side=max_side,
                threshold=PPOCR_THRESHOLD,
                box_threshold=PPOCR_BOX_THRESHOLD,
                unclip_ratio=PPOCR_UNCLIP_RATIO,
            ),
            len(entries),
            every=100,
        )
    )
    write_dump(results, args.out)
    _write_settings(
        args.out,
        {
            "backend": "ppocrv5",
            "model_dir": str(args.model_dir.resolve()),
            "device": args.device,
            "max_side": max_side,
            "threshold": PPOCR_THRESHOLD,
            "box_threshold": PPOCR_BOX_THRESHOLD,
            "unclip_ratio": PPOCR_UNCLIP_RATIO,
            "images": len(entries),
            "versions": package_versions(
                "paddleocr", "paddlex", "paddlepaddle-gpu"
            ),
        },
    )
    print(f"{summarise_dump(results)} -> {args.out}")
    return 0


def _cmd_ppyoloe_detector(args: argparse.Namespace) -> int:
    entries = read_manifest(args.manifest)
    _require_readable(entries, args.images)
    config = args.config or (
        args.paddledet_root / "configs" / "ppyoloe" / "ppyoloe_crn_s_300e_coco.yml"
    )
    results = run_paddledet_detector(
        entries,
        args.images,
        args.paddledet_root,
        config,
        args.weights,
        device=args.device,
    )
    write_dump(results, args.out)
    _write_settings(
        args.out,
        {
            "backend": "ppyoloe",
            "paddledet_root": str(args.paddledet_root.resolve()),
            "config": str(config.resolve()),
            "weights": str(args.weights.resolve()),
            "device": args.device,
            "images": len(entries),
            "versions": package_versions("paddlepaddle-gpu", "paddlepaddle"),
        },
    )
    print(f"{summarise_dump(results)} -> {args.out}")
    return 0


def _cmd_filter(args: argparse.Namespace) -> int:

    ocr_results = read_dump(args.ocr)
    client = build_transformers_client(
        args.model, max_tokens=MAX_TOKENS, chat_template=args.chat_template
    )
    results = run_filter(ocr_results, args.prompt, client, batch_size=args.batch_size)
    write_filtered(results, args.out)
    _write_settings(
        args.out,
        {
            "model": args.model,
            "revision": client.revision,
            "prompt": args.prompt,
            "batch_size": args.batch_size,
            "max_new_tokens": MAX_TOKENS,
            "chat_template": args.chat_template,
            "greedy": True,
            "ocr_dump": str(args.ocr),
            "versions": package_versions("torch", "transformers", "accelerate"),
        },
    )
    print(f"{len(results)} images -> {args.out}")
    return 0


def _write_settings(out: Path, settings: dict) -> None:
    """Record the settings of a run in ``<out>.settings.json``."""
    path = Path(out).with_suffix(".settings.json")
    prepared(path).write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def _cmd_ap(args: argparse.Namespace) -> int:
    """AP of each prediction file against the COCO ground truth."""
    names, scored = scored_ground_truth(args.manifest, read_coco_gt(args.gt))
    ocr_results = read_dump(args.ocr) if args.ocr else None
    runs = []
    for path in args.pred:
        results, kind = boxed_results(path, ocr_results)
        preds = detections(results, names)
        runs.append(
            {
                "prediction_file": str(path),
                "prediction_kind": kind,
                "ground_truth": str(args.gt),
                "n_images": len(scored),
                "n_gt_boxes": sum(len(b) for b in scored.values()),
                "n_predictions": len(preds),
                "iou_thresholds": list(args.iou),
                "ap": evaluate(scored, preds, args.iou),
            }
        )

    first = f"AP@{args.iou[0]:.2f}"
    runs.sort(key=lambda r: (r["ap"][first] is not None, r["ap"][first] or 0.0),
              reverse=True)
    prepared(args.out).write_text(
        json.dumps({"runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    width = max(len(Path(r["prediction_file"]).name) for r in runs)
    for run in runs:
        scores = "  ".join(
            f"{k}={'n/a' if v is None else f'{v:.4f}'}" for k, v in run["ap"].items()
        )
        print(
            f"{Path(run['prediction_file']).name:<{width}}  "
            f"pred={run['n_predictions']:<6} {scores}"
        )
    print(f"n={runs[0]['n_images']} gt={runs[0]['n_gt_boxes']} -> {args.out}")
    return 0


def _cmd_fps(args: argparse.Namespace) -> int:
    """FPS = 1 / (mean OCR time + mean LLM time), over every manifest image."""
    entries = read_manifest(args.manifest)
    ocr_results = read_dump(args.ocr)
    failures = [r for r in ocr_results if r.error]
    if failures:
        raise SystemExit(
            f"{len(failures)} image(s) failed during OCR "
            f"(first {failures[0].id}: {failures[0].error}); rerun the dump"
        )
    ocr_times = {r.id: r.elapsed_s for r in ocr_results}
    llm_times = {r.id: r.elapsed_s for r in read_filtered(args.filtered)}
    for times, what in ((ocr_times, "OCR result"), (llm_times, "selection")):
        missing = [e.id for e in entries if e.id not in times]
        if missing:
            raise SystemExit(f"no {what} for {len(missing)} image(s), first: {missing[0]}")

    n = len(entries)
    mean_ocr = sum(ocr_times[e.id] for e in entries) / n
    mean_llm = sum(llm_times[e.id] for e in entries) / n
    mean_total = mean_ocr + mean_llm
    payload = {
        "n": n,
        "mean_ocr_s": mean_ocr,
        "mean_llm_s": mean_llm,
        "mean_total_s": mean_total,
        "fps": 1.0 / mean_total if mean_total else 0.0,
    }
    prepared(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"n={n} ocr={mean_ocr * 1000:.1f}ms llm={mean_llm * 1000:.1f}ms "
        f"({payload['fps']:.2f} FPS) -> {args.out}"
    )
    return 0


def _cmd_detector_fps(args: argparse.Namespace) -> int:
    """FPS from end-to-end elapsed time stored in a detector dump."""
    entries = read_manifest(args.manifest)
    if not entries:
        raise SystemExit("manifest is empty")
    results = read_dump(args.pred)
    by_id = {result.id: result for result in results}
    missing = [entry.id for entry in entries if entry.id not in by_id]
    if missing:
        raise SystemExit(
            f"no detector result for {len(missing)} image(s), first: {missing[0]}"
        )
    failures = [by_id[entry.id] for entry in entries if by_id[entry.id].error]
    if failures:
        raise SystemExit(
            f"{len(failures)} image(s) failed during detection "
            f"(first {failures[0].id}: {failures[0].error})"
        )

    mean_s = sum(by_id[entry.id].elapsed_s for entry in entries) / len(entries)
    payload = {
        "n": len(entries),
        "prediction_file": str(args.pred),
        "mean_total_s": mean_s,
        "fps": 1.0 / mean_s if mean_s else 0.0,
    }
    prepared(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"n={len(entries)} detector={mean_s * 1000:.1f}ms "
        f"({payload['fps']:.2f} FPS) -> {args.out}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_benchmark_model_and_llm.py", description=__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("manifest", help="list the images of a split and assign ids")
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(func=_cmd_manifest)

    p = sub.add_parser("ocr", help="dump all text regions (PaddleOCR)")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--det-model", default=DET_MODEL,
                   choices=["PP-OCRv5_server_det", "PP-OCRv5_mobile_det"])
    p.set_defaults(func=_cmd_ocr)

    p = sub.add_parser(
        "ppocrv5-detector", help="run a fine-tuned PP-OCRv5 detector"
    )
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_PPOCRV5_MODEL_DIR)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="gpu:0")
    p.add_argument(
        "--max-side",
        type=_positive_int,
        default=PPOCR_MAX_SIDE,
        help="cap the long image side",
    )
    p.set_defaults(func=_cmd_ppocrv5_detector)

    p = sub.add_parser(
        "ppyoloe-detector", help="run a fine-tuned PP-YOLOE checkpoint"
    )
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--paddledet-root", type=Path, default=DEFAULT_PADDLEDET_ROOT)
    p.add_argument(
        "--config",
        type=Path,
        help="default: <paddledet-root>/configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml",
    )
    p.add_argument("--weights", type=Path, default=DEFAULT_PPYOLOE_WEIGHTS)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="gpu:0")
    p.set_defaults(func=_cmd_ppyoloe_detector)

    p = sub.add_parser("filter", help="select navigation-relevant text with an LLM")
    p.add_argument("--ocr", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="p1", choices=sorted(PROMPTS))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=True,
                   help="wrap each prompt in the model's chat template, thinking off")
    p.set_defaults(func=_cmd_filter)

    p = sub.add_parser("ap", help="average precision against the annotated COCO split")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--gt", required=True, type=Path,
                   help="COCO instances file holding the manual boxes")
    p.add_argument("--pred", required=True, type=Path, nargs="+",
                   help="OCR dumps or LLM selections (the latter need --ocr)")
    p.add_argument("--ocr", type=Path,
                   help="OCR dump the selections came from")
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(func=_cmd_ap, iou=list(IOU_THRESHOLDS))

    p = sub.add_parser("fps", help="end-to-end FPS of OCR plus LLM filtering")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--ocr", required=True, type=Path)
    p.add_argument("--filtered", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(func=_cmd_fps)

    p = sub.add_parser("detector-fps", help="end-to-end FPS of a detector dump")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--pred", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(func=_cmd_detector_fps)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
