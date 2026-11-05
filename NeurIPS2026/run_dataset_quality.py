#!/usr/bin/env python3
"""Table 1: text-list F1 of the LLM generator on hand-labelled images.

The LLM reads the OCR strings of each image and returns the place names. It is
not given the place name. OCR settings and the scorer (normalisation, greedy
1:1 matching at similarity 0.7, micro average) follow NavOCR's
evaluate_navocr_f1.py. `score` also writes the Unfiltered OCR row.

OCR and LLM run in separate environments and share only ocr_cache.jsonl:

    python run_dataset_quality.py ocr --image-dir IMAGES --gt-file GT.txt --out-dir out
    python run_dataset_quality.py filter --out-dir out --model MODEL --prompts p1,p3
    python run_dataset_quality.py score --gt-file GT.txt --out-dir out
    python run_dataset_quality.py table --out-dir out
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import textdistance
from PIL import Image, ImageOps

from navocr_bench.llm import build_transformers_client
from navocr_bench.ocr import DET_MODEL, MAX_SIDE, REC_MODEL, package_versions
from navocr_bench.parse_output import parse_selection
from navocr_bench.prompts import PROMPTS, render


GT_THRESHOLD = 0.7
UNFILTERED_LABEL = "unfiltered"
OCR_DEVICE = "gpu:0"
REC_BATCH_SIZE = 1
# Room for several Korean names copied verbatim.
MAX_NEW_TOKENS = 128


def normalize_text(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[\s_]+", "", text)
    text = re.sub(r"[^\w가-힣]", "", text)
    return text


def normalized_levenshtein_similarity(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    return float(textdistance.levenshtein.normalized_similarity(a_norm, b_norm))


def greedy_match(
    predictions: list[str], targets: list[str], threshold: float
) -> list[dict[str, Any]]:
    candidates: list[tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(predictions):
        for target_idx, target in enumerate(targets):
            score = normalized_levenshtein_similarity(pred, target)
            if score >= threshold:
                candidates.append((score, pred_idx, target_idx))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    matches: list[dict[str, Any]] = []
    for score, pred_idx, target_idx in candidates:
        if pred_idx in used_predictions or target_idx in used_targets:
            continue
        used_predictions.add(pred_idx)
        used_targets.add(target_idx)
        matches.append(
            {
                "prediction": predictions[pred_idx],
                "target": targets[target_idx],
                "similarity": round(score, 6),
            }
        )
    return matches


def dedupe_texts(texts: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for text in texts:
        key = normalize_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def f1_from_counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


@dataclass(frozen=True)
class Row:
    image: str
    gt_texts: list[str]


def split_brace_fields(line: str) -> tuple[str, list[str]]:
    if "," not in line:
        raise ValueError(f"missing filename separator: {line}")
    image_name, rest = line.split(",", 1)
    fields: list[str] = []
    pos = 0
    while True:
        start = rest.find("{", pos)
        if start < 0:
            break
        end = rest.find("}", start + 1)
        if end < 0:
            raise ValueError(f"missing closing brace: {line}")
        fields.append(rest[start + 1 : end])
        pos = end + 1
    return image_name.strip(), fields


def split_text_set(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def read_gt(path: Path) -> list[Row]:
    """Image name and the labels in the second field; later fields are ignored."""
    rows: list[Row] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        image_name, fields = split_brace_fields(line)
        if not fields:
            raise ValueError(f"{path}:{line_no}: expected a brace field after image name")
        rows.append(Row(image=image_name, gt_texts=split_text_set(fields[0])))
    return rows


def init_ocr() -> Any:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PaddleOCR  # installed only in navocr-ocr

    return PaddleOCR(
        device=OCR_DEVICE,
        text_detection_model_name=DET_MODEL,
        text_recognition_model_name=REC_MODEL,
        text_recognition_batch_size=REC_BATCH_SIZE,
        text_det_limit_side_len=MAX_SIDE,
        text_det_limit_type="max",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def load_for_ocr(image_path: Path) -> Any:
    """The image path, or a downscaled BGR array if the long side exceeds MAX_SIDE."""
    with Image.open(image_path) as image:
        if max(image.size) <= MAX_SIDE:
            return str(image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        ratio = MAX_SIDE / max(image.size)
        size = (round(image.width * ratio), round(image.height * ratio))
        resized = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(resized)[:, :, ::-1].copy()


def run_ocr(ocr: Any, image_path: Path) -> list[str]:
    texts: list[str] = []
    for page in ocr.predict(load_for_ocr(image_path)):
        texts.extend(page["rec_texts"])
    return dedupe_texts(texts)


def load_jsonl_by_image(path: Path, value_key: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    out: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        out[item["image"]] = item.get(value_key)
    return out


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_ocr(args: argparse.Namespace) -> int:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / "ocr_cache.jsonl"

    rows = read_gt(args.gt_file.resolve())
    image_dir = args.image_dir.resolve()
    missing = [r.image for r in rows if not (image_dir / r.image).exists()]
    if missing:
        raise FileNotFoundError(f"missing image files: {missing[:5]}")

    if args.force:
        cache_file.unlink(missing_ok=True)
    cache = load_jsonl_by_image(cache_file, "ocr_texts")

    todo = [r for r in rows if r.image not in cache]
    print(f"{len(rows)} images, {len(cache)} cached, {len(todo)} to run", flush=True)
    if not todo:
        return 0

    ocr = init_ocr()
    for idx, row in enumerate(todo, 1):
        texts = run_ocr(ocr, image_dir / row.image)
        append_jsonl(cache_file, {"image": row.image, "ocr_texts": texts})
        print(f"[{idx}/{len(todo)}] {row.image}: ocr={len(texts)}", flush=True)

    (out_dir / "ocr_settings.json").write_text(
        json.dumps(
            {
                "detection_model": DET_MODEL,
                "recognition_model": REC_MODEL,
                "recognition_batch_size": REC_BATCH_SIZE,
                "max_side": MAX_SIDE,
                "det_limit_type": "max",
                "image_count": len(rows),
                "versions": package_versions("paddleocr", "paddlex", "paddlepaddle-gpu", "paddlepaddle"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {cache_file}")
    return 0


def safe_label(label: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", label)


def selection_path(out_dir: Path, label: str, prompt_key: str) -> Path:
    return out_dir / f"sel.{safe_label(label)}.{prompt_key}.jsonl"


def cmd_filter(args: argparse.Namespace) -> int:
    out_dir = args.out_dir.resolve()
    cache_file = out_dir / "ocr_cache.jsonl"
    if not cache_file.exists():
        raise FileNotFoundError(f"{cache_file} not found; run the ocr stage first")

    cache = load_jsonl_by_image(cache_file, "ocr_texts")
    images = list(cache)

    prompt_keys = [k.strip() for k in args.prompts.split(",") if k.strip()]
    unknown = [k for k in prompt_keys if k not in PROMPTS]
    if unknown:
        raise KeyError(f"unknown prompts {unknown}; have {sorted(PROMPTS)}")

    if args.label is None:
        args.label = args.model.rstrip("/").split("/")[-1]
    client = build_transformers_client(args.model, max_tokens=MAX_NEW_TOKENS)

    for prompt_key in prompt_keys:
        path = selection_path(out_dir, args.label, prompt_key)
        if args.force:
            path.unlink(missing_ok=True)
        done = load_jsonl_by_image(path, "selected")
        # Images without OCR text skip the model; their labels still count as missed.
        todo = [i for i in images if i not in done and cache[i]]
        empty = [i for i in images if i not in done and not cache[i]]
        for image in empty:
            append_jsonl(
                path,
                {"image": image, "selected": [], "raw": "", "elapsed_s": 0.0, "skipped": "no-ocr"},
            )

        print(
            f"{prompt_key}: {len(todo)} to run, {len(done)} done, "
            f"{len(empty)} empty-ocr",
            flush=True,
        )
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            prompts = [render(prompt_key, cache[i]) for i in batch]
            t0 = time.perf_counter()
            replies = list(client(prompts))
            if len(replies) != len(batch):
                raise RuntimeError(
                    f"client returned {len(replies)} replies for {len(batch)} prompts"
                )
            per_image = (time.perf_counter() - t0) / len(batch)
            for image, reply in zip(batch, replies):
                append_jsonl(
                    path,
                    {
                        "image": image,
                        "selected": parse_selection(reply),
                        "raw": reply,
                        "elapsed_s": round(per_image, 6),
                    },
                )
            print(
                f"[{min(start + args.batch_size, len(todo))}/{len(todo)}] "
                f"{prompt_key} {per_image * 1000:.0f} ms/img",
                flush=True,
            )
        print(f"Wrote {path}")

    (out_dir / f"filter_settings.{safe_label(args.label)}.json").write_text(
        json.dumps(
            {
                "label": args.label,
                "model": args.model,
                "prompts": prompt_keys,
                "batch_size": args.batch_size,
                "max_new_tokens": MAX_NEW_TOKENS,
                "chat_template": True,
                "enable_thinking": False,
                "greedy": True,
                "versions": package_versions("torch", "transformers", "accelerate"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def score_selection(
    rows: list[Row], selected_by_image: dict[str, list[str]], gt_threshold: float
) -> dict[str, Any]:
    """Micro and macro P/R/F1 over all rows, plus per-image match details."""
    total_tp = total_fp = total_fn = 0
    f1_sum = precision_sum = recall_sum = 0.0
    details: list[dict[str, Any]] = []

    for row in rows:
        predictions = dedupe_texts(selected_by_image.get(row.image) or [])
        matches = greedy_match(predictions, row.gt_texts, gt_threshold)
        tp = len(matches)
        fp = len(predictions) - tp
        fn = len(row.gt_texts) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if tp + fp + fn == 0:
            # Nothing labelled, nothing selected: 1.0 in the macro average.
            f1_sum += 1.0
            precision_sum += 1.0
            recall_sum += 1.0
        else:
            per_image = f1_from_counts(tp, fp, fn)
            f1_sum += per_image["f1"]
            precision_sum += per_image["precision"]
            recall_sum += per_image["recall"]

        matched_predictions = {m["prediction"] for m in matches}
        matched_targets = {m["target"] for m in matches}
        details.append(
            {
                "image": row.image,
                "gt_texts": row.gt_texts,
                "predictions": predictions,
                "gt_matches": matches,
                "unmatched_predictions": [
                    p for p in predictions if p not in matched_predictions
                ],
                "unmatched_gt_texts": [
                    t for t in row.gt_texts if t not in matched_targets
                ],
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    n = len(rows) or 1
    return {
        "micro": f1_from_counts(total_tp, total_fp, total_fn),
        "macro": {
            "precision": round(precision_sum / n, 6),
            "recall": round(recall_sum / n, 6),
            "f1": round(f1_sum / n, 6),
        },
        "image_count": len(rows),
        "details": details,
    }


def cmd_score(args: argparse.Namespace) -> int:
    out_dir = args.out_dir.resolve()
    rows = read_gt(args.gt_file.resolve())
    cache_file = out_dir / "ocr_cache.jsonl"
    if not cache_file.exists():
        raise FileNotFoundError(f"{cache_file} not found; run the ocr stage first")

    # Unfiltered OCR first, then one row per sel.<label>.<prompt>.jsonl.
    runs = [(UNFILTERED_LABEL, "ocr", cache_file, load_jsonl_by_image(cache_file, "ocr_texts"))]
    for path in sorted(out_dir.glob("sel.*.jsonl")):
        stem = path.name[len("sel.") : -len(".jsonl")]
        label, _, prompt = stem.rpartition(".")
        runs.append((label, prompt, path, load_jsonl_by_image(path, "selected")))

    for label, prompt, path, selected in runs:
        covered = sum(1 for r in rows if r.image in selected)
        if covered < len(rows):
            print(
                f"WARNING {path.name}: {len(rows) - covered} of {len(rows)} images "
                "have no selection row; they score as empty predictions",
                flush=True,
            )

        score = score_selection(rows, selected, GT_THRESHOLD)

        report = {
            "label": label,
            "prompt": prompt,
            "selection_file": str(path),
            "gt_file": str(args.gt_file.resolve()),
            "gt_threshold": GT_THRESHOLD,
            "image_count": score["image_count"],
            "images_with_selection": covered,
            "micro": score["micro"],
            "macro": score["macro"],
        }
        (out_dir / f"report.{label}.{prompt}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (out_dir / f"matches.{label}.{prompt}.jsonl").open(
            "w", encoding="utf-8"
        ) as fh:
            for item in score["details"]:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

        m = score["micro"]
        print(
            f"{label} {prompt}: micro F1={m['f1']:.4f} P={m['precision']:.4f} "
            f"R={m['recall']:.4f} (tp={m['tp']} fp={m['fp']} fn={m['fn']})",
            flush=True,
        )
    return 0


def cmd_table(args: argparse.Namespace) -> int:
    out_dir = args.out_dir.resolve()
    reports = sorted(out_dir.glob("report.*.json"))
    if not reports:
        raise FileNotFoundError(f"no report.*.json under {out_dir}")

    print("| Method                             | F1-score | Precision | Recall |")
    print("|------------------------------------|----------|-----------|--------|")
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["label"] == UNFILTERED_LABEL:
            name = "Unfiltered OCR"
        else:
            name = f"LLM generator ({data['label']}, {data['prompt']})"
        m = data["micro"]
        print(
            f"| {name:<34} |  {m['f1']:.3f}   |   {m['precision']:.3f}   "
            f"| {m['recall']:.3f}  |"
        )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_ocr = sub.add_parser("ocr", help="run PP-OCR over the images into ocr_cache.jsonl")
    p_ocr.add_argument("--image-dir", type=Path, required=True)
    p_ocr.add_argument("--gt-file", type=Path, required=True)
    p_ocr.add_argument("--out-dir", type=Path, required=True)
    p_ocr.add_argument("--force", action="store_true", help="discard the OCR cache")
    p_ocr.set_defaults(func=cmd_ocr)

    p_filter = sub.add_parser("filter", help="ask an LLM which OCR strings name a place")
    p_filter.add_argument("--out-dir", type=Path, required=True)
    p_filter.add_argument("--model", required=True, help="model id or local path")
    p_filter.add_argument(
        "--label", help="row name in the table; defaults to the model name, e.g. Qwen3.5-9B"
    )
    p_filter.add_argument("--prompts", default="p1,p2,p3,p4,p5,p6")
    p_filter.add_argument("--batch-size", type=int, default=8)
    p_filter.add_argument("--force", action="store_true", help="discard existing selections")
    p_filter.set_defaults(func=cmd_filter)

    p_score = sub.add_parser("score", help="score selections against the manual labels")
    p_score.add_argument("--out-dir", type=Path, required=True)
    p_score.add_argument("--gt-file", type=Path, required=True)
    p_score.set_defaults(func=cmd_score)

    p_table = sub.add_parser("table", help="print the Table 1 rows")
    p_table.add_argument("--out-dir", type=Path, required=True)
    p_table.set_defaults(func=cmd_table)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
