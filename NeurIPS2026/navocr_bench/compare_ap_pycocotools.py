"""Recompute `run_benchmark_detection.py ap` with pycocotools on the same predictions.

detect_eval integrates the precision-recall curve exactly; pycocotools samples
101 recall points. Both use the same images, one class, no detection cap, no
area filter. Runs on CPU.

    python -m navocr_bench.compare_ap_pycocotools --manifest out/detection/manifest.json \
        --gt data/detection_benchmark/instances_test.json --ocr out/detection/ocr.jsonl \
        --pred out/detection/Qwen3.5-*.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from navocr_bench.ocr import read_dump
from navocr_bench.detect_eval import (
    IOU_THRESHOLDS,
    boxed_results,
    detections,
    evaluate,
    read_coco_gt,
    scored_ground_truth,
)


CATEGORY_ID = 1


def load_ground_truth(gt_path, keep_names):
    """Return a class-agnostic COCO object over the kept images, and its id map.

    Images without annotations are kept, as in detect_eval.
    """
    payload = json.loads(Path(gt_path).read_text(encoding='utf-8'))

    images = [img for img in payload['images'] if img['file_name'] in keep_names]
    kept_ids = {img['id'] for img in images}
    annotations = []
    for ann in payload['annotations']:
        if ann['image_id'] not in kept_ids:
            continue
        ann = copy.deepcopy(ann)
        ann['category_id'] = CATEGORY_ID
        ann['iscrowd'] = 0
        ann.setdefault('area', float(ann['bbox'][2]) * float(ann['bbox'][3]))
        annotations.append(ann)

    coco = COCO()
    coco.dataset = {
        'info': {},
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': [{'id': CATEGORY_ID, 'name': 'text'}],
    }
    coco.createIndex()

    return coco, {img['file_name']: img['id'] for img in images}


def to_coco_detections(preds, image_ids):
    """Our Detection list to pycocotools' result records, same boxes, same order."""
    return [
        {
            'image_id': image_ids[pred.image],
            'category_id': CATEGORY_ID,
            'bbox': [float(v) for v in pred.bbox],
            'score': float(pred.score),
        }
        for pred in preds
        if pred.image in image_ids
    ]


def pycocotools_ap(coco_gt, records, threshold, max_dets):
    """AP at one IoU threshold, class-agnostic, no area filter, no detection cap."""
    if not records:
        return 0.0

    coco_dt = coco_gt.loadRes(copy.deepcopy(records))
    evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())
    evaluator.params.catIds = [CATEGORY_ID]
    evaluator.params.iouThrs = np.array([threshold])
    evaluator.params.areaRng = [[0.0, 1e10]]
    evaluator.params.areaRngLbl = ['all']
    evaluator.params.maxDets = [max_dets]
    evaluator.evaluate()
    evaluator.accumulate()

    precision = evaluator.eval['precision']
    valid = precision[precision > -1]
    return float(valid.mean()) if valid.size else 0.0


def busiest_image(records):
    counts = {}
    for record in records:
        counts[record['image_id']] = counts.get(record['image_id'], 0) + 1
    return max(counts.values(), default=1)


def main():
    parser = argparse.ArgumentParser(
        description='Compare detect_eval average precision against pycocotools')
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--gt', required=True, type=Path)
    parser.add_argument('--pred', required=True, type=Path, nargs='+')
    parser.add_argument('--ocr', type=Path,
                        help='OCR dump the selections came from')
    parser.add_argument('--iou', type=float, nargs='+', default=list(IOU_THRESHOLDS))
    parser.add_argument('--out', type=Path,
                        help='Optional JSON report path')
    args = parser.parse_args()

    names, scored = scored_ground_truth(args.manifest, read_coco_gt(args.gt))
    coco_gt, image_ids = load_ground_truth(args.gt, set(scored))

    n_gt = sum(len(boxes) for boxes in scored.values())
    print(f'images={len(scored)}  ground-truth boxes={n_gt}')
    print()

    rows = []
    ocr_results = read_dump(args.ocr) if args.ocr else None
    for path in args.pred:
        results, kind = boxed_results(path, ocr_results)
        preds = detections(results, names)
        records = to_coco_detections(preds, image_ids)
        max_dets = max(busiest_image(records), 100)

        ours = evaluate(scored, preds, args.iou)
        row = {
            'prediction_file': str(path),
            'prediction_kind': kind,
            'n_predictions': len(preds),
            'max_dets': max_dets,
            'thresholds': {},
        }
        for threshold in args.iou:
            key = f'AP@{threshold:.2f}'
            mine = ours[key]
            theirs = pycocotools_ap(coco_gt, records, threshold, max_dets)
            row['thresholds'][key] = {
                'detect_eval': mine,
                'pycocotools': theirs,
                'delta': None if mine is None else mine - theirs,
            }
        rows.append(row)

    width = max(len(Path(row['prediction_file']).name) for row in rows)
    header = f'{"file":<{width}}  {"IoU":>5}  {"ours":>8}  {"pycoco":>8}  {"delta":>9}'
    print(header)
    print('-' * len(header))
    worst = 0.0
    for row in rows:
        name = Path(row['prediction_file']).name
        for key, values in row['thresholds'].items():
            mine, theirs, delta = (
                values['detect_eval'], values['pycocotools'], values['delta'])
            mine_text = 'n/a' if mine is None else f'{mine:.6f}'
            delta_text = 'n/a' if delta is None else f'{delta:+.6f}'
            if delta is not None:
                worst = max(worst, abs(delta))
            print(f'{name:<{width}}  {key[-4:]:>5}  {mine_text:>8}  '
                  f'{theirs:.6f}  {delta_text:>9}')
    print()
    print(f'largest absolute difference: {worst:.6f}')

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({'rows': rows, 'max_abs_delta': worst},
                       ensure_ascii=False, indent=2),
            encoding='utf-8')
        print(f'-> {args.out}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
