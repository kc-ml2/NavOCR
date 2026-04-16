#!/usr/bin/env python3
"""
Standalone NavOCR inference using the shared ros params YAML.

- Supports either a single image or a directory of images.
- Reuses load_detector_config / load_ocr_config.
- Supports both Paddle and OpenVINO backends through backend_factory.
- Prints per-image det/rec/e2e FPS summary.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2

from navocr import create_detector, create_ocr
from navocr.config_loader import load_detector_config, load_ocr_config
from navocr.pipeline_utils import clip_bbox, draw_detection, load_image


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
OCR_FAIL_RESULTS = {'no_text_detected', 'empty_crop', 'ocr_error'}


class NavOCRPipeline:
    def __init__(self, detector, ocr, conf_threshold: float, output_dir: str):
        self.detector = detector
        self.ocr = ocr
        self.conf_threshold = conf_threshold
        self.output_dir = output_dir

    def save_result(self, orig_bgr: np.ndarray, results: list[dict], output_path: str):
        vis = orig_bgr.copy()
        for result in results:
            x1, y1, x2, y2 = result['box']
            text = result['text'] if result['text'] not in OCR_FAIL_RESULTS else 'text'
            draw_detection(vis, x1, y1, x2, y2, text)
        cv2.imwrite(str(output_path), vis)
        print(f'[OUT] Saved -> {output_path}')

    def infer_image(self, image_path: str):
        frame_start = time.perf_counter()
        cv_image = load_image(image_path)

        detection_start = time.perf_counter()
        det_results = self.detector.infer_loaded_images([cv_image])
        if det_results is None:
            det_results = self.detector.infer([image_path])
        detection_time = time.perf_counter() - detection_start

        frame_ocr_time = 0.0
        parsed_results = []
        valid_detections = 0

        if det_results and isinstance(det_results, list) and 'bbox' in det_results[0]:
            for bbox in det_results[0]['bbox']:
                cls_id, score, x1, y1, x2, y2 = bbox
                if float(score) < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = clip_bbox(cv_image, x1, y1, x2, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                valid_detections += 1
                cropped_image = cv_image[y1:y2, x1:x2]

                ocr_start = time.perf_counter()
                ocr_text = self.ocr.recognize(cropped_image) if cropped_image.size > 0 else 'no_text_detected'
                frame_ocr_time += time.perf_counter() - ocr_start

                parsed_results.append({
                    'box': (x1, y1, x2, y2),
                    'score': float(score),
                    'text': ocr_text,
                    'class_id': int(cls_id),
                })

        total_time = time.perf_counter() - frame_start
        det_fps = (1.0 / detection_time) if detection_time > 0 else 0.0
        rec_fps = (1.0 / frame_ocr_time) if frame_ocr_time > 0 else None
        e2e_fps = (1.0 / total_time) if total_time > 0 else 0.0

        return {
            'image': image_path,
            'orig_bgr': cv_image,
            'results': parsed_results,
            'det_count': valid_detections,
            'detection_time': detection_time,
            'ocr_time': frame_ocr_time,
            'total_time': total_time,
            'det_fps': det_fps,
            'rec_fps': rec_fps,
            'e2e_fps': e2e_fps,
        }


def collect_images(infer_dir: str) -> list[Path]:
    infer_path = Path(infer_dir)
    if not infer_path.exists():
        raise FileNotFoundError(f'Directory not found: {infer_dir}')
    image_paths = sorted([path for path in infer_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS])
    if not image_paths:
        raise FileNotFoundError(f'No supported images found in {infer_dir}')
    return image_paths


def print_per_image_summary(info: dict):
    rec_fps_str = f"{info['rec_fps']:.2f}" if info['rec_fps'] is not None else 'N/A'
    print(
        f"[FPS] {Path(info['image']).name} | "
        f"det_fps={info['det_fps']:.2f} | "
        f"rec_fps={rec_fps_str} | "
        f"e2e_fps={info['e2e_fps']:.2f} | "
        f"det_count={info['det_count']}"
    )


def print_detection_lines(info: dict):
    print(f"[DET] {info['det_count']} region(s)")
    for result in info['results']:
        x1, y1, x2, y2 = result['box']
        print(f"  [{x1},{y1},{x2},{y2}] det={result['score']:.2f} -> '{result['text']}'")


def infer_directory(
    pipeline: NavOCRPipeline,
    infer_dir: str,
    output_dir: str,
    save_image: bool = False,
    log_interval: int = 1,
):
    image_paths = collect_images(infer_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_det_time = 0.0
    total_ocr_time = 0.0
    total_e2e_time = 0.0
    total_det_count = 0
    errors = []
    num_images = len(image_paths)
    width = len(str(num_images))

    print(f'[CFG] Images      : {num_images} in {infer_dir}')
    print(f'[CFG] Output dir  : {out_dir}')
    print(f'[CFG] Save image  : {save_image}')
    print()

    for index, image_path in enumerate(image_paths, start=1):
        try:
            info = pipeline.infer_image(str(image_path))
            total_det_time += info['detection_time']
            total_ocr_time += info['ocr_time']
            total_e2e_time += info['total_time']
            total_det_count += info['det_count']

            if (index % log_interval) == 0 or index == num_images:
                print(f'[{index:{width}}/{num_images}] {image_path.name}')
                print(f"[IMG] {info['orig_bgr'].shape[1]}x{info['orig_bgr'].shape[0]}")
                print_detection_lines(info)
                print_per_image_summary(info)

            if save_image:
                output_path = out_dir / image_path.name
                pipeline.save_result(info['orig_bgr'], info['results'], str(output_path))
        except Exception as exc:
            errors.append((image_path.name, str(exc)))
            print(f'[ERROR] {image_path.name}: {exc}')

    processed = num_images - len(errors)
    if processed > 0:
        print('\n' + '=' * 60)
        print('BENCHMARK SUMMARY')
        print('=' * 60)
        print(f'Total images processed : {processed}')
        print(f'Total errors           : {len(errors)}')
        print(f'Total detections       : {total_det_count}')
        print(f'Detection time total   : {total_det_time:.4f} s')
        print(f'OCR time total         : {total_ocr_time:.4f} s')
        print(f'E2E time total         : {total_e2e_time:.4f} s')
        print(f'Detection FPS avg      : {processed / total_det_time:.2f}' if total_det_time > 0 else 'Detection FPS avg      : N/A')
        print(f'OCR FPS avg            : {processed / total_ocr_time:.2f}' if total_ocr_time > 0 else 'OCR FPS avg            : N/A')
        print(f'E2E FPS avg            : {processed / total_e2e_time:.2f}' if total_e2e_time > 0 else 'E2E FPS avg            : N/A')
        print('=' * 60)

    if errors:
        print('\nFailed images:')
        for name, err in errors:
            print(f'  {name}: {err}')


def build_parser():
    parser = argparse.ArgumentParser(description='Standalone NavOCR inference with shared params YAML')
    parser.add_argument(
        '--params-file',
        type=str,
        required=True,
        help='Path to navocr_openvino.params.yaml or navocr_paddle.params.yaml',
    )
    parser.add_argument('--input', '-i', type=str, default=None, help='Path to input image')
    parser.add_argument('--infer_dir', type=str, default=None, help='Directory of input images')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output file path for single-image mode')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for folder mode')
    parser.add_argument('--save_image', action='store_true', help='Save annotated images')
    parser.add_argument('--log_interval', type=int, default=1, help='Print progress every N images in folder mode')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.input is None and args.infer_dir is None:
        parser.error('Either --input or --infer_dir must be provided.')
    if args.input is not None and args.infer_dir is not None:
        parser.error('Use only one of --input or --infer_dir.')

    detector_cfg = load_detector_config(args.params_file)
    ocr_cfg = load_ocr_config(args.params_file)
    detector = create_detector(detector_cfg)
    ocr = create_ocr(ocr_cfg)

    output_dir = args.output_dir or detector_cfg.output_dir
    if not output_dir:
        parser.error('Either --output_dir or output_dir in the params YAML must be provided.')
    os.makedirs(output_dir, exist_ok=True)

    print(f'[CFG] Params file : {args.params_file}')
    print(f'[CFG] Detector    : {detector_cfg.backend}')
    print(f'[CFG] OCR         : {ocr_cfg.backend}')
    print(f'[CFG] Output dir  : {output_dir}')

    pipeline = NavOCRPipeline(
        detector=detector,
        ocr=ocr,
        conf_threshold=detector_cfg.detection_threshold,
        output_dir=output_dir,
    )

    if args.input:
        output_path = args.output or str(Path(output_dir) / Path(args.input).name)
        info = pipeline.infer_image(args.input)
        bgr = info['orig_bgr']
        print(f"[IMG] {Path(args.input).name} | {bgr.shape[1]}x{bgr.shape[0]}")
        print_detection_lines(info)
        print_per_image_summary(info)

        if args.save_image:
            pipeline.save_result(info['orig_bgr'], info['results'], output_path)
        return

    infer_directory(
        pipeline=pipeline,
        infer_dir=args.infer_dir,
        output_dir=output_dir,
        save_image=args.save_image,
        log_interval=max(1, args.log_interval),
    )


if __name__ == '__main__':
    main()
