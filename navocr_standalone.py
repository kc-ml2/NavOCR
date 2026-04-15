#!/usr/bin/env python3
"""
Standalone NavOCR inference using PaddleDetection + PaddleOCR.

- Keeps detector.py import structure when available.
- Supports single-image inference and folder-level inference.
- Prints per-image detection FPS, OCR recognition FPS, and end-to-end FPS.
- Saves annotated images optionally, similar to openvino_inf.py workflow.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from navocr.detector_paddle import PaddleDetector

try:
    from paddleocr import PaddleOCR
except ImportError as e:
    raise ImportError(
        "PaddleOCR is required. Install with: pip install paddleocr"
    ) from e


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


class Tee:
    """Duplicates writes to sys.stdout and a log file simultaneously."""

    def __init__(self, log_path: str):
        self._stdout = sys.stdout
        self._file = open(log_path, 'w', buffering=1, encoding='utf-8')
        sys.stdout = self

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


class DetectorFlags:
    def __init__(self, config, weights, draw_threshold, output_dir):
        self.config = config
        self.weights = weights
        self.draw_threshold = draw_threshold
        self.output_dir = output_dir


class PaddleOCRPipeline:
    OCR_NO_TEXT = "no_text_detected"
    OCR_ERROR = "ocr_error"
    OCR_FAIL_RESULTS = {OCR_NO_TEXT, "empty_crop", OCR_ERROR}

    def __init__(
        self,
        config_path: str,
        weights_path: str,
        conf_threshold: float,
        output_dir: str,
        ocr_language: str = 'en',
        use_angle_cls: bool = True,
        ocr_conf_threshold: float = 0.6,
        ocr_max_resize: int = 800,
    ):
        self.conf_threshold = conf_threshold
        self.output_dir = output_dir
        self.ocr_conf_threshold = ocr_conf_threshold
        self.ocr_max_resize = ocr_max_resize

        flags = DetectorFlags(
            config=config_path,
            weights=weights_path,
            draw_threshold=conf_threshold,
            output_dir=output_dir,
        )
        self.detector = PaddleDetector(flags)

        try:
            self.ocr = PaddleOCR(
                lang=ocr_language,
                use_angle_cls=use_angle_cls,
                det_db_thresh=0.25,
                det_db_box_thresh=0.4,
                rec_batch_num=32,
                enable_mkldnn=False,
            )
        except Exception:
            self.ocr = PaddleOCR(lang=ocr_language)

        self.bbox_color = (0, 255, 0)
        self.bbox_thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.2
        self.font_thickness = 2
        self.text_color = (0, 255, 0)
        self.text_bg_color = (0, 0, 0)

    def load_image(self, image_path: str) -> np.ndarray:
        bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return bgr

    def perform_ocr_immediate(self, image_crop: np.ndarray) -> str:
        if image_crop.size == 0:
            return self.OCR_NO_TEXT

        try:
            h, w = image_crop.shape[:2]
            if max(h, w) > self.ocr_max_resize:
                scale = self.ocr_max_resize / max(h, w)
                image_crop = cv2.resize(image_crop, (int(w * scale), int(h * scale)))

            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
            image_crop = cv2.equalizeHist(image_crop)
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_GRAY2BGR)

            results = self.ocr.ocr(image_crop)
            recognized_text = None

            if results and len(results) > 0:
                result = results[0]

                if isinstance(result, dict):
                    if 'rec_texts' in result and 'rec_scores' in result:
                        texts = []
                        for text, conf in zip(result['rec_texts'], result['rec_scores']):
                            if conf > self.ocr_conf_threshold:
                                text = text.strip()
                                if text:
                                    texts.append(text)
                        if texts:
                            recognized_text = ' '.join(' '.join(texts).split())

                elif isinstance(result, list) and len(result) > 0:
                    texts = []
                    for line in result:
                        if len(line) > 1:
                            text = line[1][0].strip()
                            conf = line[1][1]
                            if conf > self.ocr_conf_threshold and text:
                                texts.append(text)
                    if texts:
                        recognized_text = ' '.join(' '.join(texts).split())

            return recognized_text if recognized_text else self.OCR_NO_TEXT

        except Exception as e:
            print(f"[OCR-ERROR] {e}")
            return self.OCR_ERROR

    def draw_detection(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, label_text: str):
        cv2.rectangle(image, (x1, y1), (x2, y2), self.bbox_color, self.bbox_thickness)
        (tw, th), bl = cv2.getTextSize(label_text, self.font, self.font_scale, self.font_thickness)
        ty = max(th + bl + 2, y1 - 4)
        cv2.rectangle(
            image,
            (x1, ty - th - bl),
            (x1 + tw, ty + bl),
            self.text_bg_color,
            cv2.FILLED,
        )
        cv2.putText(
            image,
            label_text,
            (x1, ty),
            self.font,
            self.font_scale,
            self.text_color,
            self.font_thickness,
            cv2.LINE_AA,
        )

    def save_result(self, orig_bgr: np.ndarray, results: list, output_path: str):
        vis = orig_bgr.copy()
        for r in results:
            x1, y1, x2, y2 = r['box']
            text = r['text'] if r['text'] not in self.OCR_FAIL_RESULTS else 'text'
            self.draw_detection(vis, x1, y1, x2, y2, text)
        cv2.imwrite(str(output_path), vis)
        print(f"[OUT] Saved -> {output_path}")

    def infer_image(self, image_path: str):
        frame_start = time.perf_counter()
        cv_image = self.load_image(image_path)

        detection_start = time.perf_counter()
        det_results = self.detector.infer([image_path], visualize=False, save_results=False)
        detection_time = time.perf_counter() - detection_start

        frame_ocr_time = 0.0
        parsed_results = []
        valid_detections = 0

        if det_results and isinstance(det_results, list) and 'bbox' in det_results[0]:
            for bbox in det_results[0]['bbox']:
                cls_id, score, x1, y1, x2, y2 = bbox
                if score < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(cv_image.shape[1], x2)
                y2 = min(cv_image.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                valid_detections += 1
                cropped_image = cv_image[y1:y2, x1:x2]

                ocr_start = time.perf_counter()
                ocr_text = self.perform_ocr_immediate(cropped_image) if cropped_image.size > 0 else self.OCR_NO_TEXT
                ocr_elapsed = time.perf_counter() - ocr_start
                frame_ocr_time += ocr_elapsed

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


def default_output_dir(weights_path: str, ocr_lang: str) -> str:
    stem = Path(weights_path).stem
    return str(Path('outputs') / f'{stem}_paddleocr_{ocr_lang}')


def collect_images(infer_dir: str):
    infer_path = Path(infer_dir)
    if not infer_path.exists():
        raise FileNotFoundError(f'Directory not found: {infer_dir}')
    image_paths = sorted([p for p in infer_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
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
    for r in info['results']:
        x1, y1, x2, y2 = r['box']
        print(f"  [{x1},{y1},{x2},{y2}] det={r['score']:.2f} -> '{r['text']}'")


def infer_single_image(pipeline: PaddleOCRPipeline, image_path: str, output_path: str = None, save_image: bool = False):
    info = pipeline.infer_image(image_path)
    bgr = info['orig_bgr']
    print(f"[IMG] {Path(image_path).name} | {bgr.shape[1]}x{bgr.shape[0]}")
    print_detection_lines(info)
    print_per_image_summary(info)

    if save_image and output_path:
        pipeline.save_result(info['orig_bgr'], info['results'], output_path)


def infer_directory(
    pipeline: PaddleOCRPipeline,
    infer_dir: str,
    output_dir: str,
    save_image: bool = False,
    log_interval: int = 1,
):
    image_paths = collect_images(infer_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / 'benchmark_results.txt'
    tee = Tee(str(log_path))

    total_det_time = 0.0
    total_ocr_time = 0.0
    total_e2e_time = 0.0
    total_det_count = 0
    errors = []
    n = len(image_paths)
    width = len(str(n))

    print(f"[CFG] Images      : {n} in {infer_dir}")
    print(f"[CFG] Output dir  : {out_dir}")
    print(f"[CFG] Log file    : {log_path}")
    print(f"[CFG] Save image  : {save_image}")
    print()

    for idx, img_path in enumerate(image_paths, start=1):
        try:
            info = pipeline.infer_image(str(img_path))
            total_det_time += info['detection_time']
            total_ocr_time += info['ocr_time']
            total_e2e_time += info['total_time']
            total_det_count += info['det_count']

            if (idx % log_interval) == 0 or idx == n:
                print(f"[{idx:{width}}/{n}] {img_path.name}")
                print(f"[IMG] {info['orig_bgr'].shape[1]}x{info['orig_bgr'].shape[0]}")
                print_detection_lines(info)
                print_per_image_summary(info)

            if save_image:
                output_path = out_dir / img_path.name
                pipeline.save_result(info['orig_bgr'], info['results'], output_path)

        except Exception as e:
            errors.append((img_path.name, str(e)))
            print(f"[ERROR] {img_path.name}: {e}")

    processed = n - len(errors)
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
        if total_det_time > 0:
            print(f'Detection FPS avg      : {processed / total_det_time:.2f}')
        else:
            print('Detection FPS avg      : N/A')
        if total_ocr_time > 0:
            print(f'OCR FPS avg            : {processed / total_ocr_time:.2f}')
        else:
            print('OCR FPS avg            : N/A')
        if total_e2e_time > 0:
            print(f'E2E FPS avg            : {processed / total_e2e_time:.2f}')
        else:
            print('E2E FPS avg            : N/A')
        print('=' * 60)

    if errors:
        print('\nFailed images:')
        for name, err in errors:
            print(f'  {name}: {err}')

    real_stdout = tee._stdout
    tee.close()
    print(f'Log saved to {log_path}', file=real_stdout)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Standalone NavOCR inference (PaddleDetection + PaddleOCR)'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to PaddleDetection config file')
    parser.add_argument('--weights', type=str, default='model/navocr.pdparams',
                        help='Path to PaddleDetection weights (.pdparams)')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to input image')
    parser.add_argument('--infer_dir', type=str, default=None,
                        help='Directory of input images')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path for single-image mode')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for folder mode')
    parser.add_argument('--draw_threshold', type=float, default=0.5,
                        help='Detection confidence threshold')
    parser.add_argument('--ocr_language', type=str, default='en',
                        help='PaddleOCR language')
    parser.add_argument('--use_angle_cls', action='store_true', default=False,
                        help='Enable angle classifier in PaddleOCR')
    parser.add_argument('--save_image', action='store_true',
                        help='Save annotated images')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Print progress every N images in folder mode')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.input is None and args.infer_dir is None:
        parser.error('Either --input or --infer_dir must be provided.')
    if args.input is not None and args.infer_dir is not None:
        parser.error('Use only one of --input or --infer_dir.')

    output_dir = args.output_dir or default_output_dir(args.weights, args.ocr_language)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[CFG] Config      : {args.config}")
    print(f"[CFG] Weights     : {args.weights}")
    print(f"[CFG] OCR lang    : {args.ocr_language}")
    print(f"[CFG] Threshold   : {args.draw_threshold}")
    print(f"[CFG] Output dir  : {output_dir}")

    pipeline = PaddleOCRPipeline(
        config_path=args.config,
        weights_path=args.weights,
        conf_threshold=args.draw_threshold,
        output_dir=output_dir,
        ocr_language=args.ocr_language,
        use_angle_cls=args.use_angle_cls,
    )

    if args.input:
        out_path = args.output or str(Path(output_dir) / Path(args.input).name)
        infer_single_image(
            pipeline=pipeline,
            image_path=args.input,
            output_path=out_path,
            save_image=args.save_image,
        )
    else:
        infer_directory(
            pipeline=pipeline,
            infer_dir=args.infer_dir,
            output_dir=output_dir,
            save_image=args.save_image,
            log_interval=max(1, args.log_interval),
        )


if __name__ == '__main__':
    main()
