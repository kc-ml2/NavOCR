# Copyright (c) 2026 Chaehyeuk Lee (KC ML2). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import List

import cv2
import numpy as np


class OpenVINODetector:
    def __init__(self, flags):
        try:
            import openvino as ov
        except ImportError as exc:
            raise ImportError("OpenVINO not installed. Run: pip install openvino") from exc

        self.ov = ov
        self.xml_path = getattr(flags, 'det_model_path', None) or getattr(flags, 'weights', None)
        if not self.xml_path:
            raise ValueError('Detector model path is required')
        if not os.path.isfile(self.xml_path):
            raise FileNotFoundError(f'OpenVINO detector model not found: {self.xml_path}')

        self.draw_threshold = float(getattr(flags, 'draw_threshold', 0.5))
        self.output_dir = getattr(flags, 'output_dir', '')
        self.device = getattr(flags, 'device', 'CPU')
        self.imgsz = int(getattr(flags, 'imgsz', 640))

        core = ov.Core()
        config = {'PERFORMANCE_HINT': 'LATENCY'}
        if str(self.device).upper().startswith('GPU'):
            config['INFERENCE_PRECISION_HINT'] = 'f32'

        model = core.read_model(self.xml_path)
        self.compiled = core.compile_model(model, self.device, config=config)
        self.infer_request = self.compiled.create_infer_request()

        self.images_input = self._resolve_input(['images'])
        self.orig_sizes_input = self._resolve_input(['orig_target_sizes'])
        self.labels_output = self._resolve_output(['labels'])
        self.boxes_output = self._resolve_output(['boxes'])
        self.scores_output = self._resolve_output(['scores'])

    def _resolve_input(self, preferred_names: List[str]):
        for name in preferred_names:
            try:
                return self.compiled.input(name)
            except Exception:
                pass
        for port in self.compiled.inputs:
            any_name = getattr(port, 'any_name', '') or ''
            if any(pref in any_name for pref in preferred_names):
                return port
        if len(self.compiled.inputs) == 1:
            return self.compiled.input(0)
        raise RuntimeError(f'Failed to resolve input from {preferred_names}')

    def _resolve_output(self, preferred_names: List[str]):
        for name in preferred_names:
            try:
                return self.compiled.output(name)
            except Exception:
                pass
        for port in self.compiled.outputs:
            any_name = getattr(port, 'any_name', '') or ''
            if any(pref in any_name for pref in preferred_names):
                return port
        raise RuntimeError(f'Failed to resolve output from {preferred_names}')

    @staticmethod
    def _letterbox(image: np.ndarray, size: int = 640):
        h, w = image.shape[:2]
        ratio = min(size / w, size / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_w = (size - new_w) // 2
        pad_h = (size - new_h) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        return padded, ratio, pad_w, pad_h

    def _preprocess(self, bgr: np.ndarray):
        padded, ratio, pad_w, pad_h = self._letterbox(bgr, self.imgsz)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = rgb.transpose(2, 0, 1)[np.newaxis]
        orig_size = np.array([[self.imgsz, self.imgsz]], dtype=np.int64)
        return blob, orig_size, ratio, pad_w, pad_h

    @staticmethod
    def _map_boxes_to_original(boxes: np.ndarray, ratio: float, pad_w: int, pad_h: int,
                               orig_w: int, orig_h: int) -> np.ndarray:
        boxes = boxes.copy()
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes /= ratio
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)
        return boxes

    def infer(self, image_list, visualize=False, save_results=False):
        del save_results
        results = []
        for image_path in image_list:
            bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f'Cannot read image: {image_path}')

            orig_h, orig_w = bgr.shape[:2]
            blob, orig_size, ratio, pad_w, pad_h = self._preprocess(bgr)
            outputs = self.infer_request.infer({
                self.images_input.any_name: blob,
                self.orig_sizes_input.any_name: orig_size,
            })

            labels = outputs[self.labels_output][0]
            boxes = outputs[self.boxes_output][0]
            scores = outputs[self.scores_output][0]
            boxes = self._map_boxes_to_original(boxes, ratio, pad_w, pad_h, orig_w, orig_h)

            keep = scores > self.draw_threshold
            packed = []
            for lbl, score, box in zip(labels[keep], scores[keep], boxes[keep]):
                x1, y1, x2, y2 = box.astype(np.float32).tolist()
                packed.append([float(lbl), float(score), x1, y1, x2, y2])

            results.append({'bbox': np.asarray(packed, dtype=np.float32)})

            if visualize and self.output_dir:
                vis = bgr.copy()
                for _, score, x1, y1, x2, y2 in packed:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(
                        vis,
                        f'text {score:.2f}',
                        (int(x1), max(0, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                os.makedirs(self.output_dir, exist_ok=True)
                cv2.imwrite(os.path.join(self.output_dir, os.path.basename(image_path)), vis)

        return results
