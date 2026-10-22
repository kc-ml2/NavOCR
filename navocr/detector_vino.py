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
import openvino as ov

from navocr.detector_base import BaseDetector, OpenVINODetectorConfig


class OpenVINODetector(BaseDetector):
    def __init__(self, config: OpenVINODetectorConfig):
        super().__init__(config)
        self.xml_path = self.config.model_path
        if not self.xml_path:
            raise ValueError('Detector model path is required')
        if not os.path.isfile(self.xml_path):
            raise FileNotFoundError(f'OpenVINO detector model not found: {self.xml_path}')
        if self.config.imgsz is None:
            raise ValueError('OpenVINO detector imgsz is required')

        self.device = self.config.device or 'CPU'
        self.imgsz = int(self.config.imgsz)

        core = ov.Core()
        config = {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32'}

        model = core.read_model(self.xml_path)
        self.compiled = core.compile_model(model, self.device, config=config)
        self.infer_request = self.compiled.create_infer_request()

        self.image_input = self.compiled.input('image')
        self.scale_input = self.compiled.input('scale_factor')
        self.bbox_output = self.compiled.output('fetch_name_0')
        self.count_output = self.compiled.output('fetch_name_1')

    def _preprocess(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        ratio = self.imgsz / max(h, w)
        new_h, new_w = int(h * ratio + 0.5), int(w * ratio + 0.5)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, None, fx=new_w / w, fy=new_h / h,
            interpolation=cv2.INTER_CUBIC,
        )

        padded = np.ones((self.imgsz, self.imgsz, 3), dtype=np.float32)
        padded *= np.array([123.675, 116.28, 103.53], dtype=np.float32)
        padded[:new_h, :new_w] = resized
        padded *= 1.0 / 255.0
        padded -= np.array([0.485, 0.456, 0.406])[None, None, :]
        padded /= np.array([0.229, 0.224, 0.225])[None, None, :]

        return {
            'image': np.ascontiguousarray(padded.transpose(2, 0, 1)[None]),
            'scale_factor': np.array([[new_h / h, new_w / w]], dtype=np.float32),
        }

    def infer(self, image_list):
        results = []
        for image_path in image_list:
            bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f'Cannot read image: {image_path}')
            results.append(self._infer_bgr(bgr))
        return results

    def infer_loaded_images(self, image_list: List[np.ndarray]):
        return [self._infer_bgr(bgr) for bgr in image_list]

    def _infer_bgr(self, bgr: np.ndarray):
        if bgr is None:
            raise ValueError('Input image is None')
        if not isinstance(bgr, np.ndarray):
            raise TypeError(f'Expected np.ndarray, got {type(bgr)!r}')

        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f'Expected BGR image with shape [H, W, 3], got {bgr.shape}')

        feed = self._preprocess(bgr)
        outputs = self.infer_request.infer({
            self.image_input.any_name: feed['image'],
            self.scale_input.any_name: feed['scale_factor'],
        })
        boxes = np.asarray(outputs[self.bbox_output], dtype=np.float32)
        count = outputs[self.count_output].item()
        if boxes.shape != (count, 6):
            raise RuntimeError(f'Expected bbox shape ({count}, 6), got {boxes.shape}')

        keep = (boxes[:, 0] >= 0) & (boxes[:, 1] >= self.detection_threshold)
        return {'bbox': boxes[keep]}
