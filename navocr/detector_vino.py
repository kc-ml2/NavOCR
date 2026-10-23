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

import numpy as np
import openvino as ov

from navocr.detector_base import BaseDetector, OpenVINODetectorConfig
from navocr.detector_preprocess import preprocess_ppyoloe, validate_preprocess_input_shape
from navocr.pipeline_utils import load_image


class OpenVINODetector(BaseDetector):
    DETECTIONS_OUTPUT_NAME = 'fetch_name_0'
    DETECTION_COUNT_OUTPUT_NAME = 'fetch_name_1'

    def __init__(self, config: OpenVINODetectorConfig):
        super().__init__(config)
        self.xml_path = self.config.model_path
        if not self.xml_path:
            raise ValueError('Detector model path is required')
        if not os.path.isfile(self.xml_path):
            raise FileNotFoundError(f'OpenVINO detector model not found: {self.xml_path}')
        if self.config.preprocess is None:
            raise ValueError('OpenVINO detector preprocessing config is required')

        self.device = self.config.device or 'CPU'

        core = ov.Core()
        config = {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32'}

        model = core.read_model(self.xml_path)
        shape = [dim.get_length() if dim.is_static else None for dim in model.input('image').partial_shape]
        validate_preprocess_input_shape(shape, self.config.preprocess)
        self.compiled = core.compile_model(model, self.device, config=config)
        self.infer_request = self.compiled.create_infer_request()

        self.image_input = self.compiled.input('image')
        self.scale_input = self.compiled.input('scale_factor')
        self.detections_output = self.compiled.output(self.DETECTIONS_OUTPUT_NAME)
        self.detection_count_output = self.compiled.output(self.DETECTION_COUNT_OUTPUT_NAME)

    def _preprocess(self, bgr: np.ndarray):
        return preprocess_ppyoloe(bgr, self.config.preprocess)

    def infer(self, image_list):
        results = []
        for image_path in image_list:
            results.append(self._infer_bgr(load_image(image_path)))
        return results

    def infer_loaded_images(self, image_list: List[np.ndarray]):
        return [self._infer_bgr(bgr) for bgr in image_list]

    def _infer_bgr(self, bgr: np.ndarray):
        feed = self._preprocess(bgr)
        outputs = self.infer_request.infer({
            self.image_input.any_name: feed['image'],
            self.scale_input.any_name: feed['scale_factor'],
        })
        boxes = np.asarray(outputs[self.detections_output], dtype=np.float32)
        count = outputs[self.detection_count_output].item()
        if boxes.shape != (count, 6):
            raise RuntimeError(f'Expected bbox shape ({count}, 6), got {boxes.shape}')

        keep = (boxes[:, 0] >= 0) & (boxes[:, 1] >= self.detection_threshold)
        return {'bbox': boxes[keep]}
