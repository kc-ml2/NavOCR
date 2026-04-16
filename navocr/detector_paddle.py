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
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import paddle
from ppdet.core.workspace import load_config
from ppdet.engine import Trainer
from ppdet.utils.check import check_version, check_config
from ppdet.utils.cli import merge_args

from navocr.config_loader import infer_project_root
from navocr.detector_base import BaseDetector, PaddleDetectorConfig


class PaddleDetector(BaseDetector):
    def __init__(self, config: PaddleDetectorConfig):
        super().__init__(config)
        if not self.config.detector_config_path:
            raise ValueError('Paddle detector config path is required')
        if not self.config.model_path:
            raise ValueError('Paddle detector model path is required')

        self.cfg = load_config(self.config.detector_config_path)
        merge_args(self.cfg, self.config)
        self.cfg.weights = self.config.model_path

        # Resolve relative dataset_dir paths to absolute using config file location
        config_path = Path(self.config.detector_config_path).resolve()
        navocr_root = infer_project_root(config_path)
        for key in ['TestDataset', 'EvalDataset', 'TrainDataset']:
            if key in self.cfg and 'dataset_dir' in self.cfg[key]:
                ds_dir = self.cfg[key]['dataset_dir']
                if not os.path.isabs(ds_dir):
                    self.cfg[key]['dataset_dir'] = str(navocr_root / ds_dir)

        self._set_device()

        check_config(self.cfg)
        check_version()

        # PaddleDetection Trainer initialization
        self.cfg["pretrain_weights"] = None
        self.trainer = Trainer(self.cfg, mode="test")
        self.trainer.load_weights(self.config.model_path)

    def _set_device(self):
        if self.config.device:
            paddle.set_device(str(self.config.device).lower())
            return

        device_list = ['gpu', 'npu', 'xpu', 'mlu', 'gcu']
        target_device = 'cpu'
        for dev in device_list:
            if self.cfg.get(f'use_{dev}', False):
                target_device = dev
                break
        paddle.set_device(target_device)

    def infer(self, image_list):
        if self.config.quiet_logs:
            sink = StringIO()
            with redirect_stdout(sink), redirect_stderr(sink):
                results = self.trainer.predict(
                    image_list,
                    draw_threshold=self.detection_threshold,
                    output_dir=self.output_dir,
                    save_results=False,
                    visualize=False,
                    save_threshold=self.detection_threshold
                )
        else:
            results = self.trainer.predict(
                image_list,
                draw_threshold=self.detection_threshold,
                output_dir=self.output_dir,
                save_results=False,
                visualize=False,
                save_threshold=self.detection_threshold
            )

        return results
