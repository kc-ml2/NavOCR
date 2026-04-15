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

import paddle
from ppdet.core.workspace import load_config
from ppdet.engine import Trainer
from ppdet.utils.check import check_version, check_config
from ppdet.utils.cli import merge_args


class PaddleDetector:
    def __init__(self, flags):
        self.cfg = load_config(flags.config)
        merge_args(self.cfg, flags)

        # Resolve relative dataset_dir paths to absolute using config file location
        config_dir = os.path.dirname(os.path.abspath(flags.config))
        navocr_root = os.path.dirname(os.path.dirname(config_dir))
        for key in ['TestDataset', 'EvalDataset', 'TrainDataset']:
            if key in self.cfg and 'dataset_dir' in self.cfg[key]:
                ds_dir = self.cfg[key]['dataset_dir']
                if not os.path.isabs(ds_dir):
                    self.cfg[key]['dataset_dir'] = os.path.join(navocr_root, ds_dir)
        
        self._set_device()
        
        check_config(self.cfg)
        check_version()
        
        # PaddleDetection Trainer initialization
        self.cfg["pretrain_weights"] = None
        self.trainer = Trainer(self.cfg, mode="test")
        self.trainer.load_weights(self.cfg.weights)
        
        self.draw_threshold = flags.draw_threshold
        self.output_dir = flags.output_dir
        

    def _set_device(self):
        device_list = ['gpu', 'npu', 'xpu', 'mlu', 'gcu']
        target_device = 'cpu'
        for dev in device_list:
            if self.cfg.get(f'use_{dev}', False):
                target_device = dev
                break
        paddle.set_device(target_device)

    def infer(self, image_list, visualize=False, save_results=False):
        results = self.trainer.predict(
            image_list,
            draw_threshold=self.draw_threshold,
            output_dir=self.output_dir,
            save_results=save_results, # bbox JSON save option
            visualize=visualize,       # Image save option
            save_threshold=self.draw_threshold
        )

        return results
