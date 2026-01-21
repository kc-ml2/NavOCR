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


import paddle
from ppdet.core.workspace import load_config
from ppdet.engine import Trainer
from ppdet.utils.check import check_version, check_config
from ppdet.utils.cli import merge_args


class PaddleDetector:
    def __init__(self, flags):
        """Initialize and load weights"""
        self.cfg = load_config(flags.config)
        merge_args(self.cfg, flags)
        
        self._set_device()
        
        check_config(self.cfg)
        check_version()
        
        # PaddleDetection Trainer initialization
        self.cfg["pretrain_weights"] = None
        self.trainer = Trainer(self.cfg, mode="test")
        self.trainer.load_weights(self.cfg.weights)
        
        self.draw_threshold = flags.draw_threshold
        self.output_dir = flags.output_dir
        
        # Metadata fro visualization
        self.clsid2catid = getattr(self.trainer.dataset, 'clsid2catid', None)
        self.catid2name = getattr(self.trainer.dataset, 'catid2name', None)

    def _set_device(self):
        device_list = ['gpu', 'npu', 'xpu', 'mlu', 'gcu']
        target_device = 'cpu'
        for dev in device_list:
            if self.cfg.get(f'use_{dev}', False):
                target_device = dev
                break
        paddle.set_device(target_device)

    def infer(self, image_list, visualize=False, save_results=False):
        """
        Given image list, return inference result.
        """

        results = self.trainer.predict(
            image_list,
            draw_threshold=self.draw_threshold,
            output_dir=self.output_dir,
            save_results=save_results, # bbox JSON save option
            visualize=visualize,       # Image save option
            save_threshold=self.draw_threshold
        )

        return results
