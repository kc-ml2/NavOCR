"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import re
import copy

from ._config import BaseConfig
from .workspace import create
from .yaml_utils import load_config, merge_config, merge_dict


class YAMLConfig(BaseConfig):
    def __init__(self, cfg_path: str, **kwargs) -> None:
        super().__init__()

        cfg = load_config(cfg_path)
        cfg = merge_dict(cfg, kwargs)

        self.yaml_cfg = copy.deepcopy(cfg)

        for k in super().__dict__:
            if not k.startswith('_') and k in cfg:
                self.__dict__[k] = cfg[k]

    @property
    def global_cfg(self, ):
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)

    @property
    def model(self, ) -> torch.nn.Module:
        if self._model is None and 'model' in self.yaml_cfg:
            self._model = create(self.yaml_cfg['model'], self.global_cfg)
        return super().model

    @property
    def teacher_model(self, ) -> torch.nn.Module:
        if self._teacher_model is None and 'teacher_model' in self.yaml_cfg:
            raise ModuleNotFoundError(
                'Teacher models are not available in torch_engine.infer_only. '
                'Use torch_engine.training for distillation training.'
            )
        return super().teacher_model

    @property
    def postprocessor(self, ) -> torch.nn.Module:
        if self._postprocessor is None and 'postprocessor' in self.yaml_cfg:
            self._postprocessor = create(self.yaml_cfg['postprocessor'], self.global_cfg)
        return super().postprocessor

    @property
    def criterion(self, ) -> torch.nn.Module:
        if self._criterion is None and 'criterion' in self.yaml_cfg:
            raise ModuleNotFoundError(
                'Criterion is not available in torch_engine.infer_only. '
                'Use torch_engine.training for training losses.'
            )
        return super().criterion

    @property
    def optimizer(self, ) -> optim.Optimizer:
        if self._optimizer is None and 'optimizer' in self.yaml_cfg:
            raise ModuleNotFoundError(
                'Optimizer is not available in torch_engine.infer_only. '
                'Use torch_engine.training for optimization.'
            )
        return super().optimizer

    @property
    def lr_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        if self._lr_scheduler is None and 'lr_scheduler' in self.yaml_cfg:
            raise ModuleNotFoundError(
                'LR scheduler is not available in torch_engine.infer_only. '
                'Use torch_engine.training for training schedules.'
            )
        return super().lr_scheduler

    @property
    def lr_warmup_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        if self._lr_warmup_scheduler is None and 'lr_warmup_scheduler' in self.yaml_cfg :
            raise ModuleNotFoundError(
                'Warmup scheduler is not available in torch_engine.infer_only. '
                'Use torch_engine.training for training schedules.'
            )
        return super().lr_warmup_scheduler

    @property
    def train_dataloader(self, ) -> DataLoader:
        if self._train_dataloader is None and 'train_dataloader' in self.yaml_cfg:
            raise ModuleNotFoundError(
                'Dataloader creation is not available in torch_engine.infer_only. '
                'Use torch_engine.training for dataset pipelines.'
            )
        return super().train_dataloader

    @property
    def val_dataloader(self, ) -> DataLoader:
        if self._val_dataloader is None and 'val_dataloader' in self.yaml_cfg:
            raise ModuleNotFoundError(
                'Dataloader creation is not available in torch_engine.infer_only. '
                'Use torch_engine.training for dataset pipelines.'
            )
        return super().val_dataloader

    @property
    def ema(self, ) -> torch.nn.Module:
        if self._ema is None and self.yaml_cfg.get('use_ema', False):
            raise ModuleNotFoundError(
                'EMA is not available in torch_engine.infer_only. '
                'Use torch_engine.training for training features.'
            )
        return super().ema

    @property
    def scaler(self, ):
        if self._scaler is None and self.yaml_cfg.get('use_amp', False):
            raise ModuleNotFoundError(
                'AMP scaler is not available in torch_engine.infer_only. '
                'Use torch_engine.training for mixed precision training.'
            )
        return super().scaler

    @property
    def evaluator(self, ):
        if self._evaluator is None and 'evaluator' in self.yaml_cfg:
            raise ModuleNotFoundError(
                'Evaluation helpers are not available in torch_engine.infer_only. '
                'Use torch_engine.training for validation and COCO evaluation.'
            )
        return super().evaluator

    @staticmethod
    def get_optim_params(cfg: dict, model: nn.Module):
        """
        E.g.:
            ^(?=.*a)(?=.*b).*$  means including a and b
            ^(?=.*(?:a|b)).*$   means including a or b
            ^(?=.*a)(?!.*b).*$  means including a, but not b
        """
        assert 'type' in cfg, ''
        cfg = copy.deepcopy(cfg)

        if 'params' not in cfg:
            return model.parameters()

        assert isinstance(cfg['params'], list), ''

        param_groups = []
        visited = []
        for pg in cfg['params']:
            pattern = pg['params']
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
            pg['params'] = params.values()
            param_groups.append(pg)
            visited.extend(list(params.keys()))
            # print(params.keys())

        names = [k for k, v in model.named_parameters() if v.requires_grad]

        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            param_groups.append({'params': params.values()})
            visited.extend(list(params.keys()))
            # print(params.keys())

        assert len(visited) == len(names), ''

        return param_groups

    @staticmethod
    def get_rank_batch_size(cfg):
        """compute batch size for per rank if total_batch_size is provided.
        """
        raise ModuleNotFoundError(
            'Rank-aware batch sizing is not available in torch_engine.infer_only. '
            'Use torch_engine.training for dataset pipelines.'
        )

    def build_dataloader(self, name: str):
        raise ModuleNotFoundError(
            'Dataloader construction is not available in torch_engine.infer_only. '
            'Use torch_engine.training for dataset pipelines.'
        )
