"""Utilidades de entrenamiento: FrozenBatchNorm, EMA, optimizador y scheduler."""

from .ema import ModelEMA
from .frozen_bn import FrozenBatchNorm2d, freeze_batchnorm
from .optim import build_optimizer, build_scheduler, param_groups
from .visualize import EpochVisualizer, select_fixed_images
from .workdir import ETA, GPUMonitor, create_work_dir, format_eta, get_logger

__all__ = [
    "FrozenBatchNorm2d", "freeze_batchnorm", "ModelEMA",
    "build_optimizer", "build_scheduler", "param_groups",
    "EpochVisualizer", "select_fixed_images",
    "create_work_dir", "get_logger", "format_eta", "ETA", "GPUMonitor",
]
