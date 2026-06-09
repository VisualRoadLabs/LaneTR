"""EMA (Exponential Moving Average) de los pesos — estilo CLRerNet (ExpMomentumEMA).

Mantiene una copia "suavizada" del modelo: `ema = decay·ema + (1-decay)·modelo`, con un
**warmup** del decay (`decay·(1-exp(-updates/tau))`) para no quedar anclado a la
inicialización aleatoria durante los primeros pasos (clave en DETR, donde el inicio es
inestable). Promedia también los buffers de BatchNorm (floats). El número final de la tesis y
los checkpoints se evalúan/entregan con los pesos EMA.

Referencia: CLRerNet usa momentum=0.0001 (=decay 0.9999), tau=2000, update_buffers=True.
"""
from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: float = 2000.0):
        self.ema = deepcopy(_unwrap(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / self.tau))  # decay con warmup
        msd = _unwrap(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach().to(v.device), alpha=1.0 - d)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)
