"""Optimizador y scheduler estilo DETR.

- **Grupos de parámetros con LR diferenciado** (clave en DETR):
    · backbone        -> lr · `backbone_mult` (0.1×)
    · módulos "lentos" (sampling_offsets de la deformable + anclas) -> lr · `slow_mult` (0.1×)
    · resto (decoder, FFN, cabezas) -> lr
- **Scheduler**: warmup lineal + cosine decay (amortigua la inestabilidad inicial del húngaro).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _is_slow(name: str) -> bool:
    """Parámetros de la atención deformable / prior posicional que conviene mover despacio."""
    return ("sampling_offsets" in name) or name.endswith("anchors.anchors")


def param_groups(model: nn.Module, lr: float, backbone_mult: float = 0.1,
                 slow_mult: float = 0.1) -> list[dict]:
    backbone, slow, rest = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone"):
            backbone.append(p)
        elif _is_slow(name):
            slow.append(p)
        else:
            rest.append(p)
    groups = [
        {"params": rest, "lr": lr, "name": "rest"},
        {"params": backbone, "lr": lr * backbone_mult, "name": "backbone"},
        {"params": slow, "lr": lr * slow_mult, "name": "slow"},
    ]
    return [g for g in groups if len(g["params"]) > 0]


def build_optimizer(model: nn.Module, lr: float = 2e-4, weight_decay: float = 1e-4,
                    backbone_mult: float = 0.1, slow_mult: float = 0.1,
                    betas=(0.9, 0.999)) -> torch.optim.Optimizer:
    groups = param_groups(model, lr, backbone_mult, slow_mult)
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay, betas=betas)


def build_scheduler(optimizer: torch.optim.Optimizer, total_iters: int,
                    warmup_iters: int = 1000, min_lr_ratio: float = 0.0):
    """LambdaLR: rampa lineal hasta `warmup_iters`, luego cosine hasta `min_lr_ratio`."""
    def fn(it: int) -> float:
        if it < warmup_iters:
            return (it + 1) / max(1, warmup_iters)
        prog = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        prog = min(1.0, prog)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
