"""Configuración del entrenamiento (cargada de YAML, con valores por defecto).

`load_config(path, overrides)` devuelve un dict con TODOS los hiperparámetros, fusionando:
    DEFAULT  <  fichero YAML  <  overrides (p.ej. flags de ablation).
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT: dict = {
    "name": "lanetr_dla34_culane",
    "model": {
        "backbone": "dla34", "pretrained": True, "d_model": 256,
        "num_queries": 12, "num_layers": 6, "num_rows": 144,
        "use_anchors": True, "deformable": True, "n_points": 4,
    },
    "data": {
        "img_w": 800, "img_h": 320, "num_rows": 144,
        "batch_size": 32, "num_workers": 8, "augment": True,
    },
    "optim": {
        "lr": 2.0e-4, "weight_decay": 1.0e-4,
        "backbone_mult": 0.1, "slow_mult": 0.1, "warmup_iters": 1000,
    },
    "loss": {
        "w_cls": 2.0, "w_iou": 4.0, "w_xy": 0.2, "w_ext": 0.5,
        "focal_alpha": 0.25, "focal_gamma": 2.0,
        "aux_one_to_many": False, "o2m_k": 4, "w_smooth": 0.0,
    },
    "ema": {"enabled": True, "decay": 0.9999, "tau": 2000.0},
    "train": {
        "epochs": 15, "grad_clip": 0.1, "amp": True, "channels_last": True,
        "freeze_bn": True, "eval_interval": 3, "log_interval": 50,
        "ckpt_dir": "outputs/checkpoints", "seed": 0,
        "eval_conf_thresh": 0.5, "eval_max_images": None, "eval_batch_size": 16,
    },
}


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT)
    if path is not None:
        user = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        _deep_update(cfg, user)
    if overrides:
        _deep_update(cfg, overrides)
    return cfg
