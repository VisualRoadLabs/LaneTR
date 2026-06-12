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
        "n_ref_points": 1,        # puntos de referencia a lo largo del carril (Paso 7); 1 = modelo orig.
        "ref_refine": False,      # refinamiento iterativo de las refs hacia el carril predicho (Paso 7.2)
        "ref_refine_mode": "mlp", # "mlp" (DAB-DETR) | "xs" (deriva del xs ya predicho por la cabeza)
        "ref_y_top": 0.15, "ref_y_bottom": 0.95,  # alturas de las refs (n_ref>1); baja y_bottom = sin borde inferior
    },
    "data": {
        "img_w": 800, "img_h": 320, "num_rows": 144,
        "batch_size": 32, "num_workers": 8, "augment": True,
        "train_split": "train",   # "train"=filtrado (coche parado); "train_full"=88.880 (ablation filtro)
        # sobre-muestreo de curvas (Paso 7.3); requiere list/train_curvature.npz (compute_curvature.py)
        "curve_oversample": False, "curve_alpha": 4.0, "curve_top_frac": 0.1,
    },
    "optim": {
        "lr": 2.0e-4, "weight_decay": 1.0e-4,
        "backbone_mult": 0.1, "slow_mult": 0.1, "warmup_iters": 1000,
    },
    "loss": {
        "w_cls": 2.0, "w_iou": 4.0, "w_xy": 0.2, "w_ext": 0.5,
        "focal_alpha": 0.25, "focal_gamma": 2.0,
        "aux_one_to_many": False, "o2m_k": 4, "w_smooth": 0.0,
        "geo_metric": "laneiou",  # "laneiou" (tesis) | "lineiou" | "distance" (ablation)
        # énfasis en curvas (Paso 7.3): pesa el término geométrico por la curvatura del GT
        "curve_gamma": 0.0,       # 0 = sin énfasis (igual que antes); ~4 para enfatizar curvas
        "curve_thresh": 0.005, "curve_scale": 0.03, "curve_cap": 2.0,  # recto≈0, curva ~0.03 (xs norm.)
        "w_curv": 0.0,            # término directo que iguala la 2ª diferencia (curvatura) pred vs GT
    },
    "ema": {"enabled": True, "decay": 0.9999, "tau": 2000.0},
    "train": {
        "epochs": 15, "grad_clip": 0.1, "amp": True, "channels_last": True,
        "freeze_bn": True, "eval_interval": 3, "log_interval": 50,
        "ckpt_dir": "outputs/checkpoints", "seed": 0,
        "eval_conf_thresh": 0.5, "eval_max_images": None, "eval_batch_size": 16,
        "final_eval": True,   # al acabar: calibra umbral en val + eval test global y por categoría -> results.json
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
