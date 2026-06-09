"""Evaluación F1 de un modelo LaneTR sobre CULane (Paso 6.3).

Corre el modelo sobre una lista de imágenes, mapea las predicciones a 1640×590 y calcula el F1
con la métrica Python (validada == C++ en el Paso 2). Incluye:
  - `evaluate_list`: F1 sobre un fichero de lista (val/test/categoría) a un umbral dado.
  - `calibrate_threshold`: barre umbrales de confianza y elige el mejor (NMS-free: umbral + tope 4).
  - `evaluate_categories`: F1 por las 9 categorías de test (Crossroad cuenta FP).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.culane_annotation import lines_path_for_image
from ..data.culane_dataset import CULaneDataset, collate_lanes
from ..models.head import decode_lanes
from . import culane as M
from . import format as F

CATEGORIES = {
    "normal": "test_split/test0_normal.txt",
    "crowd": "test_split/test1_crowd.txt",
    "dazzle": "test_split/test2_hlight.txt",
    "shadow": "test_split/test3_shadow.txt",
    "noline": "test_split/test4_noline.txt",
    "arrow": "test_split/test5_arrow.txt",
    "curve": "test_split/test6_curve.txt",
    "cross": "test_split/test7_cross.txt",
    "night": "test_split/test8_night.txt",
}


@torch.no_grad()
def infer(model, list_file, device="cuda", batch_size=16, num_workers=0,
          img_w=800, img_h=320, num_rows=144, max_images=None):
    """Devuelve (raw, annos):
       raw[i]   = [{'conf': float, 'points': (N,2) en coords 1640×590}] para TODAS las queries.
       annos[i] = carriles GT (coords originales) leídos del `.lines.txt`.
    """
    model.eval()
    ds = CULaneDataset("test", list_file=list_file, augment=False, encode_targets=False,
                       img_w=img_w, img_h=img_h, num_rows=num_rows)
    if max_images:
        ds.entries = ds.entries[:max_images]
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    collate_fn=collate_lanes)
    raw, annos = [], []
    for batch in dl:
        images = batch["image"].to(device)
        pred = model(images)
        lanes_b = decode_lanes(pred, layer=-1, conf_thresh=None, num_rows=num_rows,
                               img_w=img_w, img_h=img_h)
        for b, meta in enumerate(batch["meta"]):
            raw.append([{"conf": l["conf"], "points": F.resized_to_orig(l["points"], img_w, img_h)}
                        for l in lanes_b[b]])
            annos.append(M.load_culane_img_data(str(lines_path_for_image(meta["image_path"]))))
    return raw, annos


def f1_at_threshold(raw, annos, thr, max_lanes=4):
    """F1 filtrando por confianza > thr y quedándose con los `max_lanes` más confiados (sin NMS)."""
    preds = []
    for per_img in raw:
        kept = sorted([r for r in per_img if r["conf"] >= thr], key=lambda r: -r["conf"])[:max_lanes]
        preds.append([r["points"] for r in kept])
    return M.evaluate(preds, annos)[0.5]


def evaluate_list(model, list_file, device="cuda", conf_thresh=0.5, **kw):
    raw, annos = infer(model, list_file, device, **kw)
    return f1_at_threshold(raw, annos, conf_thresh)


def calibrate_threshold(model, list_file, device="cuda", thresholds=None, **kw):
    """Barre umbrales sobre `list_file` (típicamente val) y devuelve (mejor_umbral, scores)."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.85, 0.05)]
    raw, annos = infer(model, list_file, device, **kw)
    scores = {float(t): f1_at_threshold(raw, annos, float(t)) for t in thresholds}
    best = max(scores, key=lambda t: scores[t]["F1"])
    return best, scores


def evaluate_categories(model, device="cuda", conf_thresh=0.5, **kw):
    """F1 por categoría. Para `cross` (Crossroad) el número relevante es FP (no hay GT)."""
    return {name: evaluate_list(model, lf, device, conf_thresh, **kw)
            for name, lf in CATEGORIES.items()}
