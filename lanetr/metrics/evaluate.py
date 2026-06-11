"""Evaluación F1 de un modelo LaneTR sobre CULane (Paso 6.3).

Corre el modelo sobre una lista de imágenes, mapea las predicciones a 1640×590 y calcula el F1
con la métrica Python (validada == C++ en el Paso 2). Incluye:
  - `evaluate_list`: F1 sobre un fichero de lista (val/test/categoría) a un umbral dado.
  - `calibrate_threshold`: barre umbrales de confianza y elige el mejor (NMS-free: umbral + tope 4).
  - `evaluate_test_and_categories`: F1 de test global + las 9 categorías en UNA sola inferencia
    (Crossroad cuenta FP). Mucho más rápido que evaluar cada categoría por separado.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import paths
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
    """Devuelve (raw, annos, rels):
       raw[i]   = [{'conf': float, 'points': (N,2) en coords 1640×590}] para TODAS las queries.
       annos[i] = carriles GT (coords originales) leídos del `.lines.txt`.
       rels[i]  = ruta relativa de la imagen i.
    """
    model.eval()
    ds = CULaneDataset("test", list_file=list_file, augment=False, encode_targets=False,
                       img_w=img_w, img_h=img_h, num_rows=num_rows)
    if max_images:
        ds.entries = ds.entries[:max_images]
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    collate_fn=collate_lanes)
    raw, annos, rels = [], [], []
    for batch in dl:
        images = batch["image"].to(device)
        pred = model(images)
        lanes_b = decode_lanes(pred, layer=-1, conf_thresh=None, num_rows=num_rows,
                               img_w=img_w, img_h=img_h)
        for b, meta in enumerate(batch["meta"]):
            raw.append([{"conf": l["conf"], "points": F.resized_to_orig(l["points"], img_w, img_h)}
                        for l in lanes_b[b]])
            annos.append(M.load_culane_img_data(str(lines_path_for_image(meta["image_path"]))))
            rels.append(meta["image_path"])
    return raw, annos, rels


def per_image_counts(raw, annos, thr, max_lanes=4):
    """TP/FP/FN por imagen al umbral `thr` (filtra conf>thr + tope 4, sin NMS). -> lista de [tp,fp,fn]."""
    out = []
    for per_img, anno in zip(raw, annos):
        kept = sorted([r for r in per_img if r["conf"] >= thr], key=lambda r: -r["conf"])[:max_lanes]
        preds = [r["points"] for r in kept]
        out.append(M.culane_metric(preds, anno)[0.5])
    return out


def _agg(counts, idx=None) -> dict:
    sel = counts if idx is None else [counts[i] for i in idx]
    tp = sum(c[0] for c in sel)
    fp = sum(c[1] for c in sel)
    fn = sum(c[2] for c in sel)
    p, r, f = M.f1_from_counts(tp, fp, fn)
    return {"TP": tp, "FP": fp, "FN": fn, "Precision": p, "Recall": r, "F1": f}


def f1_at_threshold(raw, annos, thr, max_lanes=4) -> dict:
    return _agg(per_image_counts(raw, annos, thr, max_lanes))


def evaluate_list(model, list_file, device="cuda", conf_thresh=0.5, **kw) -> dict:
    raw, annos, _ = infer(model, list_file, device, **kw)
    return f1_at_threshold(raw, annos, conf_thresh)


def calibrate_threshold(model, list_file, device="cuda", thresholds=None, **kw):
    """Barre umbrales sobre `list_file` (típicamente val/subconjunto) -> (mejor_umbral, scores)."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.85, 0.05)]
    raw, annos, _ = infer(model, list_file, device, **kw)
    scores = {float(t): f1_at_threshold(raw, annos, float(t)) for t in thresholds}
    best = max(scores, key=lambda t: scores[t]["F1"])
    return best, scores


def evaluate_test_and_categories(model, device="cuda", conf_thresh=0.5, test_list="test.txt", **kw):
    """F1 de test GLOBAL + por las 9 categorías, con UNA sola pasada de inferencia y métrica.
    Devuelve (overall, cats). Para `cross` (Crossroad) el número relevante es FP."""
    raw, annos, rels = infer(model, test_list, device, **kw)
    counts = per_image_counts(raw, annos, conf_thresh)
    overall = _agg(counts)
    rel_to_idx = {r.lstrip("/\\"): i for i, r in enumerate(rels)}
    cats = {}
    for name, lf in CATEGORIES.items():
        crels = [l.strip().lstrip("/\\")
                 for l in (paths.list_dir() / lf).read_text(encoding="utf-8").splitlines() if l.strip()]
        idx = [rel_to_idx[r] for r in crels if r in rel_to_idx]
        cats[name] = _agg(counts, idx)
    return overall, cats


def evaluate_categories(model, device="cuda", conf_thresh=0.5, **kw) -> dict:
    """F1 por categoría (una sola inferencia internamente)."""
    _, cats = evaluate_test_and_categories(model, device, conf_thresh, **kw)
    return cats
