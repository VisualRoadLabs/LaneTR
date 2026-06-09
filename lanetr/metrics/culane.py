"""Métrica F1 oficial de CULane (port fiel de la usada por CLRNet/CLRerNet).

Equivale al evaluador oficial en C++ (SCNN): cada carril se interpola con spline y se
rasteriza como una línea de `width=30` px sobre una máscara a resolución 1640×590; se calcula
la IoU de píxeles entre cada predicción y cada GT, se emparejan 1-a-1 con el algoritmo húngaro
y un par cuenta como TP si IoU > umbral (0.5 por defecto). El C++ oficial (en
`evaluation/culane_official/`) se usa en el Paso 2B para validar que estos números coinciden.

Convenio de coordenadas: los carriles llegan como listas de puntos (x, y) en PÍXELES a la
resolución ORIGINAL de CULane (1640×590). Para evaluar predicciones del modelo (espacio
800×320) hay que mapearlas antes con `lanetr.metrics.format.resized_to_orig`.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.optimize import linear_sum_assignment

IMG_SHAPE = (590, 1640)  # (H, W) original de CULane
LANE_WIDTH = 30
IOU_THRESHOLDS_DEFAULT = (0.5,)


def interp(points: np.ndarray, n: int = 5) -> np.ndarray:
    """Densifica un carril con spline (como el C++ oficial y CLRNet)."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points
    x, y = points[:, 0], points[:, 1]
    k = min(3, len(points) - 1)
    tck, u = splprep([x, y], s=0, k=k)
    u2 = np.linspace(0.0, 1.0, num=(len(u) - 1) * n + 1)
    return np.asarray(splev(u2, tck)).T  # (M, 2)


def draw_lane(lane: np.ndarray, img_shape=IMG_SHAPE, width=LANE_WIDTH) -> np.ndarray:
    """Rasteriza un carril como línea de `width` px sobre una máscara uint8."""
    img = np.zeros(img_shape, dtype=np.uint8)
    lane = np.asarray(lane, dtype=np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color=255, thickness=width)
    return img


def discrete_cross_iou(xs, ys, width=LANE_WIDTH, img_shape=IMG_SHAPE) -> np.ndarray:
    """Matriz de IoU (px) entre cada carril de `xs` y cada carril de `ys`."""
    xm = [draw_lane(l, img_shape, width) > 0 for l in xs]
    ym = [draw_lane(l, img_shape, width) > 0 for l in ys]
    ious = np.zeros((len(xm), len(ym)))
    for i, x in enumerate(xm):
        for j, y in enumerate(ym):
            union = (x | y).sum()
            ious[i, j] = float((x & y).sum()) / union if union > 0 else 0.0
    return ious


def f1_from_counts(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def culane_metric(pred, anno, width=LANE_WIDTH, iou_thresholds=IOU_THRESHOLDS_DEFAULT,
                  img_shape=IMG_SHAPE) -> dict:
    """TP/FP/FN de una imagen. `pred`/`anno`: listas de carriles (N,2) en coords originales."""
    if len(pred) == 0:
        return {t: [0, 0, len(anno)] for t in iou_thresholds}
    if len(anno) == 0:
        return {t: [0, len(pred), 0] for t in iou_thresholds}

    interp_pred = [interp(p, n=5) for p in pred]
    interp_anno = [interp(a, n=5) for a in anno]
    ious = discrete_cross_iou(interp_pred, interp_anno, width, img_shape)
    row, col = linear_sum_assignment(1 - ious)

    out = {}
    for t in iou_thresholds:
        tp = int((ious[row, col] > t).sum())
        out[t] = [tp, len(pred) - tp, len(anno) - tp]
    return out


def evaluate(preds, annos, width=LANE_WIDTH, iou_thresholds=IOU_THRESHOLDS_DEFAULT,
             img_shape=IMG_SHAPE) -> dict:
    """Agrega TP/FP/FN sobre un conjunto de imágenes y devuelve P/R/F1 por umbral.

    `preds`/`annos`: listas (por imagen) de listas-de-carriles (N,2) en coords originales.
    """
    totals = {t: [0, 0, 0] for t in iou_thresholds}
    for pred, anno in zip(preds, annos):
        m = culane_metric(pred, anno, width, iou_thresholds, img_shape)
        for t in iou_thresholds:
            for k in range(3):
                totals[t][k] += m[t][k]

    ret = {}
    for t in iou_thresholds:
        tp, fp, fn = totals[t]
        p, r, f = f1_from_counts(tp, fp, fn)
        ret[t] = {"TP": tp, "FP": fp, "FN": fn, "Precision": p, "Recall": r, "F1": f}
    return ret


# --------------------------------------------------------------------------- #
# Carga de carriles en formato CULane (.lines.txt)
# --------------------------------------------------------------------------- #
def load_culane_img_data(path: str) -> list[np.ndarray]:
    """Lee un `.lines.txt` -> lista de carriles (N,2) (coords originales)."""
    lanes = []
    if not os.path.exists(path):
        return lanes
    with open(path, "r") as f:
        for line in f:
            v = line.split()
            if len(v) < 4:
                continue
            coords = list(map(float, v))
            pts = np.array([(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)],
                           dtype=np.float32)
            if len(pts) >= 2:
                lanes.append(pts)
    return lanes
