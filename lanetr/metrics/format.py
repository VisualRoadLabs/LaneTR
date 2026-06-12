"""Formato de predicciones de CULane y mapeo de coordenadas.

El modelo trabaja en el espacio 800×320 (tras recorte y≥270 y resize). La métrica oficial
opera en la resolución original 1640×590. Aquí se hace el mapeo inverso y la escritura de
predicciones en el formato `.lines.txt` que consume el evaluador oficial en C++.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ORIG_W, ORIG_H = 1640, 590
IMG_W, IMG_H = 800, 320
CUT_HEIGHT = 270


def resized_to_orig(points: np.ndarray, img_w=IMG_W, img_h=IMG_H, cut_height=CUT_HEIGHT,
                    orig_w=ORIG_W, orig_h=ORIG_H) -> np.ndarray:
    """Mapea puntos del espacio del modelo (img_w×img_h) a la resolución original.

    Inverso de `CropResize`:  x_orig = x * orig_w/img_w ;
    y_orig = y / (img_h/(orig_h-cut)) + cut.
    """
    scale_x = orig_w / img_w
    scale_y = img_h / (orig_h - cut_height)
    q = np.asarray(points, dtype=np.float64).copy()
    q[:, 0] = q[:, 0] * scale_x
    q[:, 1] = q[:, 1] / scale_y + cut_height
    return q


def lane_to_line_str(points: np.ndarray) -> str:
    """Carril (N,2) -> línea de texto 'x1 y1 x2 y2 ...' (formato CULane)."""
    flat = []
    for x, y in points:
        flat.append(f"{x:.3f}")
        flat.append(f"{y:.3f}")
    return " ".join(flat)


def write_lines_file(lanes: list[np.ndarray], out_path: str | Path) -> None:
    """Escribe un `.lines.txt` con un carril por línea (vacío si no hay carriles)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [lane_to_line_str(l) for l in lanes if len(l) >= 2]
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_culane_predictions(lanes_per_image: list[list[np.ndarray]],
                             image_rels: list[str], pred_dir: str | Path,
                             to_orig: bool = True) -> None:
    """Escribe las predicciones de varias imágenes en `pred_dir`, replicando la estructura
    de carpetas de CULane (`<pred_dir>/driver_xx/.../00000.lines.txt`).

    `lanes_per_image[i]` son los carriles de la imagen `image_rels[i]` (en espacio del modelo
    si `to_orig=True`, o ya en coords originales si `to_orig=False`).
    """
    pred_dir = Path(pred_dir)
    for lanes, rel in zip(lanes_per_image, image_rels):
        mapped = [resized_to_orig(l) if to_orig else np.asarray(l) for l in lanes]
        rel_txt = rel.lstrip("/\\").replace(".jpg", ".lines.txt")
        write_lines_file(mapped, pred_dir / rel_txt)
