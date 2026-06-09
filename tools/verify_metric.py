"""Verificación VISUAL de la métrica F1 de CULane (Paso 2A).

Toma una imagen real y construye una "predicción" sintética a partir de su GT:
  - un carril casi exacto (TP),
  - un carril desplazado mucho (rompe match -> FP + FN),
  - un carril omitido (FN).
Dibuja las máscaras de 30 px (GT en verde, predicción en rojo, solape en amarillo),
escribe la IoU de cada par y el recuento TP/FP/FN y el F1 resultante. Así se ve que la
métrica casa con lo que uno esperaría a ojo.

Salida: outputs/verify/metric_demo.png

Uso:
    .\.venv\Scripts\python.exe tools\verify_metric.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lanetr import paths
from lanetr.data import culane_annotation as ann
from lanetr.metrics import culane as M


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def mask_rgb(lanes, color, img_shape=M.IMG_SHAPE):
    """Suma de máscaras de 30 px de una lista de carriles, en un color dado."""
    acc = np.zeros(img_shape, np.uint8)
    for l in lanes:
        acc |= M.draw_lane(M.interp(l, n=5), img_shape, M.LANE_WIDTH) > 0
    rgb = np.zeros((*img_shape, 3), np.uint8)
    for c in range(3):
        rgb[:, :, c] = acc * color[c]
    return rgb, acc


def main() -> int:
    print("=" * 70)
    print("VERIFICACIÓN DE LA MÉTRICA F1 DE CULANE (Paso 2A)")
    print("=" * 70)
    font = _font(22)

    # buscar una imagen de val con 4 carriles
    lines = [l for l in (paths.list_dir() / "val_gt.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    chosen = None
    for line in lines:
        image_rel, _, ex = ann.parse_gt_line(line)
        if ex == (1, 1, 1, 1):
            chosen = image_rel
            break
    gt = M.load_culane_img_data(str(ann.lines_path_for_image(chosen)))
    print(f"Imagen: {chosen}   carriles GT: {len(gt)}")

    # construir una predicción sintética a partir del GT
    pred = []
    pred.append(gt[0] + np.array([4.0, 0.0]))    # casi exacto -> TP
    pred.append(gt[1] + np.array([3.0, 0.0]))    # casi exacto -> TP
    pred.append(gt[2] + np.array([70.0, 0.0]))   # muy desplazado -> FP + FN
    # gt[3] se omite -> FN

    res = M.culane_metric(pred, gt, iou_thresholds=(0.5,))[0.5]
    tp, fp, fn = res
    p, r, f1 = M.f1_from_counts(tp, fp, fn)
    ious = M.discrete_cross_iou([M.interp(x) for x in pred], [M.interp(y) for y in gt])
    print("Matriz IoU pred×GT (filas=pred, col=GT):")
    print(np.round(ious, 2))
    print(f"\nResultado: TP={tp} FP={fp} FN={fn}  ->  P={p:.3f} R={r:.3f} F1={f1:.3f}")

    # render: imagen + máscaras
    img = np.array(Image.open(paths.image_path(chosen)).convert("RGB"))
    gt_rgb, _ = mask_rgb(gt, (0, 220, 0))
    pr_rgb, _ = mask_rgb(pred, (230, 0, 0))
    overlay = img.copy()
    overlay = (0.55 * overlay + 0.45 * gt_rgb + 0.45 * pr_rgb).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(overlay)

    d = ImageDraw.Draw(im)
    d.text((10, 10), f"verde=GT  rojo=pred  amarillo=solape   TP={tp} FP={fp} FN={fn}  "
                     f"F1={f1:.3f}", fill=(255, 255, 0), font=font)
    out = paths.outputs_dir() / "verify" / "metric_demo.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    im.resize((1640 // 2, 590 // 2)).save(out)

    print(f"\nImagen guardada en: {out}")
    print("Esperado: 2 TP (carriles casi exactos), 1 FP + 1 FN (carril muy desplazado),")
    print("          1 FN (carril omitido)  =>  TP=2 FP=1 FN=2")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
