"""Verificación VISUAL de LaneIoU vs LineIoU (Paso 4.1) — figura para la tesis.

Dos paneles:
  A) Un carril real INCLINADO con su banda virtual: LineIoU (anchura constante) vs LaneIoU
     (anchura adaptada al ángulo). Se ve que LaneIoU se ENSANCHA en los tramos inclinados,
     igual que la banda perpendicular de 30 px de la métrica.
  B) Dispersión: para muchas (carril, desplazamiento), se compara la IoU de entrenamiento
     (LaneIoU y LineIoU) con la IoU de MÁSCARA real (la métrica F1). LaneIoU se pega más a la
     diagonal y=x -> aproxima mejor la métrica. Se anota el MAE de cada uno.

Salida: outputs/verify/lane_iou.png

Uso:
    .\.venv\Scripts\python.exe tools\verify_lane_iou.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from lanetr import paths
from lanetr.data import culane_annotation as ann
from lanetr.data import target_encoding as TE
from lanetr.data import transforms as T
from lanetr.losses import lane_iou as LI
from lanetr.metrics import culane as Mc
from lanetr.metrics import format as F

R, IMG_W, IMG_H = 144, 800, 320
ROW_YS = TE.make_row_ys(IMG_H, R)
DISP_W = 1640  # mostrar en aspecto original (x sin comprimir) para que el ángulo sea real


def real_lanes(n_images=15):
    lines = [l for l in (paths.list_dir() / "val_gt.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    cr = T.CropResize(IMG_W, IMG_H, 270)
    out = []
    for line in lines[:n_images]:
        image_rel, seg, ex = ann.parse_gt_line(line)
        a = ann.load_annotation(image_rel, ex, seg)
        sample = {"image": Image.new("RGB", (1640, 590)),
                  "lanes": [l.points.copy() for l in a.lanes], "slots": [l.slot for l in a.lanes],
                  "existence": a.existence, "meta": {}}
        sample = cr(sample, np.random.default_rng(0))
        for pts in sample["lanes"]:
            t = TE.encode_lane(pts, ROW_YS, IMG_W, IMG_H)
            if t.valid.sum() >= 12:
                out.append((t.xs.astype(np.float32), t.valid))
    return out


def mask_iou(pred_xs, gt_xs, valid):
    pa = F.resized_to_orig(TE.decode_lane(pred_xs, valid, ROW_YS, IMG_W))
    pb = F.resized_to_orig(TE.decode_lane(gt_xs, valid, ROW_YS, IMG_W))
    if len(pa) < 2 or len(pb) < 2:
        return 0.0
    ma = Mc.draw_lane(Mc.interp(pa)) > 0
    mb = Mc.draw_lane(Mc.interp(pb)) > 0
    u = (ma | mb).sum()
    return float((ma & mb).sum()) / u if u > 0 else 0.0


def _t(x):
    return torch.tensor(np.asarray(x)[None])


def tiltedness(xs, valid):
    v = np.where(valid)[0]
    if len(v) < 3:
        return 0.0
    return float(np.abs(np.diff(xs[v])).mean())


def main() -> int:
    print("=" * 70)
    print("VERIFICACIÓN LaneIoU vs LineIoU (Paso 4.1)")
    print("=" * 70)
    lanes = real_lanes(15)

    # --- Panel A: el carril MÁS inclinado, con sus bandas ---
    xs, valid = max(lanes, key=lambda lv: tiltedness(*lv))
    v = np.where(valid)[0]
    x_disp = xs[v] * DISP_W
    y_disp = ROW_YS[v]
    w_lane = LI._angle_halfwidth(_t(xs), LI.LANE_WIDTH, LI.IMG_W, LI.IMG_H)[0].numpy()[v] * DISP_W
    w_line = np.full_like(w_lane, LI.LINE_WIDTH * DISP_W)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))
    axA.fill_betweenx(y_disp, x_disp - w_line, x_disp + w_line, color="tab:red", alpha=0.25,
                      label=f"LineIoU (constante, ±{LI.LINE_WIDTH*DISP_W:.0f}px)")
    axA.fill_betweenx(y_disp, x_disp - w_lane, x_disp + w_lane, color="tab:blue", alpha=0.35,
                      label="LaneIoU (sensible al ángulo)")
    axA.plot(x_disp, y_disp, "k-", lw=1.5, label="centro del carril")
    axA.invert_yaxis()
    axA.set_title("A) Banda virtual sobre un carril inclinado")
    axA.set_xlabel("x (px, aspecto original)")
    axA.set_ylabel("y (px)")
    axA.legend(loc="upper right", fontsize=8)
    axA.set_aspect("equal", adjustable="datalim")

    # --- Panel B: dispersión IoU de entrenamiento vs IoU de máscara (métrica) ---
    masks, l_lane, l_line = [], [], []
    for xs_i, valid_i in lanes:
        for delta in np.linspace(0.004, 0.055, 10):
            pred = (xs_i + delta).astype(np.float32)
            masks.append(mask_iou(pred, xs_i, valid_i))
            l_lane.append(LI.lane_iou_value(_t(pred), _t(xs_i), _t(valid_i), angle_aware=True).item())
            l_line.append(LI.lane_iou_value(_t(pred), _t(xs_i), _t(valid_i), angle_aware=False).item())
    masks, l_lane, l_line = map(np.array, (masks, l_lane, l_line))
    mae_lane = np.mean(np.abs(l_lane - masks))
    mae_line = np.mean(np.abs(l_line - masks))

    axB.plot([0, 1], [0, 1], "k--", lw=1, label="ideal (y=x)")
    axB.scatter(masks, l_line, s=14, c="tab:red", alpha=0.6, label=f"LineIoU (MAE={mae_line:.3f})")
    axB.scatter(masks, l_lane, s=14, c="tab:blue", alpha=0.6, label=f"LaneIoU (MAE={mae_lane:.3f})")
    axB.axvline(0.5, color="gray", ls=":", lw=1)
    axB.set_xlim(0, 1)
    axB.set_ylim(min(0, l_line.min()), 1.02)
    axB.set_title("B) IoU de entrenamiento vs IoU de máscara (métrica)")
    axB.set_xlabel("IoU de máscara real (la métrica F1)")
    axB.set_ylabel("IoU de entrenamiento")
    axB.legend(loc="lower right", fontsize=8)

    fig.suptitle("LaneIoU (sensible al ángulo) aproxima la métrica mejor que LineIoU", fontsize=12)
    fig.tight_layout()
    out = paths.outputs_dir() / "verify" / "lane_iou.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)

    print(f"  Carril del panel A: inclinación media |Δx|={tiltedness(xs, valid):.4f}")
    print(f"  MAE(LaneIoU vs métrica) = {mae_lane:.4f}")
    print(f"  MAE(LineIoU vs métrica) = {mae_line:.4f}   ({mae_line/mae_lane:.1f}× peor)")
    print(f"\nImagen guardada en: {out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
