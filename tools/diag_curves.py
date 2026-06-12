"""Diagnóstico: ¿el modelo entrenado predice curvas o deja las líneas rectas?

Carga el checkpoint de una ablation, lo corre sobre imágenes de CURVA reales
(test_split/test6_curve.txt), dibuja GT (blanco) vs predicción (color) y mide cuánto
se curva cada carril (desviación máxima de x respecto a la recta que une sus extremos).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from lanetr import paths
from lanetr.data import transforms as T
from lanetr.data.culane_annotation import lines_path_for_image
from lanetr.metrics import culane as M
from lanetr.models.head import decode_lanes
from train import build_model

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "work_dirs/abl_main_20260609_162152/checkpoints/best.pth"
N = 6


def curvature(pts):
    """Desviación máxima (px) de x respecto a la recta que une el primer y último punto."""
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    # x esperado si fuera recta entre extremos (en función de y)
    if abs(y[-1] - y[0]) < 1e-6:
        return 0.0
    x_line = x[0] + (x[-1] - x[0]) * (y - y[0]) / (y[-1] - y[0])
    return float(np.max(np.abs(x - x_line)))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg, device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model.eval()
    iw, ih = cfg["data"]["img_w"], cfg["data"]["img_h"]
    nr = cfg["data"]["num_rows"]
    thr = cfg["train"].get("eval_conf_thresh", 0.4)
    print(f"checkpoint: {CKPT}  (época {ckpt.get('epoch','?')}, pesos {'EMA' if 'ema' in ckpt else 'online'})")
    print(f"img {iw}x{ih}, num_rows {nr}, conf_thresh {thr}, deformable {cfg['model'].get('deformable')}")

    lines = [l.strip() for l in (paths.list_dir() / "test_split" / "test6_curve.txt")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    picks = lines[:N]

    cr = T.CropResize(iw, ih, 270)
    norm = T.Normalize()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(16, 6))
    gt_curv_all, pred_curv_all = [], []

    for ax, rel in zip(axes.flat, picks):
        rel = rel.lstrip("/\\")
        pil = Image.open(paths.image_path(rel)).convert("RGB")
        # GT en 1640x590 -> 800x320 (recorte y>=270, resize)
        gt_raw = M.load_culane_img_data(str(lines_path_for_image(rel)))
        s = {"image": pil, "lanes": [g.astype(np.float32).copy() for g in gt_raw],
             "slots": [None] * len(gt_raw), "existence": None, "meta": {}}
        s = cr(s, rng)
        rgb = np.asarray(s["image"], dtype=np.uint8)
        gt_lanes = s["lanes"]
        t = norm(s, rng)["image"]
        with torch.no_grad():
            pred = model(t.unsqueeze(0).to(device))
        lanes = decode_lanes(pred, conf_thresh=thr, num_rows=nr, img_w=iw, img_h=ih)[0]

        ax.imshow(rgb); ax.axis("off")
        gc = [curvature(g) for g in gt_lanes if len(g) >= 3]
        pc = [curvature(l["points"]) for l in lanes if len(l["points"]) >= 3]
        gt_curv_all += gc; pred_curv_all += pc
        for g in gt_lanes:
            if len(g) >= 2:
                ax.plot(g[:, 0], g[:, 1], color="white", lw=5, alpha=0.8)
        for l in lanes:
            p = l["points"]
            if len(p) >= 2:
                ax.plot(p[:, 0], p[:, 1], lw=2.5)
        ax.set_title(f"GT curv max={max(gc) if gc else 0:.0f}px | "
                     f"Pred curv max={max(pc) if pc else 0:.0f}px", fontsize=9)

    fig.suptitle(f"main (LaneIoU, época {ckpt.get('epoch','?')}) sobre 6 curvas reales — "
                 f"GT (blanco) vs predicción (color)", fontsize=12)
    fig.tight_layout()
    out = Path("outputs") / "diag_curves.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nfigura -> {out}")
    print(f"\nCurvatura (desviación de la recta, px):")
    print(f"  GT  : media {np.mean(gt_curv_all):.1f}  máx {np.max(gt_curv_all):.1f}  (n={len(gt_curv_all)})")
    print(f"  Pred: media {np.mean(pred_curv_all):.1f}  máx {np.max(pred_curv_all):.1f}  (n={len(pred_curv_all)})")


if __name__ == "__main__":
    main()
