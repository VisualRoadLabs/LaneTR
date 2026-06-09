"""Verificación VISUAL del matcher húngaro (Paso 4.2).

Sobre una imagen real de 4 carriles:
  - construye predicciones sintéticas = GT (ligeramente perturbado) colocado en queries
    concretas, + queries "vacías" de relleno + un distractor;
  - calcula la matriz de coste DESCOMPUESTA (cls / LaneIoU / xy / ext) y el total;
  - resuelve el emparejamiento húngaro y lo dibuja.

Paneles:
  (arriba) imagen con los carriles GT (blanco) y las predicciones EMPAREJADAS (color);
  (abajo)  5 mapas de calor (NQ×G): total + las 4 componentes, con la asignación marcada.

Salida: outputs/verify/matcher.png

Uso:
    .\.venv\Scripts\python.exe tools\verify_matcher.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import colorsys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from lanetr import paths
from lanetr.data import target_encoding as TE
from lanetr.data import transforms as T
from lanetr.data.culane_dataset import CULaneDataset
from lanetr.losses import HungarianMatcher

NQ, R, IMG_W, IMG_H = 12, 144, 800, 320
ROW_YS = TE.make_row_ys(IMG_H, R)


def colors(n):
    return [tuple(colorsys.hsv_to_rgb(i / n, 0.9, 1.0)) for i in range(n)]


def main() -> int:
    print("=" * 70)
    print("VERIFICACIÓN DEL MATCHER HÚNGARO (Paso 4.2)")
    print("=" * 70)

    ds = CULaneDataset("val", augment=False, encode_targets=True, num_rows=R)
    s = None
    for idx in range(len(ds)):
        cand = ds[idx]
        if cand["existence"] == (1, 1, 1, 1):
            s = cand
            break
    tg = s["targets"]
    G = tg["xs"].shape[0]
    tgt = {"xs": torch.tensor(tg["xs"]), "valid": torch.tensor(tg["valid"]),
           "start_y": torch.tensor(tg["start"][:, 1]), "length": torch.tensor(tg["length"])}
    print(f"Imagen con {G} carriles GT.")

    # --- predicciones sintéticas ---
    rng = np.random.default_rng(0)
    query_for_gt = [7, 1, 10, 4][:G]
    pred_xs = torch.tensor(rng.uniform(0.45, 0.55, size=(NQ, R)).astype("float32"))
    for g, q in enumerate(query_for_gt):
        noise = torch.tensor(rng.normal(0, 0.004, size=R).astype("float32"))
        pred_xs[q] = tgt["xs"][g] + noise            # casi-GT en su query
    pred_xs[6] = tgt["xs"][0] + 0.05                  # distractor cerca del GT 0
    conf = torch.full((NQ,), -3.0)
    conf[query_for_gt] = 3.0
    conf[6] = 1.0
    pred = {"conf": conf, "xs": pred_xs,
            "start_y": torch.ones(NQ) * float(tgt["start_y"].mean()),
            "length": torch.ones(NQ) * float(tgt["length"].mean())}

    matcher = HungarianMatcher(w_cls=2.0, w_iou=2.0, w_xy=0.0, w_ext=0.5)
    comps = matcher.cost_components(pred, tgt)
    q_idx, g_idx = matcher.match_one(pred, tgt)
    print("Emparejamiento (gt -> query):", {int(gi): int(qi) for qi, gi in zip(q_idx, g_idx)})
    print("Esperado (gt -> query):       ", {g: q for g, q in enumerate(query_for_gt)})

    # --- figura ---
    rgb = T.denormalize(s["image"])
    fig = plt.figure(figsize=(14, 7.5))
    grid = fig.add_gridspec(2, 5, height_ratios=[1.5, 1.0])
    ax_img = fig.add_subplot(grid[0, :])
    ax_img.imshow(rgb)
    ax_img.set_title("Carriles GT (blanco) y predicciones EMPAREJADAS (color)")
    ax_img.axis("off")
    pal = colors(G)
    for qi, gi in zip(q_idx.tolist(), g_idx.tolist()):
        gpts = TE.decode_lane(tg["xs"][gi], tg["valid"][gi], ROW_YS, IMG_W)
        ppts = TE.decode_lane(pred_xs[qi].numpy(), tg["valid"][gi], ROW_YS, IMG_W)
        ax_img.plot(gpts[:, 0], gpts[:, 1], color="white", lw=4, alpha=0.9)
        ax_img.plot(ppts[:, 0], ppts[:, 1], color=pal[gi], lw=2)
        ax_img.text(ppts[0, 0], ppts[0, 1] + 8, f"q{qi}→gt{gi}", color=pal[gi], fontsize=9)

    titles = {"total": "COSTE TOTAL", "cls": "cls (focal)", "iou": "1 − LaneIoU",
              "xy": "L1 xs", "ext": "L1 extensión"}
    for i, key in enumerate(["total", "cls", "iou", "xy", "ext"]):
        ax = fig.add_subplot(grid[1, i])
        M = comps[key].detach().numpy()
        im = ax.imshow(M, aspect="auto", cmap="viridis")
        ax.set_title(titles[key], fontsize=10)
        ax.set_xlabel("GT")
        ax.set_xticks(range(G))
        if i == 0:
            ax.set_ylabel("query")
        for qi, gi in zip(q_idx.tolist(), g_idx.tolist()):
            ax.add_patch(Rectangle((gi - 0.5, qi - 0.5), 1, 1, fill=False,
                                   edgecolor="red", lw=2))
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Matcher húngaro: coste = cls + (1−LaneIoU) + xy + ext  →  asignación 1-a-1 "
                 "(rojo) — sin NMS", fontsize=12)
    fig.tight_layout()
    out = paths.outputs_dir() / "verify" / "matcher.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nImagen guardada en: {out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
