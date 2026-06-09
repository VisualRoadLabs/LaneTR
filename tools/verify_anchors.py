"""Verificación VISUAL del prior posicional / anclas (Paso 5.1).

Compara, sobre una imagen real y con el modelo SIN ENTRENAR, las predicciones iniciales:
  - IZQUIERDA: SIN prior posicional → las 12 queries arrancan casi idénticas (centradas) →
    el matcher duda → matching inestable (lo que vimos en el Paso 4.4).
  - DERECHA:  CON prior posicional → cada query nace con un ancla distinta (abanico) →
    predicciones repartidas desde el primer paso → el matching se estabiliza.

Salida: outputs/verify/anchors.png

Uso:
    .\.venv\Scripts\python.exe tools\verify_anchors.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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

import colorsys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from lanetr import paths
from lanetr.data import transforms as T
from lanetr.data.culane_dataset import CULaneDataset
from lanetr.models import LaneTR
from lanetr.models.head import decode_lanes

NQ, R = 12, 144


def colors(n):
    return [tuple(colorsys.hsv_to_rgb(i / n, 0.9, 1.0)) for i in range(n)]


def draw(ax, rgb, lanes, title):
    ax.imshow(rgb)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    pal = colors(NQ)
    for lane in lanes:
        p = lane["points"]
        if len(p) >= 2:
            ax.plot(p[:, 0], p[:, 1], color=pal[lane["query"]], lw=2)


def main() -> int:
    print("=" * 70)
    print("VERIFICACIÓN DEL PRIOR POSICIONAL / ANCLAS (Paso 5.1) — SIN ENTRENAR")
    print("=" * 70)
    torch.manual_seed(0)

    ds = CULaneDataset("val", augment=False)
    s = ds[0]
    rgb = T.denormalize(s["image"])
    x = s["image"].unsqueeze(0)

    model_plain = LaneTR("dla34", pretrained=True, num_queries=NQ, num_layers=6).eval()
    torch.manual_seed(0)
    model_anchor = LaneTR("dla34", pretrained=True, num_queries=NQ, num_layers=6,
                          use_anchors=True).eval()
    with torch.no_grad():
        lanes_plain = decode_lanes(model_plain(x), conf_thresh=None, num_rows=R)[0]
        lanes_anchor = decode_lanes(model_anchor(x), conf_thresh=None, num_rows=R)[0]

    # dispersión de las posiciones de inicio (abajo) como medida de "reparto"
    def spread(lanes):
        import numpy as np
        xs0 = [l["points"][-1, 0] for l in lanes if len(l["points"])]
        return float(np.std(xs0)) if xs0 else 0.0
    print(f"Dispersión x (abajo) — SIN anclas: {spread(lanes_plain):.1f}px  |  "
          f"CON anclas: {spread(lanes_anchor):.1f}px")

    fig, ax = plt.subplots(1, 2, figsize=(13, 3.4))
    draw(ax[0], rgb, lanes_plain, "SIN prior — 12 queries casi idénticas (centradas)")
    draw(ax[1], rgb, lanes_anchor, "CON prior — anclas en abanico → repartidas")
    fig.suptitle("Prior posicional: cada query nace con un ancla distinta → predicciones "
                 "repartidas desde el inicio (estabiliza el matching)", fontsize=12)
    fig.tight_layout()
    out = paths.outputs_dir() / "verify" / "anchors.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nImagen guardada en: {out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
