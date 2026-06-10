"""Verificación VISUAL de la atención deformable (Paso 5.3).

Muestra, para varias queries, los PUNTOS DE MUESTREO de la cross-attention deformable sobre
una imagen real: cada query mira solo unos pocos puntos (n_heads × n_levels × n_points)
alrededor de su punto de referencia (el ancla), en vez de los ~5250 tokens de la atención
densa. SIN ENTRENAR: demuestra el mecanismo (muestreo disperso), no la detección.

Salida: outputs/verify/deformable.png

Uso:
    .\.venv\Scripts\python.exe tools\verify_deformable.py
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

IMG_W, IMG_H = 800, 320
SHOW = [0, 4, 7, 11]   # queries a dibujar


def main() -> int:
    print("=" * 70)
    print("VERIFICACIÓN DE LA ATENCIÓN DEFORMABLE (Paso 5.3) — SIN ENTRENAR")
    print("=" * 70)
    torch.manual_seed(0)

    model = LaneTR("dla34", pretrained=True, num_queries=12, num_layers=6,
                   use_anchors=True, deformable=True, n_ref_points=6).eval()
    n_pts = model.decoder.layers[0].cross_attn.n_points
    n_heads = model.decoder.layers[0].cross_attn.n_heads
    n_ref = model.n_ref_points
    print(f"Deformable: {n_heads} cabezas × 3 niveles × {n_ref} refs × {n_pts} puntos = "
          f"{n_heads*3*n_ref*n_pts} muestras/query  (vs 5250 tokens de la atención densa)")
    print(f"Los {n_ref} puntos de referencia (★) se reparten A LO LARGO del carril (Paso 7).")

    ds = CULaneDataset("val", augment=False)
    s = ds[0]
    rgb = T.denormalize(s["image"])
    x = s["image"].unsqueeze(0)
    with torch.no_grad():
        feats = model.fpn(model.backbone(x))
        ref = model.anchors.reference_points_multi(model.n_ref_points)   # (NQ, n_ref, 2)
        _, samps, shapes = model.decoder(feats, query_pos=model.anchors.pos_embed(),
                                         reference_points=ref, need_attn=True)
    sl = samps[-1][0][0]      # última capa, imagen 0: (NQ, n_heads, n_levels, n_ref*n_points, 2)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.imshow(rgb)
    ax.set_title("Atención deformable: cada query muestrea solo unos pocos puntos alrededor de "
                 "su ancla (★)  —  SIN ENTRENAR")
    ax.axis("off")
    pal = [colorsys.hsv_to_rgb(i / len(SHOW), 0.9, 1.0) for i in range(len(SHOW))]
    for c, q in enumerate(SHOW):
        pts = sl[q].reshape(-1, 2).cpu().numpy()          # (n_heads*n_levels*n_ref*n_points, 2)
        ax.scatter(pts[:, 0] * IMG_W, pts[:, 1] * IMG_H, s=12, color=pal[c], alpha=0.7,
                   label=f"query {q}")
        rp = ref[q].cpu().numpy()                          # (n_ref, 2): puntos de ref a lo largo del carril
        ax.scatter(rp[:, 0] * IMG_W, rp[:, 1] * IMG_H, marker="*", s=220, color=pal[c],
                   edgecolors="black", linewidths=0.8)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)
    fig.tight_layout()
    out = paths.outputs_dir() / "verify" / "deformable.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nImagen guardada en: {out}")
    print("★ = punto de referencia (ancla);  puntos = muestras de la cross-attention deformable.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
