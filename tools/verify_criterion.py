"""Verificación VISUAL del criterion (Paso 4.3).

Sobre una imagen real, calcula la pérdida y su desglose por término para tres escenarios:
  - PERFECTO  : predicciones = GT colocado en queries concretas (debería dar ~0 en geometría),
  - RUIDOSO   : GT + ruido,
  - SIN ENTRENAR: las predicciones reales del modelo LaneTR sin entrenar.
Muestra un gráfico de barras agrupadas: cada término (cls, LaneIoU, xy, ext) y el total crecen
a medida que la predicción empeora → confirma que la pérdida está bien orientada.

Salida: outputs/verify/criterion.png

Uso:
    .\.venv\Scripts\python.exe tools\verify_criterion.py
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

from lanetr import paths
from lanetr.data.culane_dataset import CULaneDataset
from lanetr.losses import LaneCriterion, prepare_targets
from lanetr.models import LaneTR

L, NQ, R = 6, 12, 144


def make_pred(tgt, noise, seed):
    """pred (L,1,NQ,...) que copia los GT (con ruido opcional) en queries 0..G-1."""
    g = torch.Generator().manual_seed(seed)
    G = tgt["xs"].shape[0]
    xs = torch.rand(L, 1, NQ, R, generator=g) * 0.1 + 0.45
    conf = torch.full((L, 1, NQ), -4.0)
    for gi in range(G):
        lane = tgt["xs"][gi].clone()
        if noise > 0:
            lane = lane + torch.randn(R, generator=g) * noise
        xs[:, 0, gi] = lane
        conf[:, 0, gi] = 4.0
    sy = torch.ones(L, 1, NQ) * float(tgt["start_y"].mean())
    ln = torch.ones(L, 1, NQ) * float(tgt["length"].mean())
    return {"conf": conf, "xs": xs, "start_y": sy, "length": ln, "theta": torch.zeros(L, 1, NQ)}


def main() -> int:
    print("=" * 70)
    print("VERIFICACIÓN DEL CRITERION (Paso 4.3)")
    print("=" * 70)

    ds = CULaneDataset("val", augment=False, encode_targets=True, num_rows=R)
    s = next(ds[i] for i in range(len(ds)) if ds[i]["existence"] == (1, 1, 1, 1))
    targets = prepare_targets([s["targets"]], "cpu")
    crit = LaneCriterion()

    out_perfect = crit(make_pred(targets[0], 0.0, 0), targets)
    out_noisy = crit(make_pred(targets[0], 0.05, 1), targets)
    torch.manual_seed(0)
    model = LaneTR(pretrained=False, num_queries=NQ, num_layers=L, num_rows=R).eval()
    with torch.no_grad():
        pred_model = model(s["image"].unsqueeze(0))
    out_model = crit(pred_model, targets)

    terms = ["cls", "iou", "xy", "ext", "total"]
    scen = {"perfecto": out_perfect, "ruidoso": out_noisy, "sin entrenar": out_model}
    for name, o in scen.items():
        print(f"  {name:14s}: " + "  ".join(f"{t}={o[t].item():.3f}" for t in terms))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(terms))
    w = 0.26
    colors = {"perfecto": "tab:green", "ruidoso": "tab:orange", "sin entrenar": "tab:red"}
    for i, (name, o) in enumerate(scen.items()):
        vals = [o[t].item() for t in terms]
        ax.bar(x + (i - 1) * w, vals, w, label=name, color=colors[name])
    ax.set_xticks(x)
    ax.set_xticklabels(["cls (focal)", "LaneIoU", "L1 xs", "L1 ext", "TOTAL"])
    ax.set_ylabel("valor de la pérdida")
    ax.set_title("Desglose de la pérdida: peor predicción → mayor pérdida (en cada término)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = paths.outputs_dir() / "verify" / "criterion.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nImagen guardada en: {out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
