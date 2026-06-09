"""Verificación VISUAL de la asignación auxiliar uno-a-muchos (Paso 5.2).

Sobreajusta el mismo batch DOS veces (anclas + matching dinámico en ambos):
  - SIN uno-a-muchos: uno-a-uno en todas las capas.
  - CON uno-a-muchos: uno-a-muchos en capas tempranas, uno-a-uno solo en la última.
Compara las curvas de pérdida (la de uno-a-muchos debe converger más rápido y suave) y dibuja
los carriles finales del modelo con uno-a-muchos.

Salida: outputs/verify/o2m.png

Uso:
    .\.venv\Scripts\python.exe tools\verify_o2m.py
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
import numpy as np
import torch

from lanetr import paths
from lanetr.data import target_encoding as TE
from lanetr.data import transforms as T
from lanetr.data.culane_dataset import CULaneDataset
from lanetr.losses import HungarianMatcher, LaneCriterion, prepare_targets
from lanetr.models import LaneTR
from lanetr.models.head import decode_lanes

K, STEPS, LR, R = 3, 220, 5e-4, 144
SEEDS = [0, 1, 2]
ROW_YS = TE.make_row_ys(320, R)


def colors(n):
    return [tuple(colorsys.hsv_to_rgb(i / max(n, 1), 0.9, 1.0)) for i in range(n)]


def train(images, targets, use_o2m, device, seed=0):
    torch.manual_seed(seed)
    model = LaneTR("dla34", pretrained=True, num_queries=12, num_layers=6, num_rows=R,
                   use_anchors=True).to(device)
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    crit = LaneCriterion(matcher=HungarianMatcher(w_cls=2.0, w_iou=2.0, w_xy=1.0, w_ext=0.5),
                         w_xy=1.0, focal_alpha=0.5, aux_one_to_many=use_o2m, o2m_k=4)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
    model.train()
    model.backbone.eval()
    hist = []
    for _ in range(STEPS):
        loss = crit(model(images), targets)
        opt.zero_grad()
        loss["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        hist.append(loss["total"].item())
    model.eval()
    with torch.no_grad():
        lanes = decode_lanes(model(images), conf_thresh=0.5, num_rows=R)
    return hist, lanes


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(f"VERIFICACIÓN UNO-A-MUCHOS (Paso 5.2) — device={device}, DLA-34 + anclas")
    print("=" * 70)

    ds = CULaneDataset("val", augment=False, encode_targets=True, num_rows=R)
    picks, i = [], 0
    while len(picks) < K and i < len(ds):
        if ds[i]["existence"] and sum(ds[i]["existence"]) >= 3:
            picks.append(i)
        i += 1
    samples = [ds[p] for p in picks]
    images = torch.stack([s["image"] for s in samples]).to(device)
    targets = prepare_targets([s["targets"] for s in samples], device)
    rgbs = [T.denormalize(s["image"]) for s in samples]

    lanes = None
    runs = {"1-a-1 (todas las capas)": [], "uno-a-muchos (capas tempranas)": []}
    for use_o2m, key in [(False, "1-a-1 (todas las capas)"), (True, "uno-a-muchos (capas tempranas)")]:
        for seed in SEEDS:
            print(f"  entrenando {key}  seed={seed} ...")
            hist, lns = train(images, targets, use_o2m, device, seed)
            runs[key].append(hist)
            if use_o2m and seed == SEEDS[0]:
                lanes = lns

    def summary(hs):
        arr = np.array(hs)                       # (n_seeds, STEPS)
        finals = arr[:, -1]
        return arr.mean(0), arr.min(0), arr.max(0), finals.mean(), finals.std()

    for key in runs:
        _, _, _, fm, fs = summary(runs[key])
        print(f"  pérdida final {key}: {fm:.2f} ± {fs:.2f}  (n={len(SEEDS)} semillas)")

    fig = plt.figure(figsize=(12, 3.0 + 2.3 * K))
    gs = fig.add_gridspec(K + 1, 1, height_ratios=[1.5] + [1] * K)
    ax = fig.add_subplot(gs[0])
    for key, c in [("1-a-1 (todas las capas)", "tab:red"),
                   ("uno-a-muchos (capas tempranas)", "tab:green")]:
        mean, lo, hi, fm, fs = summary(runs[key])
        ax.plot(mean, color=c, lw=1.6, label=f"{key}: {fm:.2f} ± {fs:.2f}")
        ax.fill_between(range(STEPS), lo, hi, color=c, alpha=0.18)
    ax.set_yscale("log")
    ax.set_xlabel("paso")
    ax.set_ylabel("pérdida (log)")
    ax.set_title(f"Media ± rango sobre {len(SEEDS)} semillas (banda = varianza entre ejecuciones). "
                 "Última capa 1-a-1 → sin NMS")
    ax.legend()
    ax.grid(alpha=0.3)

    for r in range(K):
        axi = fig.add_subplot(gs[r + 1])
        axi.imshow(rgbs[r])
        axi.set_title(f"img {picks[r]} — carriles finales (con uno-a-muchos)", fontsize=9)
        axi.axis("off")
        tg = targets[r]
        for g in range(tg["xs"].shape[0]):
            pts = TE.decode_lane(tg["xs"][g].cpu().numpy(), tg["valid"][g].cpu().numpy(), ROW_YS, 800)
            if len(pts) >= 2:
                axi.plot(pts[:, 0], pts[:, 1], color="white", lw=4, alpha=0.85)
        pal = colors(len(lanes[r]))
        for j, lane in enumerate(lanes[r]):
            p = lane["points"]
            if len(p) >= 2:
                axi.plot(p[:, 0], p[:, 1], color=pal[j], lw=2)

    fig.tight_layout()
    out = paths.outputs_dir() / "verify" / "o2m.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nImagen guardada en: {out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
