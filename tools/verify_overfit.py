"""Verificación: SOBREAJUSTE a un batch pequeño (Paso 4.4) — el momento de la verdad.

Entrena el modelo completo LaneTR (DLA-34) sobre unas pocas imágenes durante varios cientos
de pasos y comprueba que toda la cadena (modelo + matcher + LaneIoU + criterion + backprop)
funciona: la pérdida BAJA y las predicciones DEJAN de agruparse en el centro y se pegan a los
carriles reales.

Salida: outputs/verify/overfit.png  (curva de pérdida + antes/después por imagen)

Uso:
    .\.venv\Scripts\python.exe tools\verify_overfit.py
"""
from __future__ import annotations

import os
import sys
import time
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

K = 3            # nº de imágenes a sobreajustar
STEPS = 600
LR = 5e-4
R, IMG_W, IMG_H = 144, 800, 320
ROW_YS = TE.make_row_ys(IMG_H, R)

# Configurable por variables de entorno:
#   OVERFIT_ANCHORS=1  -> usa prior posicional (Paso 5.1)
#   OVERFIT_DYNAMIC=1  -> usa matching húngaro DINÁMICO (sin la muleta de asignación fija)
#   OVERFIT_DEFORM=1   -> usa atención deformable (Paso 5.3)
USE_ANCHORS = os.environ.get("OVERFIT_ANCHORS", "0") == "1"
DYNAMIC = os.environ.get("OVERFIT_DYNAMIC", "0") == "1"
USE_DEFORM = os.environ.get("OVERFIT_DEFORM", "0") == "1"
OUT_NAME = "overfit_anchors.png" if (USE_ANCHORS or DYNAMIC or USE_DEFORM) else "overfit.png"


class FixedMatcher:
    """Asignación FIJA query g -> carril GT g (las primeras G queries).

    El matching húngaro DINÁMICO con queries casi idénticas al inicio es inestable
    (inestabilidad típica de DETR; se estabiliza en el Paso 5 con prior posicional +
    asignación auxiliar uno-a-muchos). Para este sanity check fijamos la asignación y así
    comprobamos que el modelo + LaneIoU + backprop SÍ aprenden la geometría.
    """

    @torch.no_grad()
    def match(self, pred, targets):
        dev = pred["conf"].device
        out = []
        for b in range(len(targets)):
            G = targets[b]["xs"].shape[0]
            idx = torch.arange(G, dtype=torch.long, device=dev)
            out.append((idx, idx))
        return out


def colors(n):
    return [tuple(colorsys.hsv_to_rgb(i / max(n, 1), 0.9, 1.0)) for i in range(n)]


def draw(ax, rgb, gt_targets, pred_lanes, title):
    ax.imshow(rgb)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    for g in range(gt_targets["xs"].shape[0]):  # GT en blanco
        pts = TE.decode_lane(gt_targets["xs"][g], gt_targets["valid"][g], ROW_YS, IMG_W)
        if len(pts) >= 2:
            ax.plot(pts[:, 0], pts[:, 1], color="white", lw=4, alpha=0.85)
    pal = colors(len(pred_lanes))
    for i, lane in enumerate(pred_lanes):  # predicciones en color
        p = lane["points"]
        if len(p) >= 2:
            ax.plot(p[:, 0], p[:, 1], color=pal[i], lw=2)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = (f"anclas={'sí' if USE_ANCHORS else 'no'}, "
            f"matching={'DINÁMICO' if DYNAMIC else 'fijo'}, "
            f"deformable={'sí' if USE_DEFORM else 'no'}")
    print("=" * 70)
    print(f"SOBREAJUSTE A UN BATCH — device={device}, DLA-34  [{mode}]")
    print("=" * 70)
    torch.manual_seed(0)

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
    print(f"Imágenes: {picks}  (carriles: {[sum(s['existence']) for s in samples]})")

    model = LaneTR("dla34", pretrained=True, num_queries=12, num_layers=6, num_rows=R,
                   use_anchors=USE_ANCHORS, deformable=USE_DEFORM).to(device)
    # Para el sobreajuste de demostración: congelar el backbone (BN fijo, sin desajuste
    # train/eval con batch pequeño) y quitar el dropout (memorización limpia).
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params entrenables: {n_train/1e6:.1f}M (backbone DLA-34 congelado, sin dropout)")

    matcher = HungarianMatcher(w_cls=2.0, w_iou=2.0, w_xy=1.0, w_ext=0.5) if DYNAMIC else FixedMatcher()
    crit = LaneCriterion(matcher=matcher, w_xy=1.0, focal_alpha=0.5)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=1e-4)

    # --- snapshot ANTES ---
    model.eval()
    with torch.no_grad():
        before = decode_lanes(model(images), conf_thresh=None, num_rows=R)

    # --- bucle de sobreajuste ---
    model.train()
    model.backbone.eval()  # mantener BN del backbone en modo eval (congelado)
    hist = {"total": [], "cls": [], "iou": [], "ext": []}
    t0 = time.time()
    for step in range(STEPS):
        losses = crit(model(images), targets)
        opt.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        for k in hist:
            hist[k].append(losses[k].item())
        if step % 50 == 0 or step == STEPS - 1:
            print(f"  step {step:4d}  total={losses['total']:.3f}  cls={losses['cls']:.3f}  "
                  f"iou={losses['iou']:.3f}  ext={losses['ext']:.3f}")
    print(f"  ({time.time()-t0:.1f}s para {STEPS} pasos)")

    # --- snapshot DESPUÉS ---
    model.eval()
    with torch.no_grad():
        after = decode_lanes(model(images), conf_thresh=0.5, num_rows=R)
    print(f"Carriles confiables (conf>0.5) por imagen: {[len(a) for a in after]} "
          f"(GT: {[sum(s['existence']) for s in samples]})")

    # --- figura ---
    fig = plt.figure(figsize=(11, 3.2 + 2.3 * K))
    gs = fig.add_gridspec(K + 1, 2, height_ratios=[1.3] + [1] * K)
    ax_loss = fig.add_subplot(gs[0, :])
    for key, c in [("total", "k"), ("cls", "tab:green"), ("iou", "tab:blue"), ("ext", "tab:orange")]:
        ax_loss.plot(hist[key], color=c, label=key, lw=1.5 if key == "total" else 1)
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("paso")
    ax_loss.set_ylabel("pérdida (log)")
    ax_loss.set_title(f"Pérdida durante el sobreajuste ({STEPS} pasos): "
                      f"{hist['total'][0]:.1f} → {hist['total'][-1]:.2f}")
    ax_loss.legend(ncol=4, fontsize=8)
    ax_loss.grid(alpha=0.3)

    cpu_targets = [{"xs": t["xs"].cpu().numpy(), "valid": t["valid"].cpu().numpy()} for t in targets]
    for r in range(K):
        ax_b = fig.add_subplot(gs[r + 1, 0])
        ax_a = fig.add_subplot(gs[r + 1, 1])
        draw(ax_b, rgbs[r], cpu_targets[r], before[r], f"img {picks[r]} — ANTES (12 candidatos)")
        draw(ax_a, rgbs[r], cpu_targets[r], after[r], f"img {picks[r]} — DESPUÉS (conf>0.5)")

    fig.suptitle(f"Sobreajuste [{mode}]: la pérdida baja y las predicciones se pegan a los "
                 f"carriles reales", fontsize=12)
    fig.tight_layout()
    out = paths.outputs_dir() / "verify" / OUT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nImagen guardada en: {out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
