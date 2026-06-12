"""PRUEBA SINTÉTICA del entrenamiento con visualizaciones (Paso 6.x).

Mini-entrenamiento (sobreajuste de las 3 imágenes FIJAS de CULane, 3 épocas) que produce el
mismo `work_dirs/<timestamp>/` que el entrenamiento real: config.yaml, train.log (todos los
loss + total + lr + ETA + época), eval.log, gpu.log, y por época las 4 visualizaciones
(gt_vs_pred, attention, anchors, matcher). Sirve para ver qué saldrá sin entrenar de verdad.

Uso:
    .\.venv\Scripts\python.exe tools\verify_train_viz.py
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

import torch

from lanetr.config import load_config
from lanetr.data import culane_annotation as ann
from lanetr.losses import HungarianMatcher, LaneCriterion
from lanetr.metrics import culane as M
from lanetr.metrics import format as F2
from lanetr.models import LaneTR
from lanetr.models.head import decode_lanes
from lanetr.training import (
    ETA, EpochVisualizer, GPUMonitor, ModelEMA, build_optimizer, create_work_dir,
    freeze_batchnorm, get_logger,
)

EPOCHS, STEPS, LR = 3, 60, 5e-4


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(None, {"name": "PREVIEW_sintetico",
                             "model": {"pretrained": True, "num_layers": 6,
                                       "use_anchors": True, "deformable": True},
                             "train": {"epochs": EPOCHS, "eval_conf_thresh": 0.4}})
    work = create_work_dir(cfg)
    train_log = get_logger("prev.train", work / "train.log")
    eval_log = get_logger("prev.eval", work / "eval.log")
    gpu_log = get_logger("prev.gpu", work / "gpu.log")
    print("=" * 70)
    print(f"PRUEBA SINTÉTICA — work_dir: {work}")
    print("=" * 70)

    model = LaneTR("dla34", pretrained=True, num_queries=12, num_layers=6,
                   use_anchors=True, deformable=True).to(device)
    freeze_batchnorm(model.backbone)
    model.train(); model.backbone.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    viz = EpochVisualizer(device, conf_thresh=0.4)  # 3 imágenes fijas de CULane
    images = viz.images.to(device)
    targets = [{
        "xs": torch.as_tensor(s["gt"]["xs"], device=device),
        "valid": torch.as_tensor(s["gt"]["valid"], device=device),
        "start_y": torch.as_tensor(s["gt"]["start"][:, 1], device=device),
        "length": torch.as_tensor(s["gt"]["length"], device=device),
        "theta": torch.as_tensor(s["gt"]["theta"], device=device),
    } for s in viz.samples]

    criterion = LaneCriterion(matcher=HungarianMatcher(w_cls=2, w_iou=4, w_xy=1.0, w_ext=0.5),
                              w_cls=2, w_iou=4, w_xy=1.0, w_ext=0.5, focal_alpha=0.5)
    optimizer = build_optimizer(model, lr=LR, weight_decay=1e-4)
    ema = ModelEMA(model, 0.999, tau=200)
    eta = ETA(EPOCHS * STEPS)
    gpu = GPUMonitor(device)

    train_log.info(f"work_dir: {work} | device={device} | 3 imágenes fijas, {EPOCHS} épocas × {STEPS} pasos")
    it = 0
    for epoch in range(EPOCHS):
        gpu.epoch_start()
        t0 = time.time()
        for step in range(STEPS):
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                losses = criterion(model(images), targets)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            ema.update(model)
            it += 1
            eta_str = eta.step(it)
            if it % 10 == 0:
                gpu.sample()
                loss_str = "  ".join(f"loss_{k} {float(v.detach()):.4f}" for k, v in losses.items())
                lr = optimizer.param_groups[0]["lr"]
                train_log.info(f"ep {epoch+1}/{EPOCHS} it {it}/{EPOCHS*STEPS}  lr {lr:.6f}  "
                               f"{loss_str}  ETA {eta_str}")

        # GPU por época
        g = gpu.epoch_summary()
        um = f"{g['util_mean']:.0f}" if g.get("util_mean") is not None else "n/a"
        gpu_log.info(f"época {epoch+1}: mem_pico {g.get('mem_peak_gb', 0):.2f}GB  "
                     f"reservada {g.get('mem_reserved_gb', 0):.2f}GB  util media {um}%  "
                     f"máx {g.get('util_max', 0)}%")
        # F1 sobre las 3 imágenes con la EMA
        em = ema.ema
        em.eval()
        with torch.no_grad():
            lanes = decode_lanes(em(images), conf_thresh=0.4, num_rows=144)
        preds = [[F2.resized_to_orig(l["points"]) for l in lanes[i]] for i in range(len(viz.samples))]
        annos = [M.load_culane_img_data(str(ann.lines_path_for_image(s["rel"]))) for s in viz.samples]
        res = M.evaluate(preds, annos)[0.5]
        eval_log.info(f"época {epoch+1}: F1={res['F1']:.4f}  P={res['Precision']:.4f}  "
                      f"R={res['Recall']:.4f}  TP={res['TP']} FP={res['FP']} FN={res['FN']}")
        model.train(); model.backbone.eval()

        # visualizaciones de la época
        viz.visualize(ema.ema, criterion.matcher, epoch, work / "viz")
        train_log.info(f"época {epoch+1}/{EPOCHS} en {time.time()-t0:.1f}s -> viz/epoch_{epoch:03d}/")

    # árbol del work_dir
    print("\nContenido de", work.name + "/:")
    for p in sorted(work.rglob("*")):
        if p.is_file():
            print("   ", p.relative_to(work).as_posix())
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
