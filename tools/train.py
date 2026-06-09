"""Entrenamiento de LaneTR en CULane (Paso 6.2).

Bucle por épocas con AMP bf16 + gradient clipping + EMA + checkpoints + logging. Está pensado
para lanzarse en el Ubuntu/A6000, pero corre también aquí (RTX 4060) y tiene un modo `--smoke`
de pocas iteraciones para validar que todo encaja.

Uso:
    python tools/train.py --config configs/lanetr_culane.yaml
    python tools/train.py --config configs/lanetr_culane.yaml --smoke   # validación rápida
    python tools/train.py --set loss.aux_one_to_many=true model.deformable=false  # ablations
"""
from __future__ import annotations

import argparse
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

from lanetr import paths
from lanetr.config import load_config
from lanetr.data.culane_dataset import build_dataloader
from lanetr.losses import LaneCriterion, prepare_targets
from lanetr.models import LaneTR
from lanetr.training import ModelEMA, build_optimizer, build_scheduler, freeze_batchnorm


def build_model(cfg, device):
    m = cfg["model"]
    model = LaneTR(backbone=m["backbone"], pretrained=m["pretrained"], d_model=m["d_model"],
                   num_queries=m["num_queries"], num_layers=m["num_layers"],
                   num_rows=m["num_rows"], img_h=cfg["data"]["img_h"],
                   use_anchors=m["use_anchors"], deformable=m["deformable"],
                   n_points=m["n_points"])
    if cfg["train"]["freeze_bn"]:
        freeze_batchnorm(model.backbone)
    model = model.to(device)
    if cfg["train"]["channels_last"] and device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


def build_criterion(cfg):
    l = cfg["loss"]
    return LaneCriterion(w_cls=l["w_cls"], w_iou=l["w_iou"], w_xy=l["w_xy"], w_ext=l["w_ext"],
                         w_smooth=l["w_smooth"], focal_alpha=l["focal_alpha"],
                         focal_gamma=l["focal_gamma"], aux_one_to_many=l["aux_one_to_many"],
                         o2m_k=l["o2m_k"], img_h=cfg["data"]["img_h"])


def train(cfg, smoke=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["train"]["seed"])
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    bs = 2 if smoke else cfg["data"]["batch_size"]
    workers = 0 if smoke else cfg["data"]["num_workers"]
    dl = build_dataloader("train", batch_size=bs, shuffle=True, num_workers=workers,
                          seed=cfg["train"]["seed"], encode_targets=True,
                          num_rows=cfg["data"]["num_rows"], img_w=cfg["data"]["img_w"],
                          img_h=cfg["data"]["img_h"], augment=cfg["data"]["augment"])

    model = build_model(cfg, device)
    model.train()
    if cfg["train"]["freeze_bn"]:
        model.backbone.eval()  # mantener BN congelado en modo eval
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"],
                                backbone_mult=cfg["optim"]["backbone_mult"], slow_mult=cfg["optim"]["slow_mult"])
    epochs = 1 if smoke else cfg["train"]["epochs"]
    total_iters = max(1, len(dl) * epochs)
    scheduler = build_scheduler(optimizer, total_iters, warmup_iters=min(cfg["optim"]["warmup_iters"], total_iters))
    ema = ModelEMA(model, cfg["ema"]["decay"], cfg["ema"]["tau"]) if cfg["ema"]["enabled"] else None

    amp = cfg["train"]["amp"] and device == "cuda"
    cl = cfg["train"]["channels_last"] and device == "cuda"
    clip = cfg["train"]["grad_clip"]
    log_every = cfg["train"]["log_interval"]
    ckpt_dir = paths.project_root() / cfg["train"]["ckpt_dir"]

    print(f"device={device} | batch={bs} | epochs={epochs} | iters/epoca={len(dl)} | "
          f"amp(bf16)={amp} | anclas={cfg['model']['use_anchors']} deformable={cfg['model']['deformable']}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params entrenables: {n_params/1e6:.1f}M")

    it = 0
    last = {}
    for epoch in range(epochs):
        t0 = time.time()
        for batch in dl:
            images = batch["image"].to(device, non_blocking=True)
            if cl:
                images = images.to(memory_format=torch.channels_last)
            targets = prepare_targets(batch["targets"], device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
                pred = model(images)
                losses = criterion(pred, targets)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            it += 1
            last = {k: float(v.detach()) for k, v in losses.items()}
            if it % log_every == 0 or (smoke and it <= 3):
                lr = optimizer.param_groups[0]["lr"]
                print(f"  ep {epoch} it {it}  lr {lr:.2e}  total {last['total']:.3f}  "
                      f"cls {last['cls']:.3f}  iou {last['iou']:.3f}  ext {last['ext']:.3f}")
            if smoke and it >= 5:
                print("  [smoke] 5 iteraciones OK")
                return {"iters": it, "last": last}

        # checkpoint por época
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": cfg}
        if ema is not None:
            ckpt["ema"] = ema.state_dict()
        torch.save(ckpt, ckpt_dir / f"epoch_{epoch:03d}.pth")
        print(f"  época {epoch} hecha en {time.time()-t0:.1f}s  (checkpoint guardado)")

    return {"iters": it, "last": last}


def _parse_overrides(pairs):
    """--set a.b=val ...  -> dict anidado con casting básico."""
    ov: dict = {}
    for pair in pairs or []:
        key, val = pair.split("=", 1)
        if val.lower() in ("true", "false"):
            val = val.lower() == "true"
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
        node = ov
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="ruta al YAML (si no, usa los defaults)")
    ap.add_argument("--set", nargs="*", default=None, help="overrides: clave.anidada=valor")
    ap.add_argument("--smoke", action="store_true", help="5 iteraciones para validar")
    args = ap.parse_args()
    cfg = load_config(args.config, _parse_overrides(args.set))
    print(f"=== Entrenamiento LaneTR: {cfg['name']} ===")
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
