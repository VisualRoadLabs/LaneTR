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
import json
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
from lanetr.metrics import evaluate as ev
from lanetr.models import LaneTR
from lanetr.training import (
    ETA, EpochVisualizer, GPUMonitor, ModelEMA, build_optimizer, build_scheduler,
    create_work_dir, freeze_batchnorm, get_logger,
)


def build_model(cfg, device):
    m = cfg["model"]
    model = LaneTR(backbone=m["backbone"], pretrained=m["pretrained"], d_model=m["d_model"],
                   num_queries=m["num_queries"], num_layers=m["num_layers"],
                   num_rows=m["num_rows"], img_h=cfg["data"]["img_h"],
                   use_anchors=m["use_anchors"], deformable=m["deformable"],
                   n_points=m["n_points"], n_ref_points=m.get("n_ref_points", 1),
                   ref_refine=m.get("ref_refine", False),
                   ref_refine_mode=m.get("ref_refine_mode", "mlp"),
                   ref_y_top=m.get("ref_y_top", 0.15), ref_y_bottom=m.get("ref_y_bottom", 0.95))
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
                         o2m_k=l["o2m_k"], img_h=cfg["data"]["img_h"],
                         geo_metric=l.get("geo_metric", "laneiou"),
                         curve_gamma=l.get("curve_gamma", 0.0), curve_thresh=l.get("curve_thresh", 0.005),
                         curve_scale=l.get("curve_scale", 0.03), curve_cap=l.get("curve_cap", 2.0),
                         w_curv=l.get("w_curv", 0.0))


def train(cfg, smoke=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["train"]["seed"])
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    bs = 2 if smoke else cfg["data"]["batch_size"]
    workers = 0 if smoke else cfg["data"]["num_workers"]
    dl = build_dataloader(cfg["data"].get("train_split", "train"), batch_size=bs, shuffle=True,
                          num_workers=workers,
                          seed=cfg["train"]["seed"], encode_targets=True,
                          num_rows=cfg["data"]["num_rows"], img_w=cfg["data"]["img_w"],
                          img_h=cfg["data"]["img_h"], augment=cfg["data"]["augment"],
                          curve_oversample=cfg["data"].get("curve_oversample", False),
                          curve_alpha=cfg["data"].get("curve_alpha", 4.0),
                          curve_top_frac=cfg["data"].get("curve_top_frac", 0.1))

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
    conf_thr = cfg["train"]["eval_conf_thresh"]

    # work_dir + logging + GPU + visualizaciones (solo en entrenamiento real, no en smoke)
    if not smoke:
        work = create_work_dir(cfg)
        train_log = get_logger("lanetr.train", work / "train.log")
        eval_log = get_logger("lanetr.eval", work / "eval.log")
        gpu_log = get_logger("lanetr.gpu", work / "gpu.log")
        viz = EpochVisualizer(device, cfg["data"]["img_w"], cfg["data"]["img_h"],
                              cfg["data"]["num_rows"], conf_thresh=conf_thr)
        gpu = GPUMonitor(device)
        ckpt_dir = work / "checkpoints"
        log = train_log.info
        log(f"work_dir: {work}")
    else:
        work = None
        viz = gpu = eval_log = None
        ckpt_dir = paths.outputs_dir() / "smoke_ckpt"
        log = print
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log(f"device={device} | batch={bs} | epochs={epochs} | iters/epoca={len(dl)} | amp(bf16)={amp} | "
        f"anclas={cfg['model']['use_anchors']} deformable={cfg['model']['deformable']}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"params entrenables: {n_params/1e6:.1f}M")

    eta = ETA(total_iters)
    it = 0
    last = {}
    best_f1 = -1.0
    for epoch in range(epochs):
        t0 = time.time()
        if gpu:
            gpu.epoch_start()
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
            eta_str = eta.step(it)
            if it % log_every == 0 or (smoke and it <= 3):
                if gpu:
                    gpu.sample()
                lr = optimizer.param_groups[0]["lr"]
                loss_str = "  ".join(f"loss_{k} {v:.4f}" for k, v in last.items())
                log(f"ep {epoch+1}/{epochs} it {it}/{total_iters}  lr {lr:.6f}  {loss_str}  ETA {eta_str}")
            if smoke and it >= 5:
                print("  [smoke] 5 iteraciones OK")
                # validar el hook de evaluación (con EMA) sobre 4 imágenes
                eval_model = ema.ema if ema is not None else model
                res = ev.evaluate_list(eval_model, "val_gt.txt", device,
                                       conf_thresh=cfg["train"]["eval_conf_thresh"],
                                       batch_size=4, num_workers=0, img_w=cfg["data"]["img_w"],
                                       img_h=cfg["data"]["img_h"], num_rows=cfg["data"]["num_rows"],
                                       max_images=4)
                print(f"  [smoke] eval F1={res['F1']:.4f} (4 imágenes)")
                return {"iters": it, "last": last, "eval_f1": res["F1"]}

        # --- fin de época: checkpoint, GPU, evaluación, visualizaciones ---
        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": cfg}
        if ema is not None:
            ckpt["ema"] = ema.state_dict()
        torch.save(ckpt, ckpt_dir / "last.pth")
        log(f"época {epoch+1}/{epochs} hecha en {time.time()-t0:.1f}s")

        if gpu is not None:
            g = gpu.epoch_summary()
            um = f"{g['util_mean']:.0f}" if g.get("util_mean") is not None else "n/a"
            ux = f"{g['util_max']:.0f}" if g.get("util_max") is not None else "n/a"
            gpu_log.info(f"época {epoch+1}: mem_pico {g.get('mem_peak_gb', 0):.2f}GB  "
                         f"reservada {g.get('mem_reserved_gb', 0):.2f}GB  util media {um}%  máx {ux}%")

        eval_model = ema.ema if ema is not None else model
        do_eval = ((epoch + 1) % cfg["train"]["eval_interval"] == 0) or (epoch == epochs - 1)
        if do_eval:
            res = ev.evaluate_list(
                eval_model, "val_gt.txt", device, conf_thresh=conf_thr,
                batch_size=cfg["train"]["eval_batch_size"], num_workers=0,
                img_w=cfg["data"]["img_w"], img_h=cfg["data"]["img_h"],
                num_rows=cfg["data"]["num_rows"], max_images=cfg["train"]["eval_max_images"])
            msg = (f"época {epoch+1}: F1={res['F1']:.4f}  P={res['Precision']:.4f}  "
                   f"R={res['Recall']:.4f}  TP={res['TP']} FP={res['FP']} FN={res['FN']}")
            (eval_log.info if eval_log else print)(msg)
            if res["F1"] > best_f1:
                best_f1 = res["F1"]
                torch.save(ckpt, ckpt_dir / "best.pth")
                log(f"** nuevo mejor F1={best_f1:.4f} -> best.pth")

        if viz is not None:
            viz.visualize(eval_model, criterion.matcher, epoch, work / "viz")
            log(f"visualizaciones -> viz/epoch_{epoch:03d}/")

    # --- evaluación FINAL: calibra umbral (subconjunto de val) + F1 test global y por categoría
    # en UNA sola pasada de inferencia -> results.json. Protegida para no bloquear si falla. ---
    if work is not None and cfg["train"].get("final_eval", True):
        log("evaluación final: calibrando umbral en val (subconjunto) y test en 1 pasada...")
        try:
            ekw = dict(batch_size=cfg["train"]["eval_batch_size"], num_workers=0,
                       img_w=cfg["data"]["img_w"], img_h=cfg["data"]["img_h"], num_rows=cfg["data"]["num_rows"])
            best_thr, _ = ev.calibrate_threshold(eval_model, "val_gt.txt", device, max_images=3000, **ekw)
            test_res, cats = ev.evaluate_test_and_categories(eval_model, device, conf_thresh=best_thr, **ekw)
            results = {
                "name": cfg["name"], "best_f1_val": best_f1, "conf_thresh": best_thr,
                "test_F1": test_res["F1"], "test_P": test_res["Precision"], "test_R": test_res["Recall"],
                "categories": {k: (v["FP"] if k == "cross" else v["F1"]) for k, v in cats.items()},
                "overrides": {"geo_metric": cfg["loss"].get("geo_metric"),
                              "num_queries": cfg["model"]["num_queries"],
                              "deformable": cfg["model"]["deformable"],
                              "n_ref_points": cfg["model"].get("n_ref_points", 1),
                              "ref_refine": cfg["model"].get("ref_refine", False),
                              "ref_refine_mode": cfg["model"].get("ref_refine_mode", "mlp"),
                              "aux_one_to_many": cfg["loss"]["aux_one_to_many"],
                              "train_split": cfg["data"].get("train_split")},
            }
            (work / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            (eval_log.info if eval_log else print)(
                f"FINAL test F1={test_res['F1']:.4f} (umbral {best_thr:.2f}) -> results.json")
        except Exception as e:  # noqa: BLE001
            log(f"[aviso] la evaluación final falló ({e}); el entrenamiento queda guardado igualmente")

    return {"iters": it, "last": last, "best_f1": best_f1,
            "work_dir": str(work) if work else None}


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
