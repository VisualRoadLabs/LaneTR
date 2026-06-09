"""Evaluación de un checkpoint de LaneTR en CULane (Paso 6.3).

Carga un checkpoint (usa los pesos EMA si existen), opcionalmente calibra el umbral de
confianza en val, y reporta el F1 global y por categoría en test (métrica Python validada).

Uso:
    python tools/test.py --checkpoint outputs/checkpoints/best.pth --categories
    python tools/test.py --checkpoint best.pth --conf 0.4 --list test.txt
    python tools/test.py --checkpoint best.pth --calibrate         # busca el mejor umbral en val
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # para importar build_model de train.py
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

from lanetr.metrics import evaluate as ev
from train import build_model


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg, device)
    weights = ckpt.get("ema", ckpt["model"])
    model.load_state_dict(weights)
    model.eval()
    which = "EMA" if "ema" in ckpt else "online"
    print(f"checkpoint época {ckpt.get('epoch', '?')} | pesos {which}")
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--list", default="test.txt", help="lista a evaluar (relativa a list/)")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--calibrate", action="store_true", help="busca el mejor umbral en val_gt.txt")
    ap.add_argument("--categories", action="store_true", help="F1 por las 9 categorías")
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.checkpoint, device)
    kw = dict(batch_size=cfg["train"]["eval_batch_size"], num_workers=0,
              img_w=cfg["data"]["img_w"], img_h=cfg["data"]["img_h"],
              num_rows=cfg["data"]["num_rows"], max_images=args.max_images)

    conf = args.conf
    if args.calibrate:
        best, scores = ev.calibrate_threshold(model, "val_gt.txt", device, **kw)
        print("\nCalibración de umbral en val:")
        for t, s in sorted(scores.items()):
            mark = "  <-- mejor" if t == best else ""
            print(f"  thr={t:.2f}  F1={s['F1']:.4f}{mark}")
        conf = best
        print(f"Mejor umbral: {conf:.2f}")

    res = ev.evaluate_list(model, args.list, device, conf_thresh=conf, **kw)
    print(f"\n=== {args.list} | conf={conf:.2f} ===")
    print(f"F1={res['F1']:.4f}  P={res['Precision']:.4f}  R={res['Recall']:.4f}  "
          f"TP={res['TP']} FP={res['FP']} FN={res['FN']}")

    if args.categories:
        cats = ev.evaluate_categories(model, device, conf_thresh=conf, **kw)
        print("\n=== F1 por categoría ===")
        for name, s in cats.items():
            if name == "cross":
                print(f"  {name:8s}  FP={s['FP']}  (Crossroad: solo FP, mejor cuanto menor)")
            else:
                print(f"  {name:8s}  F1={s['F1']:.4f}")


if __name__ == "__main__":
    main()
