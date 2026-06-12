"""Precalcula la curvatura por frame del split de train -> list/train_curvature.npz (Paso 7.3).

Para cada frame, codifica sus carriles a la representación de filas-ancla (igual que el modelo)
y mide, por carril, la desviación máxima de `xs` respecto a su cuerda (recta extremo-extremo);
el score del frame = máximo sobre sus carriles. Se usa para SOBRE-MUESTREAR las curvas en el
entrenamiento (son ~1-3% de CULane), atacando el "colapso al carril medio".

IMPORTANTE: el .npz queda alineado a la MISMA lista que consume el dataset (train_gt_new.txt,
55.698 frames). Hay que generarlo en la máquina con el dataset (DATASET_DIR) — p.ej. el A6000.

Uso:
    python tools/compute_curvature.py                 # split train (filtrado), 144 filas
    python tools/compute_curvature.py --limit 200     # prueba rápida
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

from lanetr import paths
from lanetr.data import culane_annotation as ann
from lanetr.data import target_encoding as TE
from lanetr.data.culane_dataset import _LIST_BY_SPLIT

ORIG_W = 1640  # resolución original de CULane (la y se recorta en cut_height)


def _lane_curvature(xs: np.ndarray, valid: np.ndarray) -> float:
    """Desviación máx (xs normalizada) respecto a la cuerda entre el 1er y último punto válido."""
    idx = np.where(valid)[0]
    if len(idx) < 3:
        return 0.0
    f, l = idx[0], idx[-1]
    rows = np.arange(len(xs), dtype=np.float32)
    chord = xs[f] + (xs[l] - xs[f]) * (rows - f) / max(l - f, 1)
    dev = np.abs(xs - chord)
    dev[~valid] = 0.0
    return float(dev.max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--img-w", type=int, default=800)
    ap.add_argument("--img-h", type=int, default=320)
    ap.add_argument("--cut-height", type=int, default=270)
    ap.add_argument("--num-rows", type=int, default=144)
    ap.add_argument("--limit", type=int, default=None, help="solo N frames (prueba)")
    ap.add_argument("--out", default=None, help="ruta de salida (def: list/train_curvature.npz)")
    args = ap.parse_args()

    list_path = paths.list_dir() / _LIST_BY_SPLIT[args.split]
    entries = [l for l in list_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        entries = entries[: args.limit]
    row_ys = TE.make_row_ys(args.img_h, args.num_rows)
    sx = args.img_w / ORIG_W

    scores = np.zeros(len(entries), dtype=np.float32)
    for k, e in enumerate(entries):
        image_rel, seg_rel, existence = ann.parse_gt_line(e)
        a = ann.load_annotation(image_rel, existence, seg_rel)
        lanes = []
        for lane in a.lanes:
            p = lane.points.astype(np.float32).copy()
            p[:, 0] *= sx                       # x: 1640 -> img_w
            p[:, 1] -= args.cut_height          # y: recorte (img_h = 590-cut)
            lanes.append(p)
        enc = TE.encode_sample(lanes, [lane.slot for lane in a.lanes], row_ys, args.img_w, args.img_h)
        fr = 0.0
        for i in range(enc["xs"].shape[0]):
            fr = max(fr, _lane_curvature(enc["xs"][i], enc["valid"][i]))
        scores[k] = fr
        if (k + 1) % 5000 == 0:
            print(f"  {k + 1}/{len(entries)} frames...")

    out = Path(args.out) if args.out else (paths.list_dir() / "train_curvature.npz")
    np.savez(out, data=scores)
    q = np.quantile(scores, [0.5, 0.9, 0.95, 0.99])
    print(f"\nGuardado: {out}  ({len(scores)} frames, alineado a {_LIST_BY_SPLIT[args.split]})")
    print(f"curvatura (xs norm.): min {scores.min():.4f}  mediana {q[0]:.4f}  p90 {q[1]:.4f}  "
          f"p95 {q[2]:.4f}  p99 {q[3]:.4f}  max {scores.max():.4f}")
    print(f"frames con curvatura > 0.02 (curva clara): {(scores > 0.02).mean() * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
