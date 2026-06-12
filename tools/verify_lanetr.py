"""Verificación VISUAL del modelo completo LaneTR de extremo a extremo (Paso 3.4).

Pasa imágenes reales por el modelo entero y dibuja las líneas candidatas que emite cada query
(un color por query, con su confianza). El modelo está SIN ENTRENAR: las líneas son "basura"
geométricamente válida — el objetivo es demostrar que el modelo CORRE de extremo a extremo y
produce carriles BIEN FORMADOS (en filas-ancla), listos para añadirles la pérdida (Paso 4).

Salida: outputs/verify/lanetr_endtoend.png

Uso:
    .\.venv\Scripts\python.exe tools\verify_lanetr.py
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

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from lanetr import paths
from lanetr.data import transforms as T
from lanetr.data.culane_dataset import CULaneDataset
from lanetr.models import LaneTR

NUM_QUERIES = 12


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def query_colors(n):
    cols = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.9, 1.0)
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    return cols


def main() -> int:
    print("=" * 70)
    print("VERIFICACIÓN END-TO-END DE LaneTR (Paso 3.4) — modelo SIN ENTRENAR")
    print("=" * 70)
    font = _font(14)

    # semilla fija para que la 'basura' sea reproducible
    torch.manual_seed(0)
    model = LaneTR(backbone="dla34", pretrained=True, num_queries=NUM_QUERIES, num_layers=6)
    n = sum(p.numel() for p in model.parameters())
    print(f"LaneTR: {n/1e6:.1f}M params  ({NUM_QUERIES} queries)")
    colors = query_colors(NUM_QUERIES)

    ds = CULaneDataset("val", augment=False)
    rows = []
    for idx in [0, 4000, 9000]:
        s = ds[idx]
        lanes = model.predict(s["image"].unsqueeze(0), conf_thresh=None)[0]  # todas las queries
        rgb = T.denormalize(s["image"])
        im = Image.fromarray(rgb)
        d = ImageDraw.Draw(im)
        for lane in lanes:
            color = colors[lane["query"]]
            pts = [(float(x), float(y)) for x, y in lane["points"]]
            if len(pts) >= 2:
                d.line(pts, fill=color, width=2)
        canvas = Image.new("RGB", (im.width, im.height + 20), (0, 0, 0))
        canvas.paste(im, (0, 20))
        ImageDraw.Draw(canvas).text((4, 3), f"val #{idx}: {len(lanes)} carriles candidatos "
                                    f"(1 color/query)", fill=(120, 220, 255), font=font)
        rows.append(canvas)
        confs = ", ".join(f"{l['conf']:.2f}" for l in lanes[:NUM_QUERIES])
        print(f"  #{idx}: {len(lanes)} candidatos | confianzas: {confs}")

    sheet = Image.new("RGB", (rows[0].width, sum(r.height + 4 for r in rows) + 4), (25, 25, 25))
    yo = 4
    for r in rows:
        sheet.paste(r, (0, yo))
        yo += r.height + 4
    scale = min(1.0, 1400 / sheet.width)
    if scale < 1.0:
        sheet = sheet.resize((int(sheet.width * scale), int(sheet.height * scale)))
    out = paths.outputs_dir() / "verify" / "lanetr_endtoend.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"\nImagen guardada en: {out}")
    print("Líneas = candidatos por query (geometría válida, valores SIN ENTRENAR).")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
