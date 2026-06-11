"""Verificación VISUAL del decoder transformer (Paso 3.2).

Pasa una imagen real por backbone + FPN + decoder y dibuja, para varias queries, su mapa de
cross-attention sobre el nivel fino P3 (40×100) superpuesto a la imagen: muestra "qué zona
mira" cada query.

IMPORTANTE: el modelo está SIN ENTRENAR. Este visual demuestra que el MECANISMO está bien
conectado (cada query produce una distribución de atención sobre la imagen), no que detecte
carriles todavía. El visual con carriles reales llega en el Paso 3.4.

Salida: outputs/verify/decoder_attn.png

Uso:
    .\.venv\Scripts\python.exe tools\verify_decoder.py
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

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from lanetr import paths
from lanetr.data import transforms as T
from lanetr.data.culane_dataset import CULaneDataset
from lanetr.models import FPN, LaneDecoder, build_backbone

NUM_QUERIES = 12
SHOW_QUERIES = 6


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def overlay_attention(rgb, attn_hw):
    a = attn_hw.detach().cpu().numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    a = cv2.resize((a * 255).astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                   interpolation=cv2.INTER_CUBIC)
    hm = cv2.cvtColor(cv2.applyColorMap(a, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (0.5 * rgb + 0.5 * hm).clip(0, 255).astype(np.uint8)


def labeled(arr, text, font):
    im = Image.fromarray(arr)
    canvas = Image.new("RGB", (im.width, im.height + 20), (0, 0, 0))
    canvas.paste(im, (0, 20))
    ImageDraw.Draw(canvas).text((4, 3), text, fill=(120, 220, 255), font=font)
    return canvas


def main() -> int:
    print("=" * 70)
    print("VERIFICACIÓN DEL DECODER TRANSFORMER (Paso 3.2) — modelo SIN ENTRENAR")
    print("=" * 70)
    font = _font(14)

    backbone = build_backbone("dla34", pretrained=True).eval()
    fpn = FPN(backbone.out_channels, 256).eval()
    decoder = LaneDecoder(d_model=256, num_queries=NUM_QUERIES, num_layers=6).eval()
    print(f"Decoder: {NUM_QUERIES} queries, 6 capas, "
          f"{sum(p.numel() for p in decoder.parameters())/1e6:.2f}M params")

    ds = CULaneDataset("val", augment=False)
    s = ds[0]
    rgb = T.denormalize(s["image"])
    with torch.no_grad():
        feats = fpn(backbone(s["image"].unsqueeze(0)))
        hs, attns, shapes = decoder(feats, need_attn=True)

    print(f"Salida decoder hs: {tuple(hs.shape)} (capas, batch, queries, dim)")
    print(f"Tokens de memoria: {attns[-1].shape[-1]}  (niveles {shapes})")

    # atención de la última capa sobre P3 (nivel fino)
    h0, w0 = shapes[0]
    attn_last = attns[-1][0]                       # (NQ, total_tokens)
    attn_p3 = attn_last[:, : h0 * w0].reshape(NUM_QUERIES, h0, w0)

    tiles = [labeled(rgb, "imagen", font)]
    for qi in range(SHOW_QUERIES):
        tiles.append(labeled(overlay_attention(rgb, attn_p3[qi]), f"query {qi}", font))

    cols = (len(tiles) + 1) // 2  # 2 filas
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    tw, th = tiles[0].width, tiles[0].height
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * 4, len(rows) * th + (len(rows) + 1) * 4),
                      (25, 25, 25))
    for r, row in enumerate(rows):
        for c, t in enumerate(row):
            sheet.paste(t, (4 + c * (tw + 4), 4 + r * (th + 4)))
    scale = min(1.0, 1500 / sheet.width)
    if scale < 1.0:
        sheet = sheet.resize((int(sheet.width * scale), int(sheet.height * scale)))
    out = paths.outputs_dir() / "verify" / "decoder_attn.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"\nImagen guardada en: {out}")
    print("Cada panel: zona que mira cada query (cross-attention sobre P3). SIN ENTRENAR.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
