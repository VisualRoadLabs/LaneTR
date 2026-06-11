"""Verificación VISUAL del backbone + FPN (Paso 3.1).

Pasa imágenes reales por ResNet-18 (preentrenado) + FPN y dibuja, junto a cada imagen,
los mapas de activación medios de los 3 niveles de la pirámide (P3/P4/P5) como mapas de
calor. Si el backbone "ve" la estructura de la escena (carretera, bordes, vehículos), las
activaciones deben resaltar esas zonas y no ser ruido uniforme.

Salida: outputs/verify/backbone_fpn.png

Uso:
    .\.venv\Scripts\python.exe tools\verify_backbone_fpn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# En Windows con red corporativa, usar el almacén de certificados del sistema para que
# timm/HuggingFace pueda descargar los pesos preentrenados (best-effort).
import os
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
from lanetr.models import FPN, build_backbone

IMG_W, IMG_H = 800, 320


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def heatmap(act: torch.Tensor, size=(IMG_W, IMG_H)) -> np.ndarray:
    """Mapa de activación (H,W) -> imagen RGB con colormap, redimensionada a `size`."""
    a = act.detach().cpu().numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-6)
    a = (a * 255).astype(np.uint8)
    a = cv2.resize(a, size, interpolation=cv2.INTER_NEAREST)
    bgr = cv2.applyColorMap(a, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def labeled(img_arr, text, font):
    im = Image.fromarray(img_arr)
    canvas = Image.new("RGB", (im.width, im.height + 22), (0, 0, 0))
    canvas.paste(im, (0, 22))
    ImageDraw.Draw(canvas).text((5, 3), text, fill=(120, 220, 255), font=font)
    return canvas


def main() -> int:
    print("=" * 70)
    print("VERIFICACIÓN DEL BACKBONE + FPN (Paso 3.1)")
    print("=" * 70)
    font = _font(15)

    backbone = build_backbone("dla34", pretrained=True).eval()
    fpn = FPN(backbone.out_channels, out_channels=256).eval()
    n_bb = sum(p.numel() for p in backbone.parameters())
    n_fpn = sum(p.numel() for p in fpn.parameters())
    print(f"Backbone DLA-34: {n_bb/1e6:.1f}M params (ch={backbone.out_channels})  |  "
          f"FPN(256): {n_fpn/1e6:.2f}M params")

    ds = CULaneDataset("val", augment=False)
    rows = []
    for idx in [0, 3000, 8000]:
        s = ds[idx]
        x = s["image"].unsqueeze(0)
        with torch.no_grad():
            feats = backbone(x)
            pyr = fpn(feats)
        print(f"  #{idx}: C3/C4/C5 = {[tuple(f.shape) for f in feats]}  ->  "
              f"P3/P4/P5 = {[tuple(p.shape) for p in pyr]}")

        rgb = T.denormalize(s["image"])
        tiles = [labeled(rgb, "imagen", font)]
        for name, p in zip(["P3 (s8)", "P4 (s16)", "P5 (s32)"], pyr):
            hm = heatmap(p[0].mean(0))
            blend = (0.45 * rgb + 0.55 * hm).clip(0, 255).astype(np.uint8)
            tiles.append(labeled(blend, name, font))
        w = sum(t.width for t in tiles) + 4 * (len(tiles) + 1)
        row = Image.new("RGB", (w, tiles[0].height + 8), (25, 25, 25))
        xo = 4
        for t in tiles:
            row.paste(t, (xo, 4))
            xo += t.width + 4
        rows.append(row)

    sheet = Image.new("RGB", (rows[0].width, sum(r.height + 4 for r in rows) + 4), (25, 25, 25))
    yo = 4
    for r in rows:
        sheet.paste(r, (0, yo))
        yo += r.height + 4
    scale = min(1.0, 1500 / sheet.width)
    if scale < 1.0:
        sheet = sheet.resize((int(sheet.width * scale), int(sheet.height * scale)))
    out = paths.outputs_dir() / "verify" / "backbone_fpn.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"\nImagen guardada en: {out}")
    print("Cada fila: imagen | activación media de P3, P4, P5 (rojo=alta).")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
