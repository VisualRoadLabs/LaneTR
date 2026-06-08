"""Verificación VISUAL del Dataset/transforms de CULane (Paso 1C).

Genera una rejilla que demuestra que la transformación geométrica es correcta:
  - Fila superior: 4 muestras de VAL (solo recorte+resize, sin augmentación).
  - Fila inferior: 4 AUGMENTACIONES del MISMO frame de train (flip/rotación/escala),
    con los carriles re-proyectados encima. Si los carriles siguen sobre la carretera
    en todas, las transformaciones imagen<->puntos están sincronizadas.

Salida: outputs/verify/dataset_batch.png

Uso:
    .\.venv\Scripts\python.exe tools\verify_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lanetr import paths
from lanetr.data import culane_annotation as ann
from lanetr.data import transforms as T

SLOT_COLORS = [(230, 50, 50), (60, 210, 60), (60, 130, 255), (245, 205, 30)]
IMG_W, IMG_H = 800, 320


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def raw_sample(line):
    image_rel, seg_rel, existence = ann.parse_gt_line(line)
    a = ann.load_annotation(image_rel, existence, seg_rel)
    img = Image.open(paths.image_path(image_rel)).convert("RGB")
    return {
        "image": img,
        "lanes": [l.points.copy() for l in a.lanes],
        "slots": [l.slot for l in a.lanes],
        "existence": a.existence,
        "meta": {"image_path": image_rel},
    }


def fresh(raw):
    return {
        "image": raw["image"],
        "lanes": [p.copy() for p in raw["lanes"]],
        "slots": list(raw["slots"]),
        "existence": raw["existence"],
        "meta": dict(raw["meta"]),
    }


def tile(sample, header, font, target_w=480):
    """sample ya transformado (image=tensor) -> PIL con carriles dibujados."""
    rgb = T.denormalize(sample["image"])
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    for pts, slot in zip(sample["lanes"], sample["slots"]):
        color = SLOT_COLORS[slot] if slot is not None else (255, 255, 255)
        xy = [(float(x), float(y)) for x, y in pts]
        if len(xy) >= 2:
            d.line(xy, fill=color, width=4)
        for x, y in xy:
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
    canvas = Image.new("RGB", (im.width, im.height + 24), (0, 0, 0))
    canvas.paste(im, (0, 24))
    ImageDraw.Draw(canvas).text((5, 4), header, fill=(120, 220, 255), font=font)
    scale = target_w / canvas.width
    return canvas.resize((target_w, int(canvas.height * scale)))


def grid(tiles, cols):
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    w = cols * tiles[0].width + (cols + 1) * 4
    h = len(rows) * tiles[0].height + (len(rows) + 1) * 4
    sheet = Image.new("RGB", (w, h), (25, 25, 25))
    for r, row in enumerate(rows):
        for c, t in enumerate(row):
            sheet.paste(t, (4 + c * (t.width + 4), 4 + r * (t.height + 4)))
    return sheet


def main() -> int:
    print("=" * 70)
    print("VERIFICACIÓN DEL DATASET / TRANSFORMS (Paso 1C)")
    print("=" * 70)
    font = _font(16)

    val_lines = [l for l in (paths.list_dir() / "val_gt.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    train_lines = [l for l in (paths.list_dir() / "train_gt_new.txt").read_text(encoding="utf-8").splitlines() if l.strip()]

    val_tf = T.build_transforms("val", IMG_W, IMG_H)
    train_tf = T.build_transforms("train", IMG_W, IMG_H)

    tiles = []
    # Fila superior: 4 muestras de val (sin augmentación)
    for i, idx in enumerate([0, 1000, 4000, 9000]):
        s = val_tf(raw_sample(val_lines[idx]), np.random.default_rng(0))
        tiles.append(tile(s, f"VAL #{idx}  (recorte+resize)", font))
        if i == 0:
            print(f"\nTensor de imagen: shape={tuple(s['image'].shape)}  dtype={s['image'].dtype}")
            print(f"  rango normalizado: min={s['image'].min():.2f}  max={s['image'].max():.2f}  "
                  f"media={s['image'].mean():.2f}")

    # Fila inferior: 4 augmentaciones del mismo frame de train
    raw = raw_sample(train_lines[0])
    print(f"\nAugmentando el frame de train: {raw['meta']['image_path']}")
    for k in range(4):
        s = train_tf(fresh(raw), np.random.default_rng(k + 1))
        tiles.append(tile(s, f"TRAIN aug #{k+1}", font))

    sheet = grid(tiles, cols=4)
    out = paths.outputs_dir() / "verify" / "dataset_batch.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"\nImagen guardada en: {out}")
    print("Fila 1 = VAL (limpio) | Fila 2 = TRAIN (4 augmentaciones del mismo frame)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
