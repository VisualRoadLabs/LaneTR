"""Verificación VISUAL de la codificación a filas-ancla (Paso 1D).

Por cada muestra dibuja sobre la imagen:
  - en BLANCO fino, la polilínea ORIGINAL del carril;
  - en color por slot, los PUNTOS DECODIFICADOS desde la representación de filas-ancla
    (x predicha en cada fila `y` fija);
  - líneas de rejilla tenues en las filas ancla.
Si los puntos de colores caen sobre la línea blanca, la codificación/decodificación es fiel.
Imprime también el error medio/máx de reproyección (px).

Salida: outputs/verify/target_encoding.png

Uso:
    .\.venv\Scripts\python.exe tools\verify_target_encoding.py
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
from lanetr.data import target_encoding as TE
from lanetr.data import transforms as T

SLOT_COLORS = [(230, 50, 50), (60, 210, 60), (60, 130, 255), (245, 205, 30)]
IMG_W, IMG_H, ROWS = 800, 320, 144


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def cropped_sample(line):
    image_rel, seg, ex = ann.parse_gt_line(line)
    a = ann.load_annotation(image_rel, ex, seg)
    img = Image.open(paths.image_path(image_rel)).convert("RGB")
    sample = {"image": img, "lanes": [l.points.copy() for l in a.lanes],
              "slots": [l.slot for l in a.lanes], "existence": a.existence, "meta": {}}
    return T.CropResize(IMG_W, IMG_H, 270)(sample, np.random.default_rng(0)), image_rel


def render(line, row_ys, font, target_w=560):
    sample, name = cropped_sample(line)
    im = sample["image"].copy()
    d = ImageDraw.Draw(im)

    # rejilla de filas ancla (tenue)
    for y in row_ys[::6]:
        d.line([(0, float(y)), (IMG_W, float(y))], fill=(60, 60, 60), width=1)

    errs = []
    for pts, slot in zip(sample["lanes"], sample["slots"]):
        # original en blanco
        d.line([(float(x), float(y)) for x, y in pts], fill=(255, 255, 255), width=2)
        # decodificado en color
        t = TE.encode_lane(pts, row_ys, IMG_W, IMG_H, slot)
        dec = TE.decode_lane(t.xs, t.valid, row_ys, IMG_W)
        color = SLOT_COLORS[slot] if slot is not None else (255, 0, 255)
        for x, y in dec:
            d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=color, width=2)
        me, mx = TE.reprojection_error(pts, t, row_ys, IMG_W)
        errs.append((me, mx))

    mean_e = np.mean([e[0] for e in errs]) if errs else 0.0
    max_e = np.max([e[1] for e in errs]) if errs else 0.0
    canvas = Image.new("RGB", (im.width, im.height + 24), (0, 0, 0))
    canvas.paste(im, (0, 24))
    short = "/".join(name.split("/")[-2:])
    ImageDraw.Draw(canvas).text(
        (5, 4), f"{short}  filas={ROWS}  err medio={mean_e:.2f}px  max={max_e:.2f}px  "
                f"[blanco=original | aros=decodificado]", fill=(120, 220, 255), font=font)
    scale = target_w / canvas.width
    return canvas.resize((target_w, int(canvas.height * scale))), mean_e, max_e


def main() -> int:
    print("=" * 70)
    print("VERIFICACIÓN DE LA CODIFICACIÓN A FILAS-ANCLA (Paso 1D)")
    print("=" * 70)
    print(f"num_rows = {ROWS}  (espaciado vertical ~{(IMG_H-1)/(ROWS-1):.2f} px)")
    font = _font(15)
    row_ys = TE.make_row_ys(IMG_H, ROWS)

    lines = [l for l in (paths.list_dir() / "val_gt.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    picks = [0, 1500, 5000, 9000]

    tiles, all_mean, all_max = [], [], []
    for idx in picks:
        tile, me, mx = render(lines[idx], row_ys, font)
        tiles.append(tile)
        all_mean.append(me)
        all_max.append(mx)
        print(f"  muestra #{idx}: err medio={me:.2f}px  max={mx:.2f}px")

    sheet = Image.new("RGB", (tiles[0].width, sum(t.height + 4 for t in tiles) + 4), (25, 25, 25))
    y = 4
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.height + 4
    out = paths.outputs_dir() / "verify" / "target_encoding.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)

    print(f"\nError de reproyección global: medio={np.mean(all_mean):.2f}px  max={np.max(all_max):.2f}px")
    print(f"Imagen guardada en: {out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
