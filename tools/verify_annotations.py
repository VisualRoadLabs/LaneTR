"""Verificación VISUAL del parser de anotaciones de CULane (Paso 1B).

Qué hace, para que veas que el parseo es correcto:
  1. Selecciona automáticamente varias imágenes con distintos patrones de existencia
     (1110, 0110, 1111, ...).
  2. Por cada una genera una fila con DOS paneles lado a lado:
       - izquierda: la imagen con los carriles PARSEADOS dibujados encima
         (un color por slot, puntos marcados);
       - derecha: la máscara oficial `laneseg_label_w16` coloreada con los mismos colores.
     Si los carriles dibujados caen sobre las bandas de la máscara -> el parseo es correcto.
  3. Imprime un cross-check numérico: % de puntos de cada carril que caen sobre su slot
     en la máscara oficial.

Salida: outputs/verify/annotations_overlay.png

Uso:
    .\.venv\Scripts\python.exe tools\verify_annotations.py
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

# Un color por slot (0..3), izquierda -> derecha.
SLOT_COLORS = [(230, 50, 50), (60, 210, 60), (60, 130, 255), (245, 205, 30)]
# Patrones de existencia que queremos ilustrar (variedad de nº de carriles y huecos).
WANTED_PATTERNS = [(1, 1, 1, 1), (1, 1, 1, 0), (0, 1, 1, 0), (0, 1, 1, 1), (1, 1, 0, 0)]


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def colorize_seg(seg_arr: np.ndarray) -> Image.Image:
    h, w = seg_arr.shape
    out = np.zeros((h, w, 3), np.uint8)
    for v, color in enumerate(SLOT_COLORS, start=1):
        out[seg_arr == v] = color
    return Image.fromarray(out)


def draw_lanes(img: Image.Image, annotation: ann.LaneAnnotation) -> Image.Image:
    im = img.copy()
    d = ImageDraw.Draw(im)
    for lane in annotation.lanes:
        color = SLOT_COLORS[lane.slot] if lane.slot is not None else (255, 255, 255)
        pts = [(float(x), float(y)) for x, y in lane.points]
        if len(pts) >= 2:
            d.line(pts, fill=color, width=6)
        for x, y in pts:
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color)
    return im


def collect_samples(max_samples: int = 5) -> list:
    """Busca en train_gt.txt el primer ejemplo de cada patrón de existencia deseado."""
    found: dict[tuple, str] = {}
    gt = (paths.list_dir() / "train_gt.txt").read_text(encoding="utf-8").splitlines()
    for line in gt:
        if not line.strip():
            continue
        image, seg, existence = ann.parse_gt_line(line)
        if existence in WANTED_PATTERNS and existence not in found:
            found[existence] = line
        if len(found) == len(WANTED_PATTERNS):
            break
    return list(found.values())[:max_samples]


def build_row(line: str, header_font, target_w: int = 1300) -> tuple[Image.Image, str]:
    image, seg, existence = ann.parse_gt_line(line)
    annotation = ann.load_annotation(image, existence, seg)
    img = Image.open(paths.image_path(image)).convert("RGB")

    left = draw_lanes(img, annotation)
    if seg is not None and paths.image_path(seg).exists():
        seg_arr = np.array(Image.open(paths.image_path(seg)))
        right = colorize_seg(seg_arr)
    else:
        seg_arr = None
        right = Image.new("RGB", img.size, (15, 15, 15))

    # cross-check numérico
    agreements = []
    for lane in annotation.lanes:
        if seg_arr is not None:
            hits, total = ann.seg_agreement(lane, seg_arr)
            if total:
                agreements.append(hits / total)
    mean_agree = 100 * float(np.mean(agreements)) if agreements else float("nan")

    # componer paneles
    w, h = img.size
    row = Image.new("RGB", (2 * w + 12, h + 34), (0, 0, 0))
    row.paste(left, (0, 34))
    row.paste(right, (w + 12, 34))
    d = ImageDraw.Draw(row)
    name = "/".join(image.split("/")[-2:])
    d.text((6, 6),
           f"{name}   existencia={existence}   carriles={len(annotation)}   "
           f"acuerdo-con-máscara={mean_agree:.1f}%   [IZQ: parseado | DCHA: máscara oficial]",
           fill=(120, 220, 255), font=header_font)

    scale = target_w / row.width
    row = row.resize((target_w, int(row.height * scale)))
    summary = f"  {name}: existencia={existence}, carriles={len(annotation)}, acuerdo={mean_agree:.1f}%"
    return row, summary


def main() -> int:
    print("=" * 70)
    print("VERIFICACIÓN DEL PARSER DE ANOTACIONES (Paso 1B)")
    print("=" * 70)
    header_font = _font(20)

    lines = collect_samples()
    if not lines:
        print("[error] no se encontraron ejemplos en train_gt.txt")
        return 1

    rows, summaries = [], []
    for line in lines:
        row, summary = build_row(line, header_font)
        rows.append(row)
        summaries.append(summary)

    print("\nCross-check (centro del carril parseado vs. máscara oficial de 16 px):")
    print("\n".join(summaries))

    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows) + 6 * len(rows)),
                      (0, 0, 0))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 6
    out = paths.outputs_dir() / "verify" / "annotations_overlay.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)

    mean_all = np.nanmean([float(s.split("acuerdo=")[1].rstrip("%")) for s in summaries])
    print(f"\nAcuerdo medio global: {mean_all:.1f}%  (esperado alto: el centro cae sobre la banda)")
    print(f"Imagen guardada en: {out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
