"""Verificación VISUAL del filtro de "coche parado" (Paso 1A).

Qué hace, para que puedas comprobar tú mismo que el filtrado está bien:
  1. Carga `list/train_diffs.npz` y muestra estadísticas + histograma ASCII.
  2. Comprueba que aplicar el umbral 15 a `train_gt.txt` reproduce EXACTAMENTE
     `train_gt_new.txt` (la lista ya filtrada). -> PASS/FAIL.
  3. Genera montajes de imágenes en `outputs/verify/`:
       - `dropped_stopped_car.png`: una secuencia de frames DESCARTADOS (diff baja).
         Deben verse casi idénticos -> es el coche parado.
       - `kept_motion.png`: una secuencia de frames CONSERVADOS (diff alta).
         Debe verse movimiento entre frames.

Uso:
    .\.venv\Scripts\python.exe tools\verify_frame_filter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# permite ejecutar el script directamente (añade la raíz del proyecto al path)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# consola en UTF-8 (Windows usa cp1252 por defecto y rompe los acentos)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lanetr import paths
from lanetr.data import frame_filter as ff

THRESHOLD = ff.DEFAULT_THRESHOLD


# --------------------------------------------------------------------------- #
# Utilidades de presentación
# --------------------------------------------------------------------------- #
def ascii_histogram(diffs: np.ndarray, bins: int = 20, hi: float = 60.0, width: int = 50) -> str:
    edges = np.linspace(0, hi, bins + 1)
    counts, _ = np.histogram(np.clip(diffs, 0, hi), bins=edges)
    top = counts.max() or 1
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(round(width * c / top))
        lines.append(f"  [{edges[i]:5.1f}, {edges[i+1]:5.1f}) | {bar} {c}")
    return "\n".join(lines)


def _font(size: int = 16):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load_scaled(rel: str, scale: float = 0.32) -> Image.Image | None:
    p = paths.image_path(rel)
    if not p.exists():
        return None
    im = Image.open(p).convert("RGB")
    w, h = im.size
    return im.resize((int(w * scale), int(h * scale)))


def montage(frames: list[str], labels: list[str], out_path: Path, title: str) -> bool:
    """Apila verticalmente las imágenes con una etiqueta sobre cada una."""
    font = _font(16)
    title_font = _font(20)
    tiles = []
    for rel, label in zip(frames, labels):
        im = _load_scaled(rel)
        if im is None:
            continue
        w, h = im.size
        canvas = Image.new("RGB", (w, h + 26), (20, 20, 20))
        canvas.paste(im, (0, 26))
        d = ImageDraw.Draw(canvas)
        d.text((6, 4), label, fill=(255, 230, 120), font=font)
        tiles.append(canvas)
    if not tiles:
        print(f"  [aviso] No se encontró ninguna imagen para '{title}'.")
        return False
    w = max(t.width for t in tiles)
    total_h = 34 + sum(t.height for t in tiles)
    sheet = Image.new("RGB", (w, total_h), (0, 0, 0))
    d = ImageDraw.Draw(sheet)
    d.text((6, 6), title, fill=(120, 220, 255), font=title_font)
    y = 34
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return True


# --------------------------------------------------------------------------- #
# Búsqueda de secuencias contiguas dentro de un mismo vídeo
# --------------------------------------------------------------------------- #
def _video_dir(rel: str) -> str:
    return rel.rsplit("/", 1)[0]


def find_runs(images: list[str], diffs: np.ndarray, want_low: bool,
              threshold: float = THRESHOLD, min_len: int = 4) -> list[list[int]]:
    """Devuelve secuencias de índices contiguos (mismo vídeo) que cumplen la condición,
    ordenadas de más larga a más corta."""
    runs: list[list[int]] = []
    cur: list[int] = []
    for i in range(len(images)):
        cond = (diffs[i] < threshold) if want_low else (diffs[i] >= threshold)
        contiguous = (not cur) or (
            i == cur[-1] + 1 and _video_dir(images[i]) == _video_dir(images[cur[-1]])
        )
        if cond and contiguous:
            cur.append(i)
        else:
            if len(cur) >= min_len:
                runs.append(cur)
            cur = [i] if cond else []
    if len(cur) >= min_len:
        runs.append(cur)
    runs.sort(key=len, reverse=True)
    return runs


# --------------------------------------------------------------------------- #
def main() -> int:
    list_dir = paths.list_dir()
    full_gt = list_dir / "train_gt.txt"
    new_gt = list_dir / "train_gt_new.txt"

    print("=" * 70)
    print("VERIFICACIÓN DEL FILTRO DE COCHE PARADO (Paso 1A)")
    print("=" * 70)
    print(f"Dataset:   {paths.dataset_dir()}")
    print(f"Diffs:     {list_dir / 'train_diffs.npz'}")

    # 1) Estadísticas ------------------------------------------------------- #
    diffs = ff.load_frame_diffs()
    s = ff.stats(diffs)
    print(f"\n[1] Distribución de diffs ({s['n']} frames)")
    print(f"    min={s['min']:.2f}  max={s['max']:.2f}  media={s['mean']:.2f}  mediana={s['median']:.2f}")
    for t, info in s["thresholds"].items():
        print(f"    umbral >= {t:>4.0f}:  conserva {info['kept']:>6} ({info['pct']:.1f}%)")
    print(f"\n    Histograma (diff recortada a [0,60], umbral usado = {THRESHOLD:.0f}):")
    print(ascii_histogram(diffs))

    # 2) Equivalencia con train_gt_new.txt ---------------------------------- #
    print(f"\n[2] ¿Filtrar train_gt.txt a diff>={THRESHOLD:.0f} == train_gt_new.txt?")
    kept_lines = ff.build_filtered_list(full_gt, diffs, THRESHOLD)
    kept_imgs = {ff.image_of(l) for l in kept_lines}
    new_imgs = {ff.image_of(l) for l in ff.read_gt_list(new_gt)}
    only_filter = kept_imgs - new_imgs
    only_file = new_imgs - kept_imgs
    ok = (kept_imgs == new_imgs)
    print(f"    filtrado: {len(kept_imgs)}   train_gt_new.txt: {len(new_imgs)}")
    print(f"    solo en filtrado: {len(only_filter)}   solo en fichero: {len(only_file)}")
    print(f"    -> {'PASS [OK] (coinciden exactamente)' if ok else 'FAIL [X] (NO coinciden)'}")

    # 3) Montajes visuales --------------------------------------------------- #
    print("\n[3] Generando montajes en outputs/verify/ ...")
    images = [ff.image_of(l) for l in ff.read_gt_list(full_gt)]
    out_dir = paths.outputs_dir() / "verify"

    low_runs = find_runs(images, diffs, want_low=True, min_len=4)
    if low_runs:
        run = low_runs[0][:6]
        montage(
            [images[i] for i in run],
            [f"{images[i].split('/')[-1]}  diff={diffs[i]:.1f}  (DESCARTADO)" for i in run],
            out_dir / "dropped_stopped_car.png",
            f"DESCARTADOS (coche parado) — secuencia de {len(run)} frames casi idénticos",
        )
        print(f"    dropped_stopped_car.png  ({len(run)} frames, diff<{THRESHOLD:.0f})")
    else:
        print("    [aviso] no se encontró una secuencia larga de diffs bajas.")

    high_runs = find_runs(images, diffs, want_low=False, threshold=30.0, min_len=4)
    if high_runs:
        run = high_runs[0][:6]
        montage(
            [images[i] for i in run],
            [f"{images[i].split('/')[-1]}  diff={diffs[i]:.1f}  (CONSERVADO)" for i in run],
            out_dir / "kept_motion.png",
            f"CONSERVADOS (en movimiento) — secuencia de {len(run)} frames con cambio",
        )
        print(f"    kept_motion.png  ({len(run)} frames, diff>30)")

    print(f"\n    Abre las imágenes en: {out_dir}")
    print("\n" + "=" * 70)
    print("RESULTADO:", "TODO CORRECTO [OK]" if ok else "REVISAR [X]")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
