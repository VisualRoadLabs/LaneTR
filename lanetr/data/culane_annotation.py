"""Parser de anotaciones de CULane (`*.lines.txt`).

Formato (verificado sobre el dataset):
  - Cada línea no vacía de un `.lines.txt` = un carril = ``x1 y1 x2 y2 ...`` en píxeles
    a resolución 1640×590.
  - `y` estrictamente DESCENDENTE, en rejilla fija de 10 px, rango ~[270, 590].
  - `x` puede ser NEGATIVO o > 1640 (extrapolaciones fuera de pantalla). **No se recorta.**
  - Los carriles van de izquierda a derecha. El nº de carriles del fichero == nº de unos en
    los flags de existencia de `*_gt.txt`. El carril j del fichero ocupa el j-ésimo slot
    (0..3) con flag==1, que coincide con el valor (slot+1) en la máscara `laneseg_label_w16`.

Este módulo solo lee y estructura las anotaciones; la conversión a la representación de
filas-ancla del modelo (x por fila fija) se hará más adelante, en la codificación de targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import paths

IMG_W, IMG_H = 1640, 590
#: Mínimo de puntos para considerar un carril válido.
MIN_POINTS = 2


@dataclass
class Lane:
    """Un carril: polilínea de puntos (x, y) ordenados por y descendente."""

    points: np.ndarray  # (N, 2) float32, columnas (x, y)
    slot: int | None = None  # 0..3 (posición izq->der) si se conoce, si no None

    @property
    def xs(self) -> np.ndarray:
        return self.points[:, 0]

    @property
    def ys(self) -> np.ndarray:
        return self.points[:, 1]

    def __len__(self) -> int:
        return len(self.points)


@dataclass
class LaneAnnotation:
    """Anotación completa de una imagen."""

    image: str  # ruta relativa de la imagen (como en las listas)
    lanes: list[Lane] = field(default_factory=list)
    existence: tuple[int, int, int, int] | None = None
    seg: str | None = None  # ruta relativa de la máscara de segmentación, si se conoce

    def __len__(self) -> int:
        return len(self.lanes)


# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
def lines_path_for_image(image_rel: str) -> Path:
    """Ruta absoluta del `.lines.txt` hermano de una imagen."""
    return paths.image_path(image_rel).with_suffix(".lines.txt")


# --------------------------------------------------------------------------- #
# Parseo
# --------------------------------------------------------------------------- #
def parse_lines_text(text: str) -> list[np.ndarray]:
    """Convierte el contenido de un `.lines.txt` en una lista de arrays (N, 2)."""
    lanes: list[np.ndarray] = []
    for line in text.splitlines():
        vals = line.split()
        if len(vals) < 2 * MIN_POINTS:
            continue
        pts = np.asarray(vals, dtype=np.float32).reshape(-1, 2)  # (N, 2) = (x, y)
        # Ordenar por y descendente (de abajo arriba) por robustez ante ficheros desordenados.
        order = np.argsort(-pts[:, 1], kind="stable")
        lanes.append(pts[order])
    return lanes


def parse_lines_file(path: str | Path) -> list[np.ndarray]:
    """Lee y parsea un `.lines.txt`. Si no existe, devuelve lista vacía (0 carriles)."""
    p = Path(path)
    if not p.exists():
        return []
    return parse_lines_text(p.read_text(encoding="utf-8"))


def _slots_from_existence(existence) -> list[int] | None:
    if existence is None:
        return None
    return [i for i, f in enumerate(existence) if int(f) == 1]


def load_annotation(image_rel: str, existence=None, seg: str | None = None) -> LaneAnnotation:
    """Carga la anotación de una imagen, asignando slots si se dan los flags de existencia."""
    raw = parse_lines_file(lines_path_for_image(image_rel))
    slots = _slots_from_existence(existence)
    lanes: list[Lane] = []
    for j, pts in enumerate(raw):
        slot = slots[j] if (slots is not None and j < len(slots)) else None
        lanes.append(Lane(points=pts, slot=slot))
    ex = tuple(int(x) for x in existence) if existence is not None else None
    return LaneAnnotation(image=image_rel, lanes=lanes, existence=ex, seg=seg)


def parse_gt_line(gt_line: str):
    """Descompone una línea de `*_gt.txt` -> (image_rel, seg_rel|None, existence|None).

    Tolera líneas de `test.txt` que solo tienen la ruta de imagen.
    """
    parts = gt_line.split()
    image = parts[0]
    seg = parts[1] if len(parts) > 1 and parts[1].endswith(".png") else None
    existence = tuple(int(x) for x in parts[2:6]) if len(parts) >= 6 else None
    return image, seg, existence


def load_annotations_from_list(list_path: str | Path, limit: int | None = None) -> list[LaneAnnotation]:
    """Carga una lista de anotaciones desde un fichero `*_gt.txt` (o `test.txt`)."""
    lines = [l for l in Path(list_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit is not None:
        lines = lines[:limit]
    out = []
    for l in lines:
        image, seg, existence = parse_gt_line(l)
        out.append(load_annotation(image, existence, seg))
    return out


# --------------------------------------------------------------------------- #
# Cross-check con la máscara de segmentación
# --------------------------------------------------------------------------- #
def seg_agreement(lane: Lane, seg_arr: np.ndarray, half_window: int = 9) -> tuple[int, int]:
    """Cuántos puntos del carril (dentro de imagen) tienen el slot correcto en la máscara.

    La máscara es de 16 px de ancho; comprobamos una ventana horizontal ±`half_window`
    alrededor de cada punto de la polilínea central. Devuelve (aciertos, total_en_imagen).
    """
    if lane.slot is None:
        return 0, 0
    h, w = seg_arr.shape
    target = lane.slot + 1
    hits = total = 0
    for x, y in lane.points:
        xi, yi = int(round(float(x))), int(round(float(y)))
        if not (0 <= yi < h and 0 <= xi < w):
            continue
        total += 1
        x0, x1 = max(0, xi - half_window), min(w, xi + half_window + 1)
        if (seg_arr[yi, x0:x1] == target).any():
            hits += 1
    return hits, total
