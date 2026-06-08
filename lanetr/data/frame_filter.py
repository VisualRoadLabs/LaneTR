"""Filtro de "coche parado" para CULane (frame-diff de CLRerNet).

El fichero `list/train_diffs.npz` (clave ``data``) contiene, por cada frame de
`list/train_gt.txt`, la diferencia media de píxeles respecto al frame anterior del vídeo.
Una diferencia baja significa que el frame es casi idéntico al anterior (coche parado /
atasco) y se descarta. CLRerNet usa umbral 15, que conserva el 62.7% del train.

Este módulo NO recalcula los diffs (eso lo hizo `calculate_frame_diff.py` de CLRerNet);
solo aplica el umbral y construye la lista filtrada de forma reproducible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import paths

#: Umbral por defecto (idéntico a CLRerNet: conserva 55.698/88.880 = 62.7%).
DEFAULT_THRESHOLD = 15.0


def load_frame_diffs(npz_path: str | Path | None = None) -> np.ndarray:
    """Carga el vector de diffs (uno por frame de train_gt.txt)."""
    npz_path = Path(npz_path) if npz_path else paths.list_dir() / "train_diffs.npz"
    with np.load(npz_path) as z:
        if "data" not in z:
            raise KeyError(f"Se esperaba la clave 'data' en {npz_path}; claves: {list(z.keys())}")
        return np.asarray(z["data"], dtype=np.float64)


def read_gt_list(path: str | Path) -> list[str]:
    """Lee un fichero de lista (train_gt.txt, etc.) como líneas no vacías."""
    return [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def image_of(gt_line: str) -> str:
    """Devuelve la ruta de imagen (primer campo) de una línea de *_gt.txt."""
    return gt_line.split()[0]


def keep_mask(diffs: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Máscara booleana de frames a conservar (diff >= threshold)."""
    return diffs >= threshold


def build_filtered_list(
    full_gt_path: str | Path,
    diffs: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    out_path: str | Path | None = None,
) -> list[str]:
    """Aplica el umbral a `full_gt_path` (alineado con `diffs`) y devuelve las líneas
    conservadas. Si `out_path` se indica, también las escribe a disco."""
    rows = read_gt_list(full_gt_path)
    if len(rows) != len(diffs):
        raise ValueError(
            f"Desalineado: {len(rows)} líneas en {full_gt_path} vs {len(diffs)} diffs. "
            "El .npz debe corresponder a train_gt.txt (mismo orden)."
        )
    mask = keep_mask(diffs, threshold)
    kept = [row for row, keep in zip(rows, mask) if keep]
    if out_path is not None:
        Path(out_path).write_text("\n".join(kept) + "\n", encoding="utf-8")
    return kept


def stats(diffs: np.ndarray, thresholds=(10.0, 15.0, 20.0)) -> dict:
    """Estadísticas resumidas del vector de diffs."""
    out = {
        "n": int(diffs.size),
        "min": float(diffs.min()),
        "max": float(diffs.max()),
        "mean": float(diffs.mean()),
        "median": float(np.median(diffs)),
        "thresholds": {},
    }
    for t in thresholds:
        kept = int(keep_mask(diffs, t).sum())
        out["thresholds"][float(t)] = {"kept": kept, "pct": 100.0 * kept / diffs.size}
    return out
