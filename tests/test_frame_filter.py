"""Tests automáticos del filtro de coche parado (Paso 1A).

Corre con o sin pytest:
    .\.venv\Scripts\python.exe tests\test_frame_filter.py     # runner propio
    pytest tests/test_frame_filter.py                          # si tienes pytest
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lanetr import paths
from lanetr.data import frame_filter as ff

# Valores de referencia de CLRerNet / el dataset.
N_TRAIN = 88880
N_FILTERED_AT_15 = 55698


def test_diffs_length():
    diffs = ff.load_frame_diffs()
    assert diffs.shape == (N_TRAIN,), f"esperado ({N_TRAIN},), obtenido {diffs.shape}"


def test_threshold_15_keeps_expected_count():
    diffs = ff.load_frame_diffs()
    kept = int(ff.keep_mask(diffs, 15.0).sum())
    assert kept == N_FILTERED_AT_15, f"esperado {N_FILTERED_AT_15}, obtenido {kept}"


def test_filtered_list_matches_train_gt_new():
    """El filtrado de train_gt.txt a diff>=15 debe coincidir con train_gt_new.txt."""
    diffs = ff.load_frame_diffs()
    full = paths.list_dir() / "train_gt.txt"
    new = paths.list_dir() / "train_gt_new.txt"
    kept = {ff.image_of(l) for l in ff.build_filtered_list(full, diffs, 15.0)}
    expected = {ff.image_of(l) for l in ff.read_gt_list(new)}
    assert kept == expected, (
        f"discrepancia: solo_filtrado={len(kept - expected)}, solo_fichero={len(expected - kept)}"
    )


def test_diffs_aligned_to_train_gt():
    """El .npz debe tener exactamente una entrada por línea de train_gt.txt."""
    diffs = ff.load_frame_diffs()
    rows = ff.read_gt_list(paths.list_dir() / "train_gt.txt")
    assert len(diffs) == len(rows)


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests OK")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
