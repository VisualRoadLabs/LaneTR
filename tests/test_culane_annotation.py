"""Tests automáticos del parser de anotaciones de CULane (Paso 1B).

    .\.venv\Scripts\python.exe tests\test_culane_annotation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from lanetr import paths
from lanetr.data import culane_annotation as ann


def _sample_lines(n=200):
    gt = (paths.list_dir() / "train_gt.txt").read_text(encoding="utf-8").splitlines()
    gt = [l for l in gt if l.strip()]
    step = max(1, len(gt) // n)
    return gt[::step][:n]


def test_parse_first_image_has_four_lanes():
    line = (paths.list_dir() / "train_gt.txt").read_text(encoding="utf-8").splitlines()[0]
    image, seg, existence = ann.parse_gt_line(line)
    a = ann.load_annotation(image, existence, seg)
    assert existence == (1, 1, 1, 1)
    assert len(a) == 4
    for lane in a.lanes:
        assert lane.points.ndim == 2 and lane.points.shape[1] == 2
        assert len(lane) >= ann.MIN_POINTS


def test_existence_matches_lane_count():
    bad = []
    for line in _sample_lines():
        image, seg, existence = ann.parse_gt_line(line)
        a = ann.load_annotation(image, existence, seg)
        if existence is not None and sum(existence) != len(a):
            bad.append((image, existence, len(a)))
    assert not bad, f"{len(bad)} casos con sum(existencia)!=nº carriles, p.ej. {bad[:3]}"


def test_y_descending_and_grid():
    """y estrictamente descendente y pasos múltiplos de 10."""
    for line in _sample_lines(60):
        image, _, existence = ann.parse_gt_line(line)
        a = ann.load_annotation(image, existence)
        for lane in a.lanes:
            ys = lane.ys
            assert np.all(np.diff(ys) < 0), f"y no descendente en {image}"
            steps = np.round(-np.diff(ys), 3)
            assert np.all(np.abs(steps % 10.0) < 1e-3), f"paso de y no múltiplo de 10 en {image}"


def test_x_not_clipped():
    """Debe existir al menos un punto fuera de [0,1640] en el dataset (no se recorta)."""
    out_of_bounds = False
    for line in _sample_lines(100):
        image, _, existence = ann.parse_gt_line(line)
        a = ann.load_annotation(image, existence)
        for lane in a.lanes:
            if lane.xs.min() < 0 or lane.xs.max() > ann.IMG_W:
                out_of_bounds = True
                break
        if out_of_bounds:
            break
    assert out_of_bounds, "no se halló ningún x fuera de imagen; ¿se está recortando?"


def test_slot_mapping_with_gap():
    """Un patrón con hueco (p.ej. 0110) debe mapear a los slots correctos."""
    target = None
    for line in _sample_lines(2000):
        image, seg, existence = ann.parse_gt_line(line)
        if existence == (0, 1, 1, 0):
            target = (image, seg, existence)
            break
    if target is None:
        return  # patrón no presente en la submuestra; no es un fallo
    image, seg, existence = target
    a = ann.load_annotation(image, existence, seg)
    assert [lane.slot for lane in a.lanes] == [1, 2]


def test_seg_mask_agreement():
    """El centro del carril parseado debe caer sobre su slot en la máscara oficial."""
    agrees = []
    for line in _sample_lines(60):
        image, seg, existence = ann.parse_gt_line(line)
        if seg is None or not paths.image_path(seg).exists():
            continue
        seg_arr = np.array(Image.open(paths.image_path(seg)))
        a = ann.load_annotation(image, existence, seg)
        for lane in a.lanes:
            hits, total = ann.seg_agreement(lane, seg_arr)
            if total:
                agrees.append(hits / total)
    assert agrees, "no se pudo evaluar el acuerdo con la máscara"
    mean = float(np.mean(agrees))
    assert mean > 0.85, f"acuerdo medio con máscara demasiado bajo: {mean:.3f}"


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
