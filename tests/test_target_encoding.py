"""Tests automáticos de la codificación a filas-ancla (Paso 1D).

    .\.venv\Scripts\python.exe tests\test_target_encoding.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lanetr import paths
from lanetr.data import culane_annotation as ann
from lanetr.data import culane_dataset as cd
from lanetr.data import target_encoding as TE
from lanetr.data import transforms as T

IMG_W, IMG_H, ROWS = 800, 320, 144


def _real_lanes(n_images=20):
    """Carga carriles reales ya recortados/redimensionados (sin augmentación)."""
    lines = [l for l in (paths.list_dir() / "val_gt.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    cr = T.CropResize(IMG_W, IMG_H, 270)
    out = []
    for line in lines[:n_images]:
        image_rel, seg, ex = ann.parse_gt_line(line)
        a = ann.load_annotation(image_rel, ex, seg)
        from PIL import Image
        sample = {"image": Image.new("RGB", (1640, 590)), "lanes": [l.points.copy() for l in a.lanes],
                  "slots": [l.slot for l in a.lanes], "existence": a.existence, "meta": {}}
        sample = cr(sample, np.random.default_rng(0))
        out.extend([(p, s) for p, s in zip(sample["lanes"], sample["slots"])])
    return out


def test_row_ys_shape_and_range():
    ys = TE.make_row_ys(IMG_H, ROWS)
    assert ys.shape == (ROWS,)
    assert ys[0] == 0.0 and abs(ys[-1] - (IMG_H - 1)) < 1e-4
    assert np.all(np.diff(ys) > 0)


def test_encode_shapes():
    row_ys = TE.make_row_ys(IMG_H, ROWS)
    pts, slot = _real_lanes(1)[0]
    t = TE.encode_lane(pts, row_ys, IMG_W, IMG_H, slot)
    assert t.xs.shape == (ROWS,) and t.valid.shape == (ROWS,)
    assert t.valid.any(), "el carril no marca ninguna fila válida"


def test_valid_matches_lane_extent():
    """Las filas válidas deben caer dentro del tramo [y_top, y_bottom] del carril."""
    row_ys = TE.make_row_ys(IMG_H, ROWS)
    for pts, slot in _real_lanes(10):
        t = TE.encode_lane(pts, row_ys, IMG_W, IMG_H, slot)
        y_top, y_bottom = pts[:, 1].min(), pts[:, 1].max()
        inside = (row_ys >= y_top) & (row_ys <= y_bottom)
        assert np.array_equal(t.valid, inside)


def test_roundtrip_reprojection_subpixel():
    """Codificar->decodificar debe reproducir el carril con error sub-píxel."""
    row_ys = TE.make_row_ys(IMG_H, ROWS)
    errs = []
    for pts, slot in _real_lanes(20):
        t = TE.encode_lane(pts, row_ys, IMG_W, IMG_H, slot)
        mean_err, max_err = TE.reprojection_error(pts, t, row_ys, IMG_W)
        errs.append(mean_err)
    mean_all = float(np.mean(errs))
    assert mean_all < 1.0, f"error medio de reproyección demasiado alto: {mean_all:.3f} px"


def test_encode_sample_stacks():
    row_ys = TE.make_row_ys(IMG_H, ROWS)
    lanes = [p for p, _ in _real_lanes(1)]
    slots = [0, 1, 2, 3][:len(lanes)]
    tg = TE.encode_sample(lanes, slots, row_ys, IMG_W, IMG_H)
    L = len(lanes)
    assert tg["xs"].shape == (L, ROWS) and tg["valid"].shape == (L, ROWS)
    assert tg["start"].shape == (L, 2) and tg["length"].shape == (L,)


def test_dataset_integration():
    """El Dataset con encode_targets=True debe devolver 'targets' coherentes."""
    ds = cd.CULaneDataset("val", seed=0, encode_targets=True, num_rows=ROWS)
    s = ds[0]
    assert "targets" in s
    tg = s["targets"]
    assert tg["xs"].shape[1] == ROWS
    assert tg["xs"].shape[0] == len(s["lanes"])


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
