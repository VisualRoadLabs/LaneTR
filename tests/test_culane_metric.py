"""Tests automáticos de la métrica F1 de CULane (Paso 2A).

    .\.venv\Scripts\python.exe tests\test_culane_metric.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lanetr import paths
from lanetr.data import culane_annotation as ann
from lanetr.metrics import culane as M
from lanetr.metrics import format as F


def _gt_lanes(n_images=15):
    """Carga carriles GT reales (coords originales 1640×590) de n imágenes de val."""
    lines = [l for l in (paths.list_dir() / "val_gt.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for line in lines[:n_images]:
        image_rel, _, _ = ann.parse_gt_line(line)
        lanes = M.load_culane_img_data(str(ann.lines_path_for_image(image_rel)))
        out.append(lanes)
    return out


def test_perfect_prediction_is_f1_100():
    """pred == GT -> TP=todos, FP=FN=0, F1=1.0."""
    annos = _gt_lanes(15)
    res = M.evaluate(annos, annos)[0.5]
    assert res["FP"] == 0 and res["FN"] == 0
    assert abs(res["F1"] - 1.0) < 1e-9, res


def test_empty_prediction_all_fn():
    annos = _gt_lanes(5)
    empty = [[] for _ in annos]
    res = M.evaluate(empty, annos)[0.5]
    total_lanes = sum(len(a) for a in annos)
    assert res["TP"] == 0 and res["FP"] == 0 and res["FN"] == total_lanes


def test_spurious_prediction_all_fp():
    annos = _gt_lanes(5)
    res = M.evaluate(annos, [[] for _ in annos])[0.5]
    total_lanes = sum(len(a) for a in annos)
    assert res["TP"] == 0 and res["FN"] == 0 and res["FP"] == total_lanes


def test_small_shift_still_tp():
    """Un desplazamiento de 5 px mantiene IoU>0.5 (sigue siendo TP)."""
    annos = _gt_lanes(8)
    preds = [[lane + np.array([5.0, 0.0]) for lane in a] for a in annos]
    res = M.evaluate(preds, annos)[0.5]
    assert res["TP"] == sum(len(a) for a in annos), res


def test_large_shift_breaks_match():
    """Un desplazamiento de 60 px (> ancho 30) rompe el match: FP y FN."""
    annos = _gt_lanes(8)
    preds = [[lane + np.array([60.0, 0.0]) for lane in a] for a in annos]
    res = M.evaluate(preds, annos)[0.5]
    assert res["TP"] < sum(len(a) for a in annos)
    assert res["FP"] > 0 and res["FN"] > 0


def test_resized_to_orig_roundtrip():
    """Mapear original -> espacio modelo -> original recupera los puntos."""
    orig = np.array([[820.0, 590.0], [800.0, 400.0], [780.0, 290.0]], np.float64)
    sx, sy = F.IMG_W / F.ORIG_W, F.IMG_H / (F.ORIG_H - F.CUT_HEIGHT)
    resized = orig.copy()
    resized[:, 0] *= sx
    resized[:, 1] = (resized[:, 1] - F.CUT_HEIGHT) * sy
    back = F.resized_to_orig(resized)
    assert np.allclose(back, orig, atol=1e-6), f"\n{back}\nvs\n{orig}"


def test_threshold_075_stricter():
    """Con el mismo desplazamiento, F1@75 <= F1@50."""
    annos = _gt_lanes(10)
    preds = [[lane + np.array([12.0, 0.0]) for lane in a] for a in annos]
    res = M.evaluate(preds, annos, iou_thresholds=(0.5, 0.75))
    assert res[0.75]["F1"] <= res[0.5]["F1"] + 1e-9


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
