"""Tests del bucle de evaluación F1 (Paso 6.3).

    .\.venv\Scripts\python.exe tests\test_eval.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lanetr import paths
from lanetr.metrics import evaluate as ev
from lanetr.models import LaneTR

N = 8


def _model():
    torch.manual_seed(0)
    return LaneTR(pretrained=False, num_queries=12, num_layers=2,
                  use_anchors=True, deformable=True)


def test_categories_files_exist():
    for name, lf in ev.CATEGORIES.items():
        assert (paths.list_dir() / lf).exists(), f"falta {lf}"


def test_infer_and_f1_runs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw, annos = ev.infer(_model().to(device), "val_gt.txt", device, batch_size=4,
                          num_workers=0, max_images=N)
    assert len(raw) == N and len(annos) == N
    res = ev.f1_at_threshold(raw, annos, 0.5)
    assert math.isfinite(res["F1"]) and 0.0 <= res["F1"] <= 1.0
    # raw guarda TODAS las queries con su confianza y puntos en coords originales
    assert all("conf" in r and "points" in r for img in raw for r in img)


def test_gt_as_prediction_is_f1_1():
    """Sanity del plumbing: si las 'predicciones' son el propio GT, F1=1.0."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, annos = ev.infer(_model().to(device), "val_gt.txt", device, batch_size=4,
                        num_workers=0, max_images=N)
    raw_gt = [[{"conf": 1.0, "points": lane} for lane in img] for img in annos]
    res = ev.f1_at_threshold(raw_gt, annos, 0.5)
    assert abs(res["F1"] - 1.0) < 1e-9, res


def test_calibrate_returns_threshold():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best, scores = ev.calibrate_threshold(_model().to(device), "val_gt.txt", device,
                                          thresholds=[0.2, 0.4, 0.6], batch_size=4,
                                          num_workers=0, max_images=N)
    assert best in (0.2, 0.4, 0.6)
    assert all("F1" in s for s in scores.values())


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
