"""Tests de las cabezas (3.3) y del modelo completo LaneTR (3.4).

    .\.venv\Scripts\python.exe tests\test_head_lanetr.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lanetr.models import LaneHead, LaneTR, decode_lanes

L, B, NQ, D, R = 6, 2, 12, 256, 144


def _hs():
    return torch.randn(L, B, NQ, D)


# --------------------------------------------------------------------------- #
# 3.3 — cabezas
# --------------------------------------------------------------------------- #
def test_head_output_shapes():
    head = LaneHead(D, R).eval()
    with torch.no_grad():
        pred = head(_hs())
    assert pred["conf"].shape == (L, B, NQ)
    assert pred["xs"].shape == (L, B, NQ, R)
    assert pred["start_y"].shape == (L, B, NQ)
    assert pred["length"].shape == (L, B, NQ)
    assert pred["theta"].shape == (L, B, NQ)


def test_head_output_ranges():
    head = LaneHead(D, R).eval()
    with torch.no_grad():
        pred = head(_hs())
    for key in ("xs", "start_y", "length"):
        t = pred[key]
        assert t.min() >= 0.0 and t.max() <= 1.0, f"{key} fuera de [0,1]"


def test_decode_shapes():
    head = LaneHead(D, R).eval()
    with torch.no_grad():
        pred = head(_hs())
    # forzar confianza alta para que decodifique todas
    pred["conf"] = torch.full_like(pred["conf"], 10.0)
    lanes = decode_lanes(pred, conf_thresh=0.5, num_rows=R)
    assert len(lanes) == B
    for per_img in lanes:
        for lane in per_img:
            assert lane["points"].ndim == 2 and lane["points"].shape[1] == 2
            assert lane["points"].shape[0] <= R


def test_decode_threshold_filters():
    head = LaneHead(D, R).eval()
    with torch.no_grad():
        pred = head(_hs())
    pred["conf"] = torch.full_like(pred["conf"], -10.0)  # confianza ~0
    lanes = decode_lanes(pred, conf_thresh=0.5, num_rows=R)
    assert all(len(per_img) == 0 for per_img in lanes)


# --------------------------------------------------------------------------- #
# 3.4 — modelo completo
# --------------------------------------------------------------------------- #
def test_lanetr_forward_shapes():
    model = LaneTR(pretrained=False, num_queries=NQ, num_layers=L, num_rows=R).eval()
    x = torch.randn(B, 3, 320, 800)
    with torch.no_grad():
        pred = model(x)
    assert pred["conf"].shape == (L, B, NQ)
    assert pred["xs"].shape == (L, B, NQ, R)


def test_lanetr_predict_runs():
    model = LaneTR(pretrained=False, num_queries=NQ, num_layers=L, num_rows=R)
    x = torch.randn(1, 3, 320, 800)
    lanes = model.predict(x, conf_thresh=None)  # None -> todas las queries
    assert len(lanes) == 1
    assert len(lanes[0]) <= NQ


def test_lanetr_grad_flows():
    model = LaneTR(pretrained=False, num_queries=NQ, num_layers=L, num_rows=R)
    x = torch.randn(1, 3, 320, 800)
    pred = model(x)
    loss = pred["conf"].mean() + pred["xs"].mean() + pred["start_y"].mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


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
