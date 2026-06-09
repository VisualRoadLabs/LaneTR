"""Tests del matcher húngaro (Paso 4.2).

    .\.venv\Scripts\python.exe tests\test_matcher.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lanetr.losses import HungarianMatcher

NQ, R = 12, 144


def _const_lane(x, R=R):
    return torch.full((R,), float(x))


def _synthetic():
    """3 carriles GT (x=0.25/0.5/0.75) y 12 predicciones; las queries 5,2,9 copian los GT."""
    gt_xs = torch.stack([_const_lane(0.25), _const_lane(0.5), _const_lane(0.75)])  # (3,R)
    gt_valid = torch.ones(3, R, dtype=torch.bool)
    tgt = {"xs": gt_xs, "valid": gt_valid,
           "start_y": torch.ones(3), "length": torch.ones(3)}

    torch.manual_seed(0)
    pred_xs = torch.rand(NQ, R) * 0.1 + 0.45        # ruido cerca del centro
    pred_xs[5] = gt_xs[0]; pred_xs[2] = gt_xs[1]; pred_xs[9] = gt_xs[2]
    conf = torch.full((NQ,), -3.0)
    conf[[5, 2, 9]] = 3.0
    pred = {"conf": conf, "xs": pred_xs,
            "start_y": torch.ones(NQ), "length": torch.ones(NQ)}
    return pred, tgt, {0: 5, 1: 2, 2: 9}


def test_recovers_correct_assignment():
    pred, tgt, expected = _synthetic()
    m = HungarianMatcher(w_cls=2.0, w_iou=2.0, w_ext=0.5)
    q, g = m.match_one(pred, tgt)
    got = {int(gi): int(qi) for qi, gi in zip(q, g)}
    assert got == expected, f"{got} != {expected}"


def test_one_to_one():
    pred, tgt, _ = _synthetic()
    q, g = HungarianMatcher().match_one(pred, tgt)
    assert len(q) == 3 and len(g) == 3
    assert len(set(q.tolist())) == 3, "una query emparejada con 2 GT"
    assert sorted(g.tolist()) == [0, 1, 2], "un GT emparejado con 2 queries"


def test_no_gt_returns_empty():
    pred, _, _ = _synthetic()
    empty_tgt = {"xs": torch.zeros(0, R), "valid": torch.zeros(0, R, dtype=torch.bool),
                 "start_y": torch.zeros(0), "length": torch.zeros(0)}
    q, g = HungarianMatcher().match_one(pred, empty_tgt)
    assert len(q) == 0 and len(g) == 0


def test_cost_components_shapes():
    pred, tgt, _ = _synthetic()
    comps = HungarianMatcher().cost_components(pred, tgt)
    for key in ("cls", "iou", "xy", "ext", "total"):
        assert comps[key].shape == (NQ, 3), f"{key}: {comps[key].shape}"
    # el coste IoU en las celdas correctas (query==gt) debe ser ~0 (1 - IoU≈1)
    assert comps["iou"][5, 0] < 0.05 and comps["iou"][2, 1] < 0.05 and comps["iou"][9, 2] < 0.05


def test_batch_match():
    pred, tgt, _ = _synthetic()
    batch_pred = {k: torch.stack([v, v]) for k, v in pred.items()}  # B=2
    res = HungarianMatcher().match(batch_pred, [tgt, tgt])
    assert len(res) == 2
    for q, g in res:
        assert len(q) == 3 and len(g) == 3


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
