"""Tests de la asignación auxiliar uno-a-muchos (Paso 5.2).

    .\.venv\Scripts\python.exe tests\test_o2m.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lanetr.losses import HungarianMatcher, LaneCriterion

NQ, R = 12, 144


def _synthetic():
    gt_xs = torch.stack([torch.full((R,), 0.3), torch.full((R,), 0.7)])  # 2 GT
    tgt = {"xs": gt_xs, "valid": torch.ones(2, R, dtype=torch.bool),
           "start_y": torch.ones(2), "length": torch.ones(2)}
    torch.manual_seed(0)
    pred = {"conf": torch.zeros(NQ), "xs": torch.rand(NQ, R),
            "start_y": torch.ones(NQ), "length": torch.ones(NQ)}
    return pred, tgt


def test_one_to_many_assigns_k_per_gt():
    pred, tgt = _synthetic()
    q, g = HungarianMatcher().match_one_to_many(pred, tgt, k=4)
    # cada GT debe recibir varias queries (uno-a-muchos)
    counts = [int((g == gi).sum()) for gi in range(2)]
    assert all(c >= 2 for c in counts), f"un GT recibió pocas queries: {counts}"
    # cada query como mucho un GT
    assert len(q) == len(set(q.tolist())), "una query asignada a 2 GT"


def test_one_to_many_more_positives_than_one_to_one():
    pred, tgt = _synthetic()
    q1, _ = HungarianMatcher().match_one(pred, tgt)            # uno-a-uno: 2 positivos
    qm, _ = HungarianMatcher().match_one_to_many(pred, tgt, k=4)
    assert len(qm) > len(q1), f"o2m({len(qm)}) no tiene más positivos que 1-a-1({len(q1)})"


def test_empty_gt():
    pred, _ = _synthetic()
    empty = {"xs": torch.zeros(0, R), "valid": torch.zeros(0, R, dtype=torch.bool),
             "start_y": torch.zeros(0), "length": torch.zeros(0)}
    q, g = HungarianMatcher().match_one_to_many(pred, empty, k=4)
    assert len(q) == 0 and len(g) == 0


def test_criterion_with_o2m_runs():
    L, B = 4, 2
    def lane(x):
        return torch.full((R,), float(x))
    targets = [{"xs": torch.stack([lane(0.3), lane(0.6)]), "valid": torch.ones(2, R, dtype=torch.bool),
                "start_y": torch.ones(2), "length": torch.ones(2), "theta": torch.zeros(2)}
               for _ in range(B)]
    torch.manual_seed(0)
    pred = {"conf": torch.zeros(L, B, NQ), "xs": torch.rand(L, B, NQ, R),
            "start_y": torch.ones(L, B, NQ), "length": torch.ones(L, B, NQ),
            "theta": torch.zeros(L, B, NQ)}
    out = LaneCriterion(aux_one_to_many=True, o2m_k=4)(pred, targets)
    for k in ("cls", "iou", "xy", "ext", "total"):
        assert torch.isfinite(out[k]).all(), f"{k} no finito con o2m"


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
