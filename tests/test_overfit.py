"""Test rápido de optimizabilidad del criterion (Paso 4.4).

En vez de entrenar el modelo entero (lento), optimiza unos parámetros libres que hacen de
"salida del modelo" para comprobar que la pérdida es minimizable de extremo a extremo: baja
mucho y la confianza de las queries-carril sube por encima de 0.5. Corre en segundos (CPU).

    .\.venv\Scripts\python.exe tests\test_overfit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from lanetr.losses import LaneCriterion

L, B, NQ, R = 2, 2, 12, 144


class FixedMatcher:
    @torch.no_grad()
    def match(self, pred, targets):
        return [(torch.arange(t["xs"].shape[0]), torch.arange(t["xs"].shape[0])) for t in targets]


def _targets():
    def lane(x):
        return torch.full((R,), float(x))
    t0 = {"xs": torch.stack([lane(0.3), lane(0.7)]), "valid": torch.ones(2, R, dtype=torch.bool),
          "start_y": torch.ones(2), "length": torch.ones(2), "theta": torch.zeros(2)}
    t1 = {"xs": torch.stack([lane(0.4), lane(0.8)]), "valid": torch.ones(2, R, dtype=torch.bool),
          "start_y": torch.ones(2), "length": torch.ones(2), "theta": torch.zeros(2)}
    return [t0, t1]


def test_loss_is_minimizable():
    torch.manual_seed(0)
    tg = _targets()
    raw_xs = nn.Parameter(torch.zeros(L, B, NQ, R))      # sigmoid(0)=0.5 (centrado)
    conf = nn.Parameter(torch.zeros(L, B, NQ))
    raw_sy = nn.Parameter(torch.zeros(L, B, NQ))
    raw_ln = nn.Parameter(torch.zeros(L, B, NQ))
    theta = torch.zeros(L, B, NQ)
    opt = torch.optim.Adam([raw_xs, conf, raw_sy, raw_ln], lr=0.05)
    crit = LaneCriterion(matcher=FixedMatcher(), w_xy=1.0, focal_alpha=0.5)

    def pred():
        return {"conf": conf, "xs": raw_xs.sigmoid(), "start_y": raw_sy.sigmoid(),
                "length": raw_ln.sigmoid(), "theta": theta}

    loss0 = crit(pred(), tg)["total"].item()
    for _ in range(150):
        out = crit(pred(), tg)
        opt.zero_grad()
        out["total"].backward()
        opt.step()
    lossf = crit(pred(), tg)["total"].item()

    assert lossf < 0.25 * loss0, f"la pérdida no bajó lo suficiente: {loss0:.2f} -> {lossf:.2f}"
    # la confianza de las queries-carril (0 y 1) debe superar 0.5
    p = pred()["conf"].sigmoid()[-1]  # última capa (B,NQ)
    assert p[:, 0].mean() > 0.5 and p[:, 1].mean() > 0.5, "la confianza no subió"
    print(f"    pérdida {loss0:.2f} -> {lossf:.2f}; conf carriles ~{p[:, :2].mean():.2f}")


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
