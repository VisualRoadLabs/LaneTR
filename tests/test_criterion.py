"""Tests del criterion / pérdida total (Paso 4.3).

    .\.venv\Scripts\python.exe tests\test_criterion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lanetr.losses import LaneCriterion

L, B, NQ, R = 3, 2, 12, 144


def _targets():
    """2 imágenes, 2 carriles GT cada una (x constantes), extensión completa."""
    def lane(x):
        return torch.full((R,), float(x))
    t0 = {"xs": torch.stack([lane(0.30), lane(0.60)]), "valid": torch.ones(2, R, dtype=torch.bool),
          "start_y": torch.ones(2), "length": torch.ones(2), "theta": torch.zeros(2)}
    t1 = {"xs": torch.stack([lane(0.40), lane(0.70)]), "valid": torch.ones(2, R, dtype=torch.bool),
          "start_y": torch.ones(2), "length": torch.ones(2), "theta": torch.zeros(2)}
    return [t0, t1]


def _pred(targets, noise=0.0, seed=0):
    """Construye predicciones (L,B,NQ,...) que copian los GT en queries concretas."""
    g = torch.Generator().manual_seed(seed)
    xs = torch.rand(L, B, NQ, R, generator=g) * 0.1 + 0.45
    conf = torch.full((L, B, NQ), -4.0)
    place = [[(0, 3), (1, 8)], [(0, 1), (1, 5)]]  # por imagen: (gt, query)
    for b in range(B):
        for gi, qi in place[b]:
            lane = targets[b]["xs"][gi]
            if noise > 0:
                lane = lane + torch.randn(R, generator=g) * noise
            xs[:, b, qi] = lane
            conf[:, b, qi] = 4.0
    return {"conf": conf, "xs": xs,
            "start_y": torch.ones(L, B, NQ), "length": torch.ones(L, B, NQ),
            "theta": torch.zeros(L, B, NQ)}


def test_forward_keys_and_finite():
    targets = _targets()
    out = LaneCriterion()(_pred(targets), targets)
    for k in ("cls", "iou", "xy", "ext", "total"):
        assert k in out, f"falta {k}"
        assert torch.isfinite(out[k]).all(), f"{k} no finito"
    assert out["total"].item() >= 0


def test_perfect_better_than_noisy():
    """Predicciones que copian el GT deben dar MENOS pérdida que las ruidosas."""
    targets = _targets()
    crit = LaneCriterion()
    perfect = crit(_pred(targets, noise=0.0), targets)["total"].item()
    noisy = crit(_pred(targets, noise=0.05), targets)["total"].item()
    assert perfect < noisy, f"perfecto={perfect:.3f} no < ruidoso={noisy:.3f}"


def test_aux_layers_sum():
    """Con capas auxiliares, la pérdida suma sobre las L capas (mayor que solo la última)."""
    targets = _targets()
    pred = _pred(targets)
    with_aux = LaneCriterion(aux_layers=True)(pred, targets)["total"].item()
    last_only = LaneCriterion(aux_layers=False)(pred, targets)["total"].item()
    assert with_aux > last_only * 1.5, (with_aux, last_only)


def test_empty_gt_image():
    """Una imagen sin carriles: solo pérdida de clasificación (todo 'no-carril')."""
    targets = _targets()
    targets[1] = {"xs": torch.zeros(0, R), "valid": torch.zeros(0, R, dtype=torch.bool),
                  "start_y": torch.zeros(0), "length": torch.zeros(0), "theta": torch.zeros(0)}
    out = LaneCriterion()(_pred(_targets()), targets)
    assert torch.isfinite(out["total"]).all()


def test_grad_flows_to_model():
    """La pérdida debe propagar gradiente hasta los pesos del modelo real."""
    from lanetr.models import LaneTR
    torch.manual_seed(0)
    model = LaneTR(pretrained=False, num_queries=NQ, num_layers=L, num_rows=R)
    targets = _targets()
    x = torch.randn(B, 3, 320, 800)
    pred = model(x)
    out = LaneCriterion()(pred, targets)
    out["total"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)


def test_smoothness_term_optional():
    targets = _targets()
    out = LaneCriterion(w_smooth=0.1)(_pred(targets), targets)
    assert "smooth" in out and torch.isfinite(out["smooth"]).all()


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
