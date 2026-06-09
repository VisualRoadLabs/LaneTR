"""Tests del prior posicional / anclas (Paso 5.1).

    .\.venv\Scripts\python.exe tests\test_anchors.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lanetr.data import target_encoding as TE
from lanetr.models import LaneAnchors, LaneTR

NQ, D, R, B = 12, 256, 144, 2


def test_anchor_init_is_spread():
    a = LaneAnchors(NQ, D)
    sx = a.anchors[:, 0]
    # el abanico llega a los laterales (bordes) pero SIN salirse de la imagen [0,1]
    assert sx.min() >= 0.0 and sx.max() <= 1.0, f"el abanico se sale de la imagen: [{sx[0]:.2f},{sx[-1]:.2f}]"
    assert sx[0] < 0.05 and sx[-1] > 0.95, "el abanico no llega a los laterales"
    assert (sx.diff() > 0).all(), "los start_x deben estar repartidos crecientes"


def test_prior_xs_shape_and_spread():
    a = LaneAnchors(NQ, D)
    row_ys = torch.tensor(TE.make_row_ys(320, R))
    prior = a.prior_xs(row_ys, 320)
    assert prior.shape == (NQ, R)
    # en la fila de abajo (cerca del coche) las queries deben estar claramente separadas
    bottom = prior[:, -1]
    assert bottom.std() > 0.1, "las anclas no están repartidas por abajo"


def test_pos_embed_shape():
    a = LaneAnchors(NQ, D)
    assert a.pos_embed().shape == (NQ, D)


def test_lanetr_with_anchors_runs():
    model = LaneTR(pretrained=False, num_queries=NQ, num_layers=3, num_rows=R, use_anchors=True).eval()
    x = torch.randn(B, 3, 320, 800)
    with torch.no_grad():
        pred = model(x)
    assert pred["xs"].shape == (3, B, NQ, R)
    assert pred["conf"].shape == (3, B, NQ)


def test_initial_xs_equals_prior():
    """Con delta≈0 al inicio, las xs predichas ≈ la línea-prior de cada ancla."""
    model = LaneTR(pretrained=False, num_queries=NQ, num_layers=3, num_rows=R, use_anchors=True).eval()
    x = torch.randn(1, 3, 320, 800)
    with torch.no_grad():
        pred = model(x)
    prior = model.anchors.prior_xs(model.row_ys, model.img_h)  # (NQ,R)
    assert torch.allclose(pred["xs"][-1, 0], prior, atol=1e-4), "xs inicial != prior"


def test_predictions_are_spread_with_anchors():
    """Con anclas, las queries predicen carriles DISTINTOS (no todos centrados)."""
    model = LaneTR(pretrained=False, num_queries=NQ, num_layers=3, num_rows=R, use_anchors=True).eval()
    with torch.no_grad():
        xs = model(torch.randn(1, 3, 320, 800))["xs"][-1, 0]  # (NQ,R)
    bottom = xs[:, -1]
    assert bottom.std() > 0.1, "las predicciones no están repartidas (anclas sin efecto)"


def test_ext_prior_values():
    """El prior de extensión nace 'abajo + largo' (start_y≈0.98, length≈0.9)."""
    a = LaneAnchors(NQ, D)
    ext = a.ext_prior()
    assert ext.shape == (NQ, 2)
    assert torch.allclose(ext[:, 0], torch.full((NQ,), 0.98), atol=1e-3)
    assert torch.allclose(ext[:, 1], torch.full((NQ,), 0.9), atol=1e-3)


def test_initial_extent_equals_prior():
    """Con delta≈0, la extensión predicha ≈ el prior del ancla (abanico llega a toda la imagen)."""
    model = LaneTR(pretrained=False, num_queries=NQ, num_layers=3, num_rows=R, use_anchors=True).eval()
    with torch.no_grad():
        pred = model(torch.randn(1, 3, 320, 800))
    assert torch.allclose(pred["start_y"][-1, 0], torch.full((NQ,), 0.98), atol=1e-3)
    assert torch.allclose(pred["length"][-1, 0], torch.full((NQ,), 0.9), atol=1e-3)


def test_anchor_lanes_span_image():
    """Los carriles iniciales (con extensión anclada) cubren casi toda la altura."""
    from lanetr.models import decode_lanes
    model = LaneTR(pretrained=False, num_queries=NQ, num_layers=3, num_rows=R, use_anchors=True).eval()
    with torch.no_grad():
        lanes = decode_lanes(model(torch.randn(1, 3, 320, 800)), conf_thresh=None, num_rows=R)[0]
    ys = [p for lane in lanes for p in lane["points"][:, 1]]
    assert min(ys) < 40 and max(ys) > 280, "los carriles no llegan a (casi) toda la imagen (0..319)"


def test_anchor_grad_flows():
    model = LaneTR(pretrained=False, num_queries=NQ, num_layers=2, num_rows=R, use_anchors=True)
    pred = model(torch.randn(1, 3, 320, 800))
    pred["xs"].mean().backward()
    assert model.anchors.anchors.grad is not None
    assert torch.isfinite(model.anchors.anchors.grad).all()


def test_default_model_unchanged():
    """El modelo por defecto (sin anclas) sigue dando xs en [0,1]."""
    model = LaneTR(pretrained=False, num_queries=NQ, num_layers=2, num_rows=R).eval()
    with torch.no_grad():
        xs = model(torch.randn(1, 3, 320, 800))["xs"]
    assert xs.min() >= 0.0 and xs.max() <= 1.0


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
