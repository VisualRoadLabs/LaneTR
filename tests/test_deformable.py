"""Tests de la atención deformable (Paso 5.3).

    .\.venv\Scripts\python.exe tests\test_deformable.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lanetr.models import LaneTR, MSDeformAttn
from lanetr.models.deform_attn import ms_deform_attn_core_pytorch

B, D, NQ, R = 2, 256, 12, 144
SHAPES = [(40, 100), (20, 50), (10, 25)]
S = sum(h * w for h, w in SHAPES)
NHEADS, NLEVELS, NPOINTS = 8, 3, 4


def test_msdeformattn_output_shape():
    attn = MSDeformAttn(D, NLEVELS, NHEADS, NPOINTS).eval()
    query = torch.randn(B, NQ, D)
    ref = torch.rand(B, NQ, NLEVELS, 2)
    value = torch.randn(B, S, D)
    with torch.no_grad():
        out = attn(query, ref, value, SHAPES)
    assert out.shape == (B, NQ, D), out.shape


def test_attention_weights_sum_to_one():
    attn = MSDeformAttn(D, NLEVELS, NHEADS, NPOINTS).eval()
    query = torch.randn(B, NQ, D)
    ref = torch.rand(B, NQ, NLEVELS, 2)
    value = torch.randn(B, S, D)
    with torch.no_grad():
        _, _, w = attn(query, ref, value, SHAPES, return_sampling=True)
    # softmax sobre (n_levels*n_points) por cabeza -> suma 1
    assert torch.allclose(w.sum(dim=(-2, -1)), torch.ones(B, NQ, NHEADS), atol=1e-4)


def test_core_matches_module_path():
    """El núcleo devuelve algo finito con formas coherentes."""
    value = torch.randn(B, S, NHEADS, D // NHEADS)
    loc = torch.rand(B, NQ, NHEADS, NLEVELS, NPOINTS, 2)
    w = torch.rand(B, NQ, NHEADS, NLEVELS, NPOINTS)
    out = ms_deform_attn_core_pytorch(value, SHAPES, loc, w)
    assert out.shape == (B, NQ, D) and torch.isfinite(out).all()


def test_lanetr_deformable_runs():
    model = LaneTR(pretrained=False, num_queries=NQ, num_layers=3, num_rows=R,
                   use_anchors=True, deformable=True).eval()
    with torch.no_grad():
        pred = model(torch.randn(B, 3, 320, 800))
    assert pred["xs"].shape == (3, B, NQ, R)
    assert pred["conf"].shape == (3, B, NQ)


def test_deformable_grad_flows():
    model = LaneTR(pretrained=False, num_queries=NQ, num_layers=2, num_rows=R,
                   use_anchors=True, deformable=True)
    pred = model(torch.randn(1, 3, 320, 800))
    pred["xs"].mean().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)
    # el gradiente llega a los offsets de muestreo de la cross-attn deformable
    off = model.decoder.layers[0].cross_attn.sampling_offsets
    assert off.weight.grad is not None


def test_dense_model_still_works():
    """El modelo denso (deformable=False) sigue funcionando igual."""
    model = LaneTR(pretrained=False, num_queries=NQ, num_layers=2, num_rows=R).eval()
    with torch.no_grad():
        pred = model(torch.randn(1, 3, 320, 800))
    assert pred["xs"].shape == (2, 1, NQ, R)


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
