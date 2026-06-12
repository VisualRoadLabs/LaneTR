"""Tests de formas del backbone + FPN (Paso 3.1).

    .\.venv\Scripts\python.exe tests\test_backbone_fpn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lanetr.models import FPN, build_backbone

B, H, W = 2, 320, 800
BACKBONE = "dla34"
# strides 8/16/32 sobre 320×800
EXPECTED = [(40, 100), (20, 50), (10, 25)]
EXPECTED_CH = [128, 256, 512]


def test_backbone_output_shapes():
    bb = build_backbone(BACKBONE, pretrained=False).eval()
    assert bb.out_channels == EXPECTED_CH, bb.out_channels
    assert bb.strides == [8, 16, 32], bb.strides
    x = torch.randn(B, 3, H, W)
    with torch.no_grad():
        feats = bb(x)
    assert len(feats) == 3
    for f, ch, (h, w) in zip(feats, bb.out_channels, EXPECTED):
        assert f.shape == (B, ch, h, w), f"{tuple(f.shape)} != {(B, ch, h, w)}"


def test_backbone_has_real_weights():
    """DLA-34 debe tener ~15M de parámetros (no un stub)."""
    bb = build_backbone(BACKBONE, pretrained=False)
    n = sum(p.numel() for p in bb.parameters())
    assert n > 10_000_000, f"solo {n} parámetros; ¿se cargó bien el backbone?"


def test_fpn_output_shapes():
    bb = build_backbone(BACKBONE, pretrained=False).eval()
    fpn = FPN(bb.out_channels, out_channels=256).eval()
    x = torch.randn(B, 3, H, W)
    with torch.no_grad():
        pyr = fpn(bb(x))
    assert len(pyr) == 3
    for p, (h, w) in zip(pyr, EXPECTED):
        assert p.shape == (B, 256, h, w), f"{tuple(p.shape)} != {(B, 256, h, w)}"


def test_no_nans_and_grad_flows():
    bb = build_backbone(BACKBONE, pretrained=False)
    fpn = FPN(bb.out_channels, 256)
    x = torch.randn(1, 3, H, W, requires_grad=True)
    pyr = fpn(bb(x))
    loss = sum(p.float().mean() for p in pyr)
    loss.backward()
    assert all(torch.isfinite(p).all() for p in pyr), "hay NaN/Inf en la pirámide"
    assert x.grad is not None and torch.isfinite(x.grad).all(), "el gradiente no fluye"


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
