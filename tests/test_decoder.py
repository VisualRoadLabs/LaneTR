"""Tests de formas del decoder transformer (Paso 3.2).

    .\.venv\Scripts\python.exe tests\test_decoder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lanetr.models import LaneDecoder, PositionEmbeddingSine

B, D, NQ, NL = 2, 256, 12, 6
# pirámide FPN de juguete (strides 8/16/32 sobre 320×800)
LEVELS = [(D, 40, 100), (D, 20, 50), (D, 10, 25)]
TOTAL_TOKENS = 40 * 100 + 20 * 50 + 10 * 25  # 5250


def _feats():
    return [torch.randn(B, c, h, w) for c, h, w in LEVELS]


def test_pos_encoding_shape():
    pe = PositionEmbeddingSine(D // 2)
    pos = pe(B, 40, 100, device="cpu")
    assert pos.shape == (B, D, 40, 100), pos.shape


def test_decoder_output_shape():
    dec = LaneDecoder(d_model=D, num_queries=NQ, num_layers=NL).eval()
    with torch.no_grad():
        hs = dec(_feats())
    assert hs.shape == (NL, B, NQ, D), hs.shape


def test_memory_token_count_and_attn():
    dec = LaneDecoder(d_model=D, num_queries=NQ, num_layers=NL).eval()
    with torch.no_grad():
        hs, attns, shapes = dec(_feats(), need_attn=True)
    assert len(attns) == NL
    assert attns[-1].shape == (B, NQ, TOTAL_TOKENS), attns[-1].shape
    assert shapes == [(40, 100), (20, 50), (10, 25)]
    # la atención de cada query suma 1 (softmax sobre la memoria)
    assert torch.allclose(attns[-1].sum(-1), torch.ones(B, NQ), atol=1e-4)


def test_queries_are_distinct():
    """Las queries no deben colapsar al mismo vector (deben poder especializarse)."""
    dec = LaneDecoder(d_model=D, num_queries=NQ, num_layers=NL).eval()
    with torch.no_grad():
        out = dec(_feats())[-1]  # (B, NQ, D)
    # desviación entre queries claramente > 0
    assert out.std(dim=1).mean().item() > 1e-3


def test_no_nans_and_grad_flows():
    dec = LaneDecoder(d_model=D, num_queries=NQ, num_layers=NL)
    feats = [f.requires_grad_(True) for f in _feats()]
    hs = dec(feats)
    hs.float().mean().backward()
    assert torch.isfinite(hs).all()
    assert all(f.grad is not None and torch.isfinite(f.grad).all() for f in feats)


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
