"""Tests de las utilidades de entrenamiento (Paso 6.1).

    .\.venv\Scripts\python.exe tests\test_training_utils.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from lanetr.models import LaneTR
from lanetr.training import (
    FrozenBatchNorm2d, ModelEMA, build_optimizer, build_scheduler, freeze_batchnorm,
)
from lanetr.training.optim import param_groups


# --------------------------------------------------------------------------- #
# FrozenBatchNorm
# --------------------------------------------------------------------------- #
def test_frozen_bn_matches_eval_bn():
    torch.manual_seed(0)
    bn = nn.BatchNorm2d(8)
    bn.running_mean.normal_()
    bn.running_var.uniform_(0.5, 1.5)
    bn.weight.data.normal_()
    bn.bias.data.normal_()
    bn.eval()
    x = torch.randn(2, 8, 5, 5)
    fbn = freeze_batchnorm(nn.Sequential(bn))[0]
    assert isinstance(fbn, FrozenBatchNorm2d)
    assert torch.allclose(bn(x), fbn(x), atol=1e-5)


def test_freeze_replaces_all_bn():
    net = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(),
                        nn.Sequential(nn.BatchNorm2d(8)))
    freeze_batchnorm(net)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in net.modules())
    # FrozenBN no tiene parámetros aprendibles
    fbn = [m for m in net.modules() if isinstance(m, FrozenBatchNorm2d)][0]
    assert len(list(fbn.parameters())) == 0


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #
def test_ema_tracks_and_is_frozen():
    torch.manual_seed(0)
    model = LaneTR(pretrained=False, num_queries=12, num_layers=2, use_anchors=True)
    ema = ModelEMA(model, decay=0.9, tau=1.0)
    assert all(not p.requires_grad for p in ema.ema.parameters())
    # cambia los pesos del modelo y actualiza el EMA: debe moverse hacia el modelo
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    before = next(iter(ema.ema.state_dict().values())).clone().float().mean()
    for _ in range(5):
        ema.update(model)
    after = next(iter(ema.ema.state_dict().values())).float().mean()
    assert after != before, "el EMA no se actualizó"


def test_ema_warmup_decay():
    """El decay efectivo arranca bajo (warmup) y sube hacia el objetivo."""
    import math
    decay, tau = 0.9999, 2000.0
    d1 = decay * (1 - math.exp(-1 / tau))
    d_big = decay * (1 - math.exp(-10000 / tau))
    assert d1 < 0.01 and d_big > 0.99 * decay


# --------------------------------------------------------------------------- #
# Optimizador (grupos de parámetros)
# --------------------------------------------------------------------------- #
def test_param_groups_differentiated_lr():
    model = LaneTR(pretrained=False, num_queries=12, num_layers=2,
                   use_anchors=True, deformable=True)
    groups = param_groups(model, lr=2e-4, backbone_mult=0.1, slow_mult=0.1)
    by_name = {g["name"]: g for g in groups}
    assert abs(by_name["backbone"]["lr"] - 2e-5) < 1e-9
    assert abs(by_name["slow"]["lr"] - 2e-5) < 1e-9
    assert abs(by_name["rest"]["lr"] - 2e-4) < 1e-9
    # cobertura: cada parámetro entrenable aparece exactamente una vez
    n_groups = sum(len(g["params"]) for g in groups)
    n_model = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_groups == n_model, f"{n_groups} != {n_model} (params no cubiertos o duplicados)"


def test_build_optimizer_and_scheduler():
    model = LaneTR(pretrained=False, num_queries=12, num_layers=2, use_anchors=True)
    opt = build_optimizer(model, lr=2e-4, weight_decay=1e-4)
    sched = build_scheduler(opt, total_iters=1000, warmup_iters=100)
    lrs = []
    for _ in range(1000):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[50] < lrs[99], "el warmup debe subir el lr"
    assert abs(lrs[99] - 2e-4) < 2e-5, "al final del warmup ~ lr base"
    assert lrs[-1] < lrs[100], "tras el warmup el cosine debe bajar"


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
