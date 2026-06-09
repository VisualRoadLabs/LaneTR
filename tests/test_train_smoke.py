"""Tests del entrenamiento (Paso 6.2): carga de config, overrides y smoke de 5 iteraciones.

    .\.venv\Scripts\python.exe tests\test_train_smoke.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from lanetr.config import load_config
import train as T


def test_config_defaults_and_overrides():
    cfg = load_config(None, {"loss": {"aux_one_to_many": True}, "model": {"deformable": False}})
    assert cfg["loss"]["aux_one_to_many"] is True
    assert cfg["model"]["deformable"] is False
    assert cfg["optim"]["lr"] == 2.0e-4              # default preservado
    assert cfg["loss"]["w_iou"] == 4.0


def test_yaml_config_loads():
    cfg = load_config(ROOT / "configs" / "lanetr_culane.yaml")
    assert cfg["model"]["backbone"] == "dla34"
    assert cfg["model"]["use_anchors"] and cfg["model"]["deformable"]
    assert cfg["train"]["grad_clip"] == 0.1
    assert cfg["ema"]["decay"] == 0.9999


def test_parse_overrides_casting():
    ov = T._parse_overrides(["model.deformable=false", "optim.lr=1e-4", "model.num_queries=20"])
    assert ov["model"]["deformable"] is False
    assert ov["optim"]["lr"] == 1e-4
    assert ov["model"]["num_queries"] == 20


def test_smoke_training_runs():
    """5 iteraciones reales (modelo pequeño, sin pesos preentrenados) -> pérdida finita."""
    cfg = load_config(None, {
        "model": {"pretrained": False, "num_layers": 2, "use_anchors": True, "deformable": True},
        "data": {"num_workers": 0},
        "train": {"log_interval": 1},
        "ema": {"enabled": True},
    })
    out = T.train(cfg, smoke=True)
    assert out["iters"] >= 5
    assert math.isfinite(out["last"]["total"]), out["last"]
    assert out["last"]["total"] > 0


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
