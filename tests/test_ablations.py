"""Tests de las ablations (Paso 6.4): geo_metric, matriz/comandos y colector de tabla.

    .\.venv\Scripts\python.exe tests\test_ablations.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch

from lanetr.losses import HungarianMatcher, LaneCriterion

import collect_results as CR
import run_ablations as RA

NQ, R = 12, 144
METRICS = ["laneiou", "lineiou", "distance"]


def _targets():
    def lane(x):
        return torch.full((R,), float(x))
    return [{"xs": torch.stack([lane(0.3), lane(0.7)]), "valid": torch.ones(2, R, dtype=torch.bool),
             "start_y": torch.ones(2), "length": torch.ones(2), "theta": torch.zeros(2)}]


def _pred(L=2, B=1):
    g = torch.Generator().manual_seed(0)
    return {"conf": torch.zeros(L, B, NQ), "xs": torch.rand(L, B, NQ, R, generator=g),
            "start_y": torch.ones(L, B, NQ), "length": torch.ones(L, B, NQ),
            "theta": torch.zeros(L, B, NQ)}


def test_geo_metric_in_criterion():
    targets, pred = _targets(), _pred()
    for m in METRICS:
        out = LaneCriterion(geo_metric=m)(pred, targets)
        assert torch.isfinite(out["total"]).all(), f"{m}: total no finito"


def test_geo_metric_in_matcher():
    tgt = _targets()[0]
    pb = {"conf": torch.zeros(NQ), "xs": torch.rand(NQ, R), "start_y": torch.ones(NQ),
          "length": torch.ones(NQ)}
    for m in METRICS:
        comps = HungarianMatcher(geo_metric=m).cost_components(pb, tgt)
        assert comps["total"].shape == (NQ, 2)
        assert torch.isfinite(comps["total"]).all(), f"{m}: coste no finito"


def test_ablation_matrix_main_first():
    assert RA.ABLATIONS[0][0] == "main", "el modelo principal debe ir primero"
    names = [n for n, _ in RA.ABLATIONS]
    for needed in ("geo_distance", "q4", "q20", "with_o2m", "no_deformable", "no_filter"):
        assert needed in names, f"falta la ablation {needed}"


def test_build_cmd():
    cmd = RA.build_cmd("configs/lanetr_culane.yaml", "geo_distance", {"loss.geo_metric": "distance"})
    assert "name=abl_geo_distance" in cmd
    assert "loss.geo_metric=distance" in cmd
    assert "--config" in cmd and "configs/lanetr_culane.yaml" in cmd


def test_collect_results():
    tmp = ROOT / "outputs" / "_abl_test"
    shutil.rmtree(tmp, ignore_errors=True)
    for name, f1 in [("abl_main", 0.78), ("abl_q4", 0.71)]:
        d = tmp / f"{name}_20260101_000000"
        d.mkdir(parents=True)
        (d / "results.json").write_text(json.dumps({
            "name": name, "test_F1": f1, "conf_thresh": 0.4, "best_f1_val": f1 - 0.01,
            "categories": {c: f1 for c in CR.CATS if c != "cross"} | {"cross": 1000},
        }), encoding="utf-8")
    try:
        rows = CR.collect(tmp)
        assert len(rows) == 2
        rows.sort(key=lambda r: (r.get("name") != "abl_main", -r.get("test_F1", 0)))
        assert rows[0]["name"] == "abl_main"  # main primero
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
