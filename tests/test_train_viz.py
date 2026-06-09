"""Tests de work_dir + logging + GPU + visualizaciones por época (Paso 6.x).

    .\.venv\Scripts\python.exe tests\test_train_viz.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lanetr import paths
from lanetr.losses import HungarianMatcher
from lanetr.models import LaneTR
from lanetr.training import (
    ETA, EpochVisualizer, GPUMonitor, create_work_dir, format_eta,
)


def test_format_eta():
    assert format_eta(3725) == "1:02:05"
    assert format_eta(0) == "0:00:00"


def test_create_work_dir():
    cfg = {"name": "wd_test", "model": {"backbone": "dla34"}, "train": {"epochs": 1}}
    work = create_work_dir(cfg, base="outputs/_wd_test")
    try:
        assert (work / "config.yaml").exists()
        assert (work / "viz").is_dir()
        import yaml
        loaded = yaml.safe_load((work / "config.yaml").read_text(encoding="utf-8"))
        assert loaded["name"] == "wd_test"
    finally:
        shutil.rmtree(paths.project_root() / "outputs" / "_wd_test", ignore_errors=True)


def test_eta_and_gpu():
    eta = ETA(100)
    assert isinstance(eta.step(1), str)
    g = GPUMonitor("cuda" if torch.cuda.is_available() else "cpu")
    g.epoch_start()
    g.sample()
    summary = g.epoch_summary()
    assert isinstance(summary, dict)
    if torch.cuda.is_available():
        assert "mem_peak_gb" in summary


def test_visualizer_produces_four_pngs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = LaneTR(pretrained=False, num_queries=12, num_layers=2,
                   use_anchors=True, deformable=True).to(device)
    viz = EpochVisualizer(device, conf_thresh=0.4)
    out = paths.project_root() / "outputs" / "_viz_test"
    try:
        saved = viz.visualize(model, HungarianMatcher(), 0, out)
        assert len(saved) == 12, f"esperaba 12 figuras (4 tipos × 3 fotos), {len(saved)}"
        assert all(p.exists() for p in saved)
        for i in range(3):  # una de cada tipo por cada foto
            dimg = out / "epoch_000" / f"img{i}"
            for name in ["gt_vs_pred.png", "attention.png", "anchors.png", "matcher.png"]:
                assert (dimg / name).exists(), f"falta {dimg / name}"
    finally:
        shutil.rmtree(out, ignore_errors=True)


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
