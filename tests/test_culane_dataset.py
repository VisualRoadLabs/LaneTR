"""Tests automáticos del Dataset/transforms de CULane (Paso 1C).

    .\.venv\Scripts\python.exe tests\test_culane_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from lanetr.data import culane_dataset as cd
from lanetr.data import transforms as T

IMG_W, IMG_H, CUT = 800, 320, 270
RNG = np.random.default_rng(0)


def _toy_sample():
    """Sample sintético con un carril de geometría conocida (coords originales)."""
    from PIL import Image
    pts = np.array([[820.0, 590.0], [820.0, 400.0], [820.0, 290.0]], np.float32)  # vertical en x=820
    return {
        "image": Image.new("RGB", (1640, 590), (0, 0, 0)),
        "lanes": [pts.copy()],
        "slots": [1],
        "existence": (0, 1, 0, 0),
        "meta": {},
    }


def test_crop_resize_point_mapping():
    s = T.CropResize(IMG_W, IMG_H, CUT)(_toy_sample(), RNG)
    sx, sy = IMG_W / 1640, IMG_H / (590 - CUT)
    got = s["lanes"][0]
    exp = np.array([[820 * sx, (590 - CUT) * sy], [820 * sx, (400 - CUT) * sy],
                    [820 * sx, (290 - CUT) * sy]], np.float32)
    assert np.allclose(got, exp, atol=1e-3), f"\n{got}\nvs\n{exp}"


def test_flip_mirrors_x_and_reverses_existence():
    s = T.CropResize(IMG_W, IMG_H, CUT)(_toy_sample(), RNG)
    x_before = s["lanes"][0][:, 0].copy()
    s = T.RandomHorizontalFlip(p=1.0)(s, RNG)
    assert np.allclose(s["lanes"][0][:, 0], (IMG_W - 1) - x_before)
    assert s["existence"] == (0, 0, 1, 0)  # (0,1,0,0) invertido
    assert s["slots"] == [2]


def test_affine_identity_keeps_points():
    s = T.CropResize(IMG_W, IMG_H, CUT)(_toy_sample(), RNG)
    before = [p.copy() for p in s["lanes"]]
    s = T.RandomAffine(degrees=0.0, scale=(1.0, 1.0), translate=(0.0, 0.0), p=1.0)(s, RNG)
    assert np.allclose(s["lanes"][0], before[0], atol=1e-3)


def test_affine_pure_translation():
    M = T._affine_matrix(0.0, 1.0, 12.0, -7.0, 400, 160)
    pts = np.array([[100.0, 50.0], [200.0, 80.0]], np.float32)
    out = T._apply_matrix_to_lanes([pts], M)[0]
    assert np.allclose(out, pts + np.array([12.0, -7.0]), atol=1e-3)


def test_normalize_shape_and_range():
    s = T.CropResize(IMG_W, IMG_H, CUT)(_toy_sample(), RNG)
    s = T.Normalize()(s, RNG)
    img = s["image"]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, IMG_H, IMG_W) and img.dtype == torch.float32
    assert img.abs().max().item() < 5.0  # normalizado, no en [0,255]


def test_dataset_val_item():
    ds = cd.CULaneDataset("val", seed=0)
    s = ds[0]
    assert s["image"].shape == (3, IMG_H, IMG_W)
    if s["existence"] is not None:
        assert len(s["lanes"]) == sum(s["existence"])


def test_train_uses_filtered_list():
    ds = cd.CULaneDataset("train")
    assert len(ds) == 55698, f"esperado 55698 (filtrada), obtenido {len(ds)}"


def test_dataloader_batches():
    dl = cd.build_dataloader("val", batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(dl))
    assert batch["image"].shape == (4, 3, IMG_H, IMG_W)
    assert len(batch["lanes"]) == 4 and len(batch["meta"]) == 4


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
