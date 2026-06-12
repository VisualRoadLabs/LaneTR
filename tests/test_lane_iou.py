"""Tests de LaneIoU diferenciable (Paso 4.1).

    .\.venv\Scripts\python.exe tests\test_lane_iou.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from lanetr import paths
from lanetr.data import culane_annotation as ann
from lanetr.data import target_encoding as TE
from lanetr.data import transforms as T
from lanetr.losses import lane_iou as LI
from lanetr.metrics import culane as Mc
from lanetr.metrics import format as F

R, IMG_W, IMG_H = 144, 800, 320
ROW_YS = TE.make_row_ys(IMG_H, R)


def real_lanes(n_images=12):
    """Devuelve (xs (R,), valid (R,)) de carriles reales en el espacio 800×320."""
    lines = [l for l in (paths.list_dir() / "val_gt.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    cr = T.CropResize(IMG_W, IMG_H, 270)
    out = []
    for line in lines[:n_images]:
        image_rel, seg, ex = ann.parse_gt_line(line)
        a = ann.load_annotation(image_rel, ex, seg)
        sample = {"image": Image.new("RGB", (1640, 590)),
                  "lanes": [l.points.copy() for l in a.lanes], "slots": [l.slot for l in a.lanes],
                  "existence": a.existence, "meta": {}}
        sample = cr(sample, np.random.default_rng(0))
        for pts in sample["lanes"]:
            t = TE.encode_lane(pts, ROW_YS, IMG_W, IMG_H)
            if t.valid.sum() >= 10:
                out.append((t.xs.astype(np.float32), t.valid))
    return out


def mask_iou(pred_xs, gt_xs, valid):
    """IoU de máscara (la MÉTRICA): rasteriza ambos carriles a 30 px y calcula IoU de píxeles."""
    pa = F.resized_to_orig(TE.decode_lane(pred_xs, valid, ROW_YS, IMG_W))
    pb = F.resized_to_orig(TE.decode_lane(gt_xs, valid, ROW_YS, IMG_W))
    if len(pa) < 2 or len(pb) < 2:
        return 0.0
    ma = Mc.draw_lane(Mc.interp(pa)) > 0
    mb = Mc.draw_lane(Mc.interp(pb)) > 0
    union = (ma | mb).sum()
    return float((ma & mb).sum()) / union if union > 0 else 0.0


def _t(x):
    return torch.tensor(np.asarray(x)[None])


def test_self_iou_is_one():
    xs, valid = real_lanes(2)[0]
    iou = LI.lane_iou_value(_t(xs), _t(xs), _t(valid), angle_aware=True)
    assert abs(iou.item() - 1.0) < 1e-4, iou.item()


def test_pairwise_shape_and_diagonal():
    lanes = real_lanes(4)
    P = min(4, len(lanes))
    xs = torch.tensor(np.stack([l[0] for l in lanes[:P]]))
    valid = torch.tensor(np.stack([l[1] for l in lanes[:P]]))
    m = LI.lane_iou_pairwise(xs, xs, valid)
    assert m.shape == (P, P)
    # la diagonal (carril consigo mismo) debe ser la mayor de su fila
    for i in range(P):
        assert m[i, i] >= m[i].max() - 1e-5


def test_angle_widens_on_tilt():
    """En un tramo inclinado, la anchura angular > anchura constante."""
    # carril inclinado sintético: x crece linealmente con la fila
    xs = torch.linspace(0.3, 0.7, R)[None]
    w = LI._angle_halfwidth(xs, LI.LANE_WIDTH, LI.IMG_W, LI.IMG_H)
    assert (w > LI.LANE_WIDTH).float().mean() > 0.9  # casi todas las filas se ensanchan


def test_gradient_flows():
    # predicción DESPLAZADA respecto al GT (en el óptimo pred==GT el gradiente es ~0)
    xs, valid = real_lanes(2)[0]
    pred = (_t(xs) + 0.02).clone().requires_grad_(True)
    loss = LI.lane_iou_loss(pred, _t(xs), _t(valid))
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_laneiou_closer_to_metric_than_lineiou():
    """CLAVE (tesis): LaneIoU aproxima la IoU de máscara mejor que LineIoU."""
    lanes = real_lanes(12)
    err_lane, err_line = [], []
    for xs, valid in lanes:
        for delta in np.linspace(0.005, 0.05, 8):
            pred = (xs + delta).astype(np.float32)
            m = mask_iou(pred, xs, valid)
            li = LI.lane_iou_value(_t(pred), _t(xs), _t(valid), angle_aware=True).item()
            ln = LI.lane_iou_value(_t(pred), _t(xs), _t(valid), angle_aware=False).item()
            err_lane.append(abs(li - m))
            err_line.append(abs(ln - m))
    mae_lane, mae_line = float(np.mean(err_lane)), float(np.mean(err_line))
    print(f"    MAE(LaneIoU vs métrica)={mae_lane:.4f}  MAE(LineIoU vs métrica)={mae_line:.4f}")
    assert mae_lane < mae_line, f"LaneIoU ({mae_lane:.4f}) no mejora a LineIoU ({mae_line:.4f})"


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
