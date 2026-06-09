"""Validación de la métrica Python contra el evaluador OFICIAL en C++ (Paso 2B).

Se salta automáticamente si Docker o la imagen `culane-eval:official` no están disponibles
(para que la suite no falle en máquinas sin Docker).

    .\.venv\Scripts\python.exe tests\test_metric_official.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lanetr import paths
from lanetr.data import culane_annotation as ann
from lanetr.metrics import culane as M
from lanetr.metrics import format as F
from lanetr.metrics import official as O

N = 12


def _images():
    lines = [l for l in (paths.list_dir() / "val_gt.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [ann.parse_gt_line(l)[0] for l in lines[:N]]


def _build(images, make_pred):
    work = paths.outputs_dir() / "official_xval_test"
    if work.exists():
        shutil.rmtree(work)
    (work / "anno").mkdir(parents=True)
    (work / "pred").mkdir(parents=True)
    list_lines, py_preds, py_annos = [], [], []
    for rel in images:
        gt = M.load_culane_img_data(str(ann.lines_path_for_image(rel)))
        rel_txt = rel.lstrip("/\\").replace(".jpg", ".lines.txt")
        dst = work / "anno" / rel_txt
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ann.lines_path_for_image(rel), dst)
        pred = make_pred(gt)
        F.write_lines_file(pred, work / "pred" / rel_txt)
        list_lines.append(rel if rel.startswith("/") else "/" + rel)
        py_preds.append([np.asarray(p) for p in pred])
        py_annos.append([np.asarray(g) for g in gt])
    (work / "list.txt").write_text("\n".join(list_lines) + "\n", encoding="utf-8")
    return work, py_preds, py_annos


def test_python_matches_official_cpp():
    if not (O.docker_available() and O.image_exists()):
        print("  SKIP  (Docker o imagen 'culane-eval:official' no disponibles)")
        return

    # Escenario A: predicción = GT exacta -> F1 = 1.0
    work, py_preds, py_annos = _build(_images(), lambda g: [np.asarray(x) for x in g])
    cpp = O.run_official(work)
    py = M.evaluate(py_preds, py_annos)[0.5]
    assert cpp["FP"] == 0 and cpp["FN"] == 0, cpp
    assert abs(cpp["F1"] - 1.0) < 1e-6, cpp
    assert (py["TP"], py["FP"], py["FN"]) == (cpp["TP"], cpp["FP"], cpp["FN"]), (py, cpp)

    # Escenario B: predicción perturbada -> ambos deben coincidir en conteos
    def perturb(lanes):
        out = []
        for k, lane in enumerate(lanes):
            if k == 0:
                out.append(lane + np.array([60.0, 0.0]))
            elif k == len(lanes) - 1 and len(lanes) >= 3:
                continue
            else:
                out.append(lane + np.array([6.0, 0.0]))
        return out

    work, py_preds, py_annos = _build(_images(), perturb)
    cpp = O.run_official(work)
    py = M.evaluate(py_preds, py_annos)[0.5]
    assert (py["TP"], py["FP"], py["FN"]) == (cpp["TP"], cpp["FP"], cpp["FN"]), (py, cpp)


def _run_all() -> int:
    failed = 0
    for name, t in sorted(globals().items()):
        if name.startswith("test_") and callable(t):
            try:
                t()
                print(f"  PASS  {name}")
            except AssertionError as e:
                failed += 1
                print(f"  FAIL  {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'OK' if not failed else 'FALLOS: ' + str(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
