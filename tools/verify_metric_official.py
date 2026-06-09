"""Validación cruzada: métrica Python vs evaluador OFICIAL en C++ (Paso 2B).

Prepara un directorio de trabajo con anotaciones GT reales y predicciones sintéticas,
y compara los TP/FP/FN/F1 que dan:
  - `lanetr.metrics.culane` (Python), y
  - el evaluador oficial en C++ (Docker, imagen `culane-eval:official`).
Dos escenarios:
  A) predicción = GT exacta            -> F1 debe ser 1.0 en ambos.
  B) predicción = GT perturbada        -> ambos deben coincidir en TP/FP/FN.

Si Docker o la imagen no están disponibles, lo indica y solo corre la parte Python.

Uso:
    .\.venv\Scripts\python.exe tools\verify_metric_official.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from lanetr import paths
from lanetr.data import culane_annotation as ann
from lanetr.metrics import culane as M
from lanetr.metrics import format as F
from lanetr.metrics import official as O

N_IMAGES = 40


def pick_images() -> list[str]:
    lines = [l for l in (paths.list_dir() / "val_gt.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [ann.parse_gt_line(l)[0] for l in lines[:N_IMAGES]]


def perturb(lanes: list[np.ndarray]) -> list[np.ndarray]:
    """Predicción sintética: desplaza ligeramente unos carriles y mucho otro."""
    out = []
    for k, lane in enumerate(lanes):
        if k == 0 and len(lanes) >= 1:
            out.append(lane + np.array([60.0, 0.0]))  # muy desplazado -> rompe match
        elif k == len(lanes) - 1 and len(lanes) >= 3:
            pass  # se omite -> FN
        else:
            out.append(lane + np.array([6.0, 0.0]))  # leve -> sigue TP
    return out


def build_workdir(images, make_pred):
    work = paths.outputs_dir() / "official_xval"
    if work.exists():
        shutil.rmtree(work)
    (work / "anno").mkdir(parents=True)
    (work / "pred").mkdir(parents=True)
    list_lines = []
    py_preds, py_annos = [], []
    for rel in images:
        gt = M.load_culane_img_data(str(ann.lines_path_for_image(rel)))
        rel_txt = rel.lstrip("/\\").replace(".jpg", ".lines.txt")
        # anno = GT real (copiado para que C++ y Python lean exactamente lo mismo)
        dst_anno = work / "anno" / rel_txt
        dst_anno.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ann.lines_path_for_image(rel), dst_anno)
        # pred
        pred = make_pred(gt)
        F.write_lines_file(pred, work / "pred" / rel_txt)
        list_lines.append(rel if rel.startswith("/") else "/" + rel)
        py_preds.append([np.asarray(p) for p in pred])
        py_annos.append([np.asarray(g) for g in gt])
    (work / "list.txt").write_text("\n".join(list_lines) + "\n", encoding="utf-8")
    return work, py_preds, py_annos


def run_scenario(name, images, make_pred, run_cpp):
    work, py_preds, py_annos = build_workdir(images, make_pred)
    py = M.evaluate(py_preds, py_annos)[0.5]
    print(f"\n[{name}]")
    print(f"  PYTHON : TP={py['TP']} FP={py['FP']} FN={py['FN']} "
          f"P={py['Precision']:.4f} R={py['Recall']:.4f} F1={py['F1']:.4f}")
    if run_cpp:
        cpp = O.run_official(work)
        print(f"  C++    : TP={cpp['TP']} FP={cpp['FP']} FN={cpp['FN']} "
              f"P={cpp['Precision']:.4f} R={cpp['Recall']:.4f} F1={cpp['F1']:.4f}")
        same = (py["TP"], py["FP"], py["FN"]) == (cpp["TP"], cpp["FP"], cpp["FN"])
        f1_close = abs(py["F1"] - (cpp["F1"] or 0)) < 1e-3
        print(f"  -> TP/FP/FN idénticos: {same} | F1 coincide: {f1_close}")
        return same and f1_close
    return None


def main() -> int:
    print("=" * 70)
    print("VALIDACIÓN CRUZADA: métrica Python vs C++ OFICIAL (Paso 2B)")
    print("=" * 70)
    images = pick_images()
    print(f"Imágenes: {len(images)} (de val)")

    run_cpp = O.docker_available() and O.image_exists()
    if not run_cpp:
        if not O.docker_available():
            print("\n[aviso] Docker no disponible -> solo métrica Python.")
        else:
            print("\n[aviso] Falta la imagen 'culane-eval:official'. Constrúyela con:")
            print("        docker build -t culane-eval:official -f evaluation/Dockerfile evaluation")

    ok_a = run_scenario("A: predicción = GT exacta (esperado F1=1.0)", images,
                        lambda g: [np.asarray(x) for x in g], run_cpp)
    ok_b = run_scenario("B: predicción = GT perturbada", images, perturb, run_cpp)

    print("\n" + "=" * 70)
    if run_cpp:
        ok = bool(ok_a) and bool(ok_b)
        print("RESULTADO:", "Python y C++ COINCIDEN [OK]" if ok else "DISCREPANCIA [X]")
        print("=" * 70)
        return 0 if ok else 1
    print("RESULTADO: solo Python (sin C++). Construye la imagen para la validación completa.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
