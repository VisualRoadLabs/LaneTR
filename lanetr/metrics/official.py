"""Wrapper del evaluador OFICIAL de CULane en C++ (vía Docker).

Compila/usa la imagen `culane-eval:official` (ver `evaluation/Dockerfile`). Sirve como
patrón de oro para validar la métrica Python (`lanetr.metrics.culane`).

Estructura de trabajo esperada en `work_dir`:
    work_dir/anno/<rel>.lines.txt   anotaciones GT
    work_dir/pred/<rel>.lines.txt   predicciones
    work_dir/list.txt               rutas de imagen (un `.jpg` por línea, como en *_gt.txt)
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

DEFAULT_IMAGE = "culane-eval:official"


def docker_available() -> bool:
    try:
        r = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                           capture_output=True, text=True, timeout=20)
        return r.returncode == 0 and r.stdout.strip() != ""
    except Exception:
        return False


def image_exists(image: str = DEFAULT_IMAGE) -> bool:
    try:
        r = subprocess.run(["docker", "images", "-q", image], capture_output=True, text=True, timeout=20)
        return r.returncode == 0 and r.stdout.strip() != ""
    except Exception:
        return False


def parse_output(text: str) -> dict:
    out = {"TP": None, "FP": None, "FN": None, "Precision": None, "Recall": None, "F1": None}
    m = re.search(r"tp:\s*(\d+)\s+fp:\s*(\d+)\s+fn:\s*(\d+)", text)
    if m:
        out["TP"], out["FP"], out["FN"] = (int(g) for g in m.groups())
    for key, dst in (("precision", "Precision"), ("recall", "Recall"), ("Fmeasure", "F1")):
        mm = re.search(rf"{key}:\s*([0-9.eE+-]+)", text)
        if mm:
            out[dst] = float(mm.group(1))
    return out


def run_official(work_dir: str | Path, width: int = 30, iou: float = 0.5,
                 im_w: int = 1640, im_h: int = 590, image: str = DEFAULT_IMAGE) -> dict:
    """Ejecuta el evaluador oficial sobre `work_dir` y devuelve {TP,FP,FN,Precision,Recall,F1}."""
    work_dir = Path(work_dir).resolve()
    out_file = work_dir / "out.txt"
    if out_file.exists():
        out_file.unlink()
    cmd = [
        "docker", "run", "--rm", "-v", f"{work_dir}:/work", image,
        "-a", "/work/anno", "-d", "/work/pred", "-i", "/work/anno",
        "-l", "/work/list.txt", "-w", str(width), "-t", str(iou),
        "-c", str(im_w), "-r", str(im_h), "-f", "1", "-o", "/work/out.txt",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"docker run falló (rc={proc.returncode}):\n{proc.stderr}\n{proc.stdout}")
    text = out_file.read_text(encoding="utf-8") if out_file.exists() else proc.stdout
    res = parse_output(text)
    res["_raw"] = text
    return res
