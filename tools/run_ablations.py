"""Runner de las ablations de la tesis (Paso 6.4), repartidas entre varias GPUs.

Lanza el MODELO PRINCIPAL primero y luego las variantes (una por ablation), distribuyéndolas
entre las GPUs disponibles (p.ej. las 2 RTX A6000: GPU 0 y GPU 1) mediante CUDA_VISIBLE_DEVICES.
Cada entrenamiento crea su `work_dirs/<name>_<ts>/` con su `results.json`. Luego usa
`tools/collect_results.py` para juntar todo en una tabla.

Cada ablation = la config base (`configs/lanetr_culane.yaml`) con un override puntual.

Uso:
    python tools/run_ablations.py --config configs/lanetr_culane.yaml --gpus 0,1
    python tools/run_ablations.py --gpus 0,1 --dry-run          # ver los comandos sin lanzar
    python tools/run_ablations.py --gpus 0,1 --only main geo_distance
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (nombre, overrides sobre la config base). 'main' primero = tu modelo principal.
ABLATIONS = [
    ("main", {}),                                              # LaneIoU + 12q + anclas + deformable + filtrado
    ("geo_distance", {"loss.geo_metric": "distance"}),         # *** ablation clave: distancia simple vs LaneIoU
    ("geo_lineiou", {"loss.geo_metric": "lineiou"}),           # LineIoU (anchura constante, CLRNet)
    ("q4", {"model.num_queries": 4}),                          # 4 queries
    ("q20", {"model.num_queries": 20}),                        # 20 queries
    ("with_o2m", {"loss.aux_one_to_many": "true"}),            # + asignación uno-a-muchos
    ("no_deformable", {"model.deformable": "false"}),          # atención densa
    ("no_filter", {"data.train_split": "train_full"}),         # sin filtro de coche parado (88.880)
]


def build_cmd(config, name, overrides):
    sets = [f"name=abl_{name}"] + [f"{k}={v}" for k, v in overrides.items()]
    return [sys.executable, str(ROOT / "tools" / "train.py"), "--config", config, "--set", *sets]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "lanetr_culane.yaml"))
    ap.add_argument("--gpus", default="0,1", help="GPUs a usar, p.ej. 0,1")
    ap.add_argument("--only", nargs="*", default=None, help="subconjunto de ablations por nombre")
    ap.add_argument("--epochs", type=int, default=None, help="train.epochs para TODAS las ablations")
    ap.add_argument("--set", nargs="*", default=None,
                    help="overrides globales (clave.anidada=valor) aplicados a todas las ablations")
    ap.add_argument("--dry-run", action="store_true", help="solo imprime los comandos")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    ablations = [(n, o) for n, o in ABLATIONS if (args.only is None or n in args.only)]

    # overrides globales (p.ej. --epochs 15) aplicados a TODAS las ablations
    global_ov: dict = {}
    if args.epochs is not None:
        global_ov["train.epochs"] = args.epochs
    for kv in (args.set or []):
        k, v = kv.split("=", 1)
        global_ov[k] = v
    (ROOT / "work_dirs").mkdir(parents=True, exist_ok=True)   # crear ANTES de abrir logs

    print(f"Ablations ({len(ablations)}): {[n for n, _ in ablations]}")
    print(f"GPUs: {gpus}  (modelo principal 'main' primero)  overrides globales: {global_ov or '—'}\n")

    if args.dry_run:
        for i, (name, ov) in enumerate(ablations):
            gpu = gpus[i % len(gpus)]
            cmd = build_cmd(args.config, name, {**global_ov, **ov})
            print(f"[GPU {gpu}] {' '.join(cmd)}")
        print("\n(dry-run: nada lanzado. Quita --dry-run para entrenar.)")
        return

    queue = deque(ablations)
    running: dict[int, tuple] = {}
    while queue or running:
        for gpu in gpus:
            if gpu not in running and queue:
                name, ov = queue.popleft()
                cmd = build_cmd(args.config, name, {**global_ov, **ov})
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                logf = open(ROOT / "work_dirs" / f"_runner_{name}.out", "w", encoding="utf-8")
                p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
                running[gpu] = (name, p, logf)
                print(f"[GPU {gpu}] lanzado '{name}'  -> work_dirs/_runner_{name}.out")
        time.sleep(3)   # poll frecuente -> la GPU coge el siguiente en cuanto se libera
        for gpu, (name, p, logf) in list(running.items()):
            if p.poll() is not None:
                logf.close()
                print(f"[GPU {gpu}] terminado '{name}' (rc={p.returncode})")
                del running[gpu]

    print("\nTodas las ablations terminadas. Junta la tabla con:")
    print("    python tools/collect_results.py")


if __name__ == "__main__":
    main()
