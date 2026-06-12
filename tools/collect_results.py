"""Junta los resultados de las ablations en una tabla comparativa (Paso 6.4).

Lee `results.json` de cada `work_dirs/abl_*/` y produce:
    work_dirs/ablations_table.csv   (para Excel/análisis)
    work_dirs/ablations_table.md    (tabla Markdown para la memoria)
y la imprime por consola.

Uso:
    python tools/collect_results.py
    python tools/collect_results.py --dir work_dirs
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATS = ["normal", "crowd", "dazzle", "shadow", "noline", "arrow", "curve", "cross", "night"]


def collect(work_dirs: Path) -> list[dict]:
    rows = []
    for rj in sorted(work_dirs.glob("*/results.json")):
        try:
            rows.append(json.loads(rj.read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001
            print(f"[aviso] no se pudo leer {rj}: {e}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "work_dirs"))
    args = ap.parse_args()
    work_dirs = Path(args.dir)
    rows = collect(work_dirs)
    if not rows:
        print(f"No hay results.json en {work_dirs}. Entrena con tools/run_ablations.py primero.")
        return

    # ordena: 'main' primero, luego por F1 de test descendente
    rows.sort(key=lambda r: (r.get("name") != "abl_main", -r.get("test_F1", 0.0)))

    headers = ["modelo", "F1_test", "umbral", "F1_val"] + CATS
    csv_path = work_dirs / "ablations_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            cats = r.get("categories", {})
            w.writerow([r.get("name"), f"{r.get('test_F1', 0):.4f}", f"{r.get('conf_thresh', 0):.2f}",
                        f"{r.get('best_f1_val', 0):.4f}"]
                       + [f"{cats.get(c, 0):.4f}" if c != "cross" else str(cats.get(c, "")) for c in CATS])

    # tabla Markdown
    md = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        cats = r.get("categories", {})
        cells = [r.get("name", "?"), f"{r.get('test_F1', 0):.4f}", f"{r.get('conf_thresh', 0):.2f}",
                 f"{r.get('best_f1_val', 0):.4f}"]
        cells += [f"{cats.get(c, 0):.4f}" if c != "cross" else str(cats.get(c, "")) for c in CATS]
        md.append("| " + " | ".join(str(c) for c in cells) + " |")
    md_text = "\n".join(md)
    (work_dirs / "ablations_table.md").write_text(md_text + "\n", encoding="utf-8")

    print(f"\nTabla de ablations ({len(rows)} modelos):  (cross = FP, menor es mejor)\n")
    print(md_text)
    print(f"\nGuardado: {csv_path}  y  ablations_table.md")


if __name__ == "__main__":
    main()
