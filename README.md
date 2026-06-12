# LaneTR — Detector de carriles sin NMS basado en Transformer

Detector de carriles para **CULane** escrito **desde cero en PyTorch** (TFM / tesis). Usa un
**Transformer con atención deformable** sobre un **FPN** y **matching húngaro** (1-a-1, **sin NMS**),
en lugar de las 192 anclas + NMS de CLRNet/CLRerNet. Salida acotada a **≤ 4 carriles** (lo máximo
que anota CULane).

## Novedad de la tesis

> Combinar la **LaneIoU** sensible al ángulo (CLRerNet) como **coste del matching húngaro** *y*
> como **función de pérdida** dentro de un decoder tipo DETR. CLRerNet usa LaneIoU con asignación
> SimOTA (uno-a-muchos, **con** NMS); los detectores transformer (LSTR, Laneformer, O2SFormer)
> usan matching húngaro pero con métricas de distancia simples.
> **Nadie había combinado LaneIoU + matching húngaro.**

---

## Resultados (CULane test)

F1 oficial por categoría (en **%**), estilo CLRNet. `Cross↓` = nº de **falsos positivos**
(Crossroad no tiene carriles válidos → menor es mejor). **Dobla** = curvatura real de la predicción
en píxeles, medida con `tools/diag_curves.py` (el GT real dobla ~26 px). Todas las cifras salen de
los experimentos reales en `work_dirs/`.

| Modelo (config) | F1@50 | Normal | Crowd | Dazzle | Shadow | Noline | Arrow | Curve | Cross↓ | Night | Dobla |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LaneTR · LaneIoU** (baseline) | 75.18 | 90.66 | 73.17 | 65.33 | 73.29 | 50.85 | 85.79 | 63.83 | 1234 | 68.86 | 1.4 px |
| LaneTR · LineIoU *(ablation)* | 73.04 | 89.42 | 71.25 | 67.81 | 71.97 | 48.20 | 83.93 | 60.90 | 1840 | 66.39 | — |
| LaneTR · distancia L1 *(ablation)* | 67.62 | 86.41 | 65.83 | 60.26 | 60.01 | 43.39 | 79.05 | 56.75 | 2222 | 59.74 | — |
| + refinamiento (`refine_main`) | 75.73 | 90.94 | 74.37 | 67.34 | 73.11 | 49.66 | 85.91 | 63.57 | 1235 | 69.26 | — |
| **+ refinamiento (`refine_3pt`)** | **76.06** | 91.22 | 75.13 | 67.93 | 74.30 | 50.16 | 86.14 | 63.62 | 1314 | 69.27 | 1.3 px |
| + refinamiento (`refine_xs`) | 74.78 | 91.01 | 73.65 | 66.34 | 71.16 | 50.40 | 85.92 | 63.65 | 1779 | 67.91 | 0.7 px |
| + curvas (`curve_all`) | 73.88 | 89.58 | 72.09 | 63.88 | 74.91 | 48.53 | 83.91 | 67.32 | 1437 | 66.66 | 5.0 px |
| **+ curvas (`curve_clip`)** | 72.87 | 89.89 | 70.97 | 63.99 | 71.38 | 48.70 | 83.94 | **69.92** | 1972 | 65.26 | 5.4 px |

**Cómo leer la tabla:**
- **Hipótesis de la tesis confirmada** (3 primeras filas): `LaneIoU (75.18) > LineIoU (73.04) > distancia (67.62)`, tanto en global como en `Curve`.
- **Refinamiento** (arquitectura): `refine_3pt` da el **mejor F1 global (76.06)** y sube Normal/Crowd, **pero `Curve` no se mueve** y la predicción sigue recta (1.3 px) → la arquitectura no es el cuello de botella.
- **Curvas** (pérdida + datos): el problema es de **frecuencia** (las curvas son ~1–3 % de CULane y su gradiente se promedia con muchos rectos). Con un **peso por curvatura** + **sobre-muestreo** + aflojar el `grad_clip`, **la predicción por fin dobla** (1.3 → 5.4 px) y **`Curve` sube a 69.92**. *Trade-off*: enfatizar el ~1–3 % curvo cuesta algo de F1 global.

---

## Arquitectura

```
imagen 1640×590 → recorte (y ≥ 270) + resize 800×320
  → DLA-34                         → C3, C4, C5
  → FPN                            → P3, P4, P5 (256 canales)
  → decoder transformer (6 capas)  → 12 queries con prior posicional (anclas) +
                                     atención DEFORMABLE (muestrea pocos puntos, no 5250 tokens)
  → cabezas por query              → confianza + geometría (x en 144 filas, start_y, longitud)
Entrenamiento: matching húngaro (coste = focal + LaneIoU + L1) + pérdida (focal + LaneIoU + L1) + capas auxiliares
Inferencia:    umbral de confianza → salen ≤ 4 carriles, SIN NMS.
```

**Claves del diseño:**
- Cada carril se representa como `xs` = la **x en 144 filas fijas**, predicha como `prior + delta`
  (el prior es la línea recta del ancla; la cabeza aprende el `delta`).
- Las **anclas** nacen en abanico de borde a borde y dan a cada query una predicción distinta desde
  el primer paso → estabilizan el matching húngaro dinámico (inestabilidad clásica de DETR).
- La **atención deformable** muestrea unos pocos puntos alrededor de unos **puntos de referencia**
  (≈96 muestras/query) en vez de los 5250 tokens del FPN.
- En curvas, los puntos de referencia se reparten **a lo largo del carril** y se **refinan capa a
  capa** para seguir el doblez (estilo DAB-DETR).

**Eficiencia** (320×800): **23.7 M params · 42.35 GFLOPs · 30.2 FPS @ RTX 4060** (baseline; con
refinamiento ≈ 24.7 M). La deformable es PyTorch puro (sin kernel CUDA) y es el cuello de botella →
re-medir en A6000 para la cifra final. Reproducir: `pip install thop && python tools/profile_model.py`.

Detalle módulo a módulo, con formas de tensor y matemática, en [`ARQUITECTURA.md`](ARQUITECTURA.md).

---

## Cómo funciona el repo

- **Todo se controla por config** (`configs/lanetr_culane.yaml`) + overrides `--set clave=valor`.
  Las **ablations NO son ramas**: son la misma config con un override (p.ej. `--set loss.geo_metric=lineiou`).
- **Convención de calidad** — cada componente entrega dos cosas:
  - `tools/verify_<x>.py` → inspección **visual** (figuras en `outputs/verify/`).
  - `tests/test_<x>.py` → comprobaciones **automáticas** (`python tests/test_<x>.py`, sin pytest).
- Cada entrenamiento crea **`work_dirs/<run>_<ts>/`** con: `config.yaml`, `train.log` (loss + lr +
  ETA), `eval.log` (F1/categorías), `gpu.log`, `checkpoints/` (`last.pth`, `best.pth`),
  `viz/epoch_XXX/` (GT vs pred, atención, anclas, matcher) y `results.json` (F1 final + categorías).

### Estructura

```
lanetr/                 paquete principal
├── paths.py            resuelve rutas del dataset desde .env
├── config.py           carga de YAML + defaults + overrides
├── data/               filtro de coche parado, parser .lines.txt, dataset, codificación a filas-ancla
├── metrics/            métrica F1 (Python, = C++) + wrapper del evaluador oficial + eval por época
├── models/             backbone (DLA-34), FPN, decoder, deformable, anclas, cabezas, LaneTR
├── losses/             LaneIoU, matcher húngaro, criterion (pérdida total)
└── training/           FrozenBN, EMA, optim (lr diferenciado + scheduler), work_dir, visualizaciones
tools/                  scripts ejecutables: train, test, run_ablations, collect_results,
                        compute_curvature, diag_curves, profile_model, verify_*
tests/                  tests automáticos (asserts)
configs/                config YAML de entrenamiento
evaluation/             evaluador oficial C++ (vendorizado) + Dockerfile
```

---

## Instalación

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11. Desarrollo/pruebas en Windows; **entrenamiento** en Ubuntu 22.04 + NVIDIA RTX A6000 (48 GB).

## Dataset (CULane)

Configura la ruta en un `.env` (no se commitea):

```
DATASET_DIR=D:\CULane
```

Las rutas se resuelven siempre vía `lanetr/paths.py`. CULane: imágenes **1640×590**, anotaciones
`*.lines.txt`, máscaras `laneseg_label_w16/`, y `list/` (train/val/test + `train_gt_new.txt`, la
lista de train ya filtrada de escenas de coche parado).

## Uso

```bash
# Entrenar (baseline)
python tools/train.py --config configs/lanetr_culane.yaml

# Evaluar un checkpoint (F1 global + por categoría + calibrar umbral)
python tools/test.py --checkpoint work_dirs/<run>/checkpoints/best.pth --categories --calibrate

# Ablations de la tesis (2 GPUs) + tabla de resultados
python tools/run_ablations.py --gpus 0,1
python tools/collect_results.py            # -> work_dirs/ablations_table.md
```

### Curvas

```bash
# refinamiento iterativo de referencias (arquitectura)
python tools/run_ablations.py --gpus 0,1 --only refine_main refine_3pt refine_xs

# énfasis en curvas (pérdida + datos): primero precalcula la curvatura por frame
python tools/compute_curvature.py                       # -> list/train_curvature.npz
python tools/run_ablations.py --gpus 0,1 --only curve_w curve_os curve_all

# medir cuánto dobla la predicción (px) frente al GT
python tools/diag_curves.py work_dirs/<run>/checkpoints/best.pth
```

Flags útiles (vía `--set`): `model.ref_refine`, `model.n_ref_points`, `loss.curve_gamma`,
`data.curve_oversample`, `train.grad_clip`.

### Métrica oficial (C++ vía Docker)

```bash
docker build -t culane-eval:official -f evaluation/Dockerfile evaluation
```

La métrica en Python (`lanetr/metrics/culane.py`) está validada como **idéntica** al evaluador
oficial en C++ (TP/FP/FN/F1), así que los números son comparables con los papers.
