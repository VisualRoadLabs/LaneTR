# Changelog

Todas las versiones relevantes de **LaneTR**. Formato basado en
[Keep a Changelog](https://keepachangelog.com/); versionado [SemVer](https://semver.org/).
Las cifras son F1 oficial de CULane (test) medidas en `work_dirs/`.

## [0.3.0] — Curve emphasis (frequency) — `v0.3.0-curves`

Las curvas por fin doblan. La predicción pasa de ~1.3 px a ~5 px de curvatura y `Curve` sube
**63.83 → 69.92**, con un *trade-off* en el F1 global.

### Added
- Pérdida con **peso por curvatura por carril** (`loss.curve_gamma`): enfatiza los carriles curvos
  (raros) sin tocar los rectos.
- **Sobre-muestreo de curvas** (`data.curve_oversample`, `WeightedRandomSampler`) +
  `tools/compute_curvature.py` → `list/train_curvature.npz`.
- `train.grad_clip` configurable (0.1 estrangulaba el gradiente de doblar; 0.5 fue lo mejor).
- Estudios `curve_w`, `curve_os`, `curve_all` en el runner.

### Findings
- El cuello de botella era **frecuencia** (curvas ~1–3 % de CULane → "colapso al carril medio"),
  no la fórmula de la pérdida ni la arquitectura.
- Descartados tras verificación numérica: subir `w_xy`, `w_smooth` (empeora), `w_theta` (muerto),
  `w_curv` (2ª dif. ~1e-5, inerte) y estrechar LaneIoU (gradiente ya grande, riesgo alto).

### Results
- `curve_clip` (γ + oversampling + grad_clip 0.5): F1 **72.87** / Curve **69.92** / dobla 5.4 px.
- `curve_all` (γ + oversampling): F1 73.88 / Curve 67.32 / dobla 5.0 px.

## [0.2.0] — Iterative reference refinement — `v0.2.0-refinement`

La atención deformable aprende a **seguir** la curva. Mejor F1 global del proyecto, pero las curvas
siguen rectas → demuestra que el cuello de botella no es la arquitectura.

### Added
- **Múltiples puntos de referencia a lo largo del carril** (`model.n_ref_points`).
- **Refinamiento iterativo de referencias** (`model.ref_refine`, modos `mlp` (DAB-DETR) y `xs`):
  tras cada capa, las referencias mueven su x hacia el carril predicho.
- `ref_refine_mlp` en el grupo de lr lento (0.1×); alturas de referencia configurables.
- Estudios `refine_main`, `refine_3pt`, `refine_xs`; `tools/diag_curves.py` (curvatura predicha en px).

### Findings
- El refinamiento sube el F1 global y Normal/Crowd, **pero `Curve` no se mueve** (la predicción
  sigue recta, ~1.3 px) → la arquitectura ya deja ver la curva; el problema está en la pérdida/datos.

### Results
- `refine_3pt`: **F1 76.06** (mejor global) / Curve 63.62.
- `refine_xs`: F1 74.78 / Curve 63.65 · `refine_main`: F1 75.73 / Curve 63.57.

## [0.1.0] — Baseline LaneTR — `v0.1.0-baseline`

Baseline completa: detector NMS-free con LaneIoU + matching húngaro.

### Added
- **Datos**: filtro de coche parado, parser `.lines.txt`, dataset/dataloader, codificación a 144 filas-ancla.
- **Métrica**: F1 en Python, validada **idéntica** al evaluador oficial en C++ (Docker).
- **Modelo**: DLA-34 + FPN + decoder transformer (deformable + anclas) + cabezas por query → ≤ 4 carriles, sin NMS.
- **Pérdida**: LaneIoU diferenciable (sensible al ángulo), matcher húngaro (coste LaneIoU), criterion completo.
- **Entrenamiento**: bf16, grad-clip, FrozenBatchNorm, EMA, warmup+cosine; eval y visualizaciones por época.
- **Ablations**: runner 2-GPU + colector de resultados.

### Findings
- **Hipótesis de la tesis confirmada**: `LaneIoU (75.18) > LineIoU (73.04) > distancia (67.62)`.
- `Curve` es la categoría más floja en las tres métricas → problema estructural (motiva v0.2.0 / v0.3.0).

### Results
- LaneTR (LaneIoU): **F1 75.18** / Curve 63.83.
