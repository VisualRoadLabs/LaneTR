"""Prior posicional para las queries (Paso 5.1).

Cada query nace con un **ancla** = (start_x, start_y, slope) que define una línea-prior recta
en la imagen. Las anclas se inicializan repartidas en abanico (puntos de inicio espaciados por
abajo, convergiendo hacia un punto de fuga arriba-centro), como los carriles reales.

Con esto:
  - cada query produce de salida una predicción DISTINTA desde el primer paso (no todas
    centradas) → el matcher deja de dudar → el matching dinámico se estabiliza;
  - el ancla se codifica en el *embedding* posicional de la query (guía la atención);
  - la cabeza predice las `xs` como **prior + delta** (refina sobre la línea-prior).

Las anclas son `nn.Parameter` (aprendibles): el modelo las ajusta durante el entrenamiento.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .head import MLP


class LaneAnchors(nn.Module):
    def __init__(self, num_queries: int = 12, d_model: int = 256, num_freq: int = 8):
        super().__init__()
        self.num_queries = num_queries
        # init en abanico: start_x espaciado, start_y abajo, pendiente hacia el centro-arriba
        sx = torch.linspace(0.1, 0.9, num_queries)
        sy = torch.full((num_queries,), 0.98)
        slope = sx - 0.5
        self.anchors = nn.Parameter(torch.stack([sx, sy, slope], dim=1))  # (NQ, 3)

        self.register_buffer("freqs", (2.0 ** torch.arange(num_freq)) * math.pi)
        self.mlp = MLP(3 * 2 * num_freq, d_model, d_model)

    def prior_xs(self, row_ys: torch.Tensor, img_h: int) -> torch.Tensor:
        """Línea-prior recta: x normalizada en cada fila-ancla. -> (NQ, R)."""
        y = (row_ys.to(self.anchors.device) / (img_h - 1)).float()          # (R,) en [0,1]
        sx = self.anchors[:, 0:1]
        sy = self.anchors[:, 1:2]
        k = self.anchors[:, 2:3]
        return sx + (y[None, :] - sy) * k                                   # (NQ, R)

    def pos_embed(self) -> torch.Tensor:
        """Codifica las anclas (sinusoidal + MLP) en embeddings posicionales. -> (NQ, d_model)."""
        ang = self.anchors[..., None] * self.freqs                          # (NQ, 3, F)
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(1)          # (NQ, 3*2F)
        return self.mlp(emb)
