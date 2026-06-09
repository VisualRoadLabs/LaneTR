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
    """Ancla = carril-prior COMPLETO por query: (start_x, start_y, slope, length).

    - start_x, start_y : punto de inicio (extremo cercano, abajo).
    - slope            : pendiente dx/dy de la línea-prior (dirección).
    - length           : extensión vertical del carril (fracción de imagen).

    Init en abanico: start_x espaciado por abajo de borde a borde [0, 1] (las anclas de los
    extremos llegan a los laterales para cubrir los carriles lejanos, pero SIN salirse de la
    imagen), start_y abajo, pendiente hacia el centro-arriba, y length largo (≈0.9) → el
    abanico nace ancho y cubriendo toda la imagen. `x_margin` permite estirar más allá de los
    bordes si se quiere (por defecto 0 = justo [0, 1], sin sobresalir).
    """

    def __init__(self, num_queries: int = 12, d_model: int = 256, num_freq: int = 8,
                 x_margin: float = 0.0):
        super().__init__()
        self.num_queries = num_queries
        sx = torch.linspace(-x_margin, 1.0 + x_margin, num_queries)
        sy = torch.full((num_queries,), 0.98)
        slope = sx - 0.5
        length = torch.full((num_queries,), 0.9)
        self.anchors = nn.Parameter(torch.stack([sx, sy, slope, length], dim=1))  # (NQ, 4)

        self.register_buffer("freqs", (2.0 ** torch.arange(num_freq)) * math.pi)
        self.mlp = MLP(self.anchors.shape[1] * 2 * num_freq, d_model, d_model)

    def prior_xs(self, row_ys: torch.Tensor, img_h: int) -> torch.Tensor:
        """Línea-prior recta: x normalizada en cada fila-ancla. -> (NQ, R)."""
        y = (row_ys.to(self.anchors.device) / (img_h - 1)).float()          # (R,) en [0,1]
        sx = self.anchors[:, 0:1]
        sy = self.anchors[:, 1:2]
        k = self.anchors[:, 2:3]
        return sx + (y[None, :] - sy) * k                                   # (NQ, R)

    def ext_prior(self) -> torch.Tensor:
        """Prior de extensión (start_y, length) por query. -> (NQ, 2)."""
        return self.anchors[:, [1, 3]].clamp(1e-4, 1 - 1e-4)

    def reference_points(self, y_ref: float = 0.5) -> torch.Tensor:
        """Punto de referencia (x, y) por query para la atención deformable: la línea-prior
        evaluada a media altura (centro del carril). -> (NQ, 2) en [0,1]."""
        sx, sy, k = self.anchors[:, 0], self.anchors[:, 1], self.anchors[:, 2]
        rx = sx + (y_ref - sy) * k
        ry = torch.full_like(rx, y_ref)
        return torch.stack([rx, ry], dim=-1).clamp(0.0, 1.0)

    def pos_embed(self) -> torch.Tensor:
        """Codifica las anclas (sinusoidal + MLP) en embeddings posicionales. -> (NQ, d_model)."""
        ang = self.anchors[..., None] * self.freqs                          # (NQ, 4, F)
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(1)          # (NQ, 4*2F)
        return self.mlp(emb)
