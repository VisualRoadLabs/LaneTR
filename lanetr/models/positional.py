"""Codificación posicional 2D sinusoidal (estilo DETR).

La atención no sabe "dónde" está cada token de la imagen; esta codificación inyecta la
posición (fila, columna) de cada celda del mapa de features como un vector sinusoidal, que
se suma a la memoria antes de la cross-attention.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    """Genera un tensor (B, 2*num_pos_feats, H, W) con la posición 2D codificada.

    Con `num_pos_feats = d_model // 2`, la salida tiene `d_model` canales.
    """

    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000,
                 normalize: bool = True, scale: float | None = None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, b: int, h: int, w: int, device, dtype=torch.float32) -> torch.Tensor:
        y_embed = torch.arange(1, h + 1, device=device, dtype=dtype).view(h, 1).repeat(1, w)
        x_embed = torch.arange(1, w + 1, device=device, dtype=dtype).view(1, w).repeat(h, 1)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (h + eps) * self.scale
            x_embed = x_embed / (w + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t  # (h, w, num_pos_feats)
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=3).flatten(2)
        pos_y = torch.stack([pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=3).flatten(2)
        pos = torch.cat([pos_y, pos_x], dim=2)             # (h, w, 2*num_pos_feats)
        pos = pos.permute(2, 0, 1).unsqueeze(0)            # (1, 2*npf, h, w)
        return pos.repeat(b, 1, 1, 1)
