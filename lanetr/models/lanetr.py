"""Modelo completo LaneTR (Paso 3.4): backbone + FPN + decoder + cabezas.

    imagen (B,3,320,800)
      -> backbone (DLA-34)        -> [C3,C4,C5]
      -> FPN                      -> [P3,P4,P5] (256 ch)
      -> decoder transformer      -> hs (L, B, num_queries, 256)
      -> cabezas                  -> {conf, xs, start_y, length, theta}

`forward` devuelve el dict de predicciones (todas las capas, para pérdidas auxiliares).
`predict` decodifica la última capa en carriles dibujables.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import build_backbone
from .decoder import LaneDecoder
from .fpn import FPN
from .head import LaneHead, decode_lanes


class LaneTR(nn.Module):
    def __init__(self, backbone: str = "dla34", pretrained: bool = True, d_model: int = 256,
                 num_queries: int = 12, num_layers: int = 6, num_rows: int = 144,
                 nhead: int = 8, dim_ff: int = 1024):
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained)
        self.fpn = FPN(self.backbone.out_channels, d_model)
        self.decoder = LaneDecoder(d_model=d_model, nhead=nhead, num_layers=num_layers,
                                   num_queries=num_queries, dim_ff=dim_ff,
                                   num_levels=len(self.backbone.out_channels))
        self.head = LaneHead(d_model, num_rows)
        self.num_rows = num_rows
        self.num_queries = num_queries

    def forward(self, images: torch.Tensor) -> dict:
        feats = self.fpn(self.backbone(images))
        hs = self.decoder(feats)            # (L, B, NQ, D)
        return self.head(hs)

    @torch.no_grad()
    def predict(self, images: torch.Tensor, conf_thresh: float | None = 0.5,
                img_w: int = 800, img_h: int = 320) -> list[list[dict]]:
        self.eval()
        pred = self.forward(images)
        return decode_lanes(pred, layer=-1, conf_thresh=conf_thresh,
                            num_rows=self.num_rows, img_w=img_w, img_h=img_h)
