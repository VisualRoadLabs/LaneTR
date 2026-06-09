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

from ..data.target_encoding import make_row_ys
from .anchors import LaneAnchors
from .backbone import build_backbone
from .decoder import LaneDecoder
from .fpn import FPN
from .head import LaneHead, decode_lanes


class LaneTR(nn.Module):
    def __init__(self, backbone: str = "dla34", pretrained: bool = True, d_model: int = 256,
                 num_queries: int = 12, num_layers: int = 6, num_rows: int = 144,
                 nhead: int = 8, dim_ff: int = 1024, img_h: int = 320, use_anchors: bool = False,
                 deformable: bool = False, n_points: int = 4):
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained)
        self.fpn = FPN(self.backbone.out_channels, d_model)
        self.decoder = LaneDecoder(d_model=d_model, nhead=nhead, num_layers=num_layers,
                                   num_queries=num_queries, dim_ff=dim_ff,
                                   num_levels=len(self.backbone.out_channels),
                                   deformable=deformable, n_points=n_points)
        self.head = LaneHead(d_model, num_rows, residual_xs=use_anchors)
        self.num_rows = num_rows
        self.num_queries = num_queries
        self.img_h = img_h
        self.use_anchors = use_anchors
        self.deformable = deformable
        if use_anchors:
            self.anchors = LaneAnchors(num_queries, d_model)
            self.register_buffer("row_ys", torch.tensor(make_row_ys(img_h, num_rows)))

    def forward(self, images: torch.Tensor, return_attn: bool = False):
        feats = self.fpn(self.backbone(images))
        if self.use_anchors:
            ref = self.anchors.reference_points() if self.deformable else None  # (NQ,2)
            dec = self.decoder(feats, query_pos=self.anchors.pos_embed(),
                               reference_points=ref, need_attn=return_attn)
            prior = self.anchors.prior_xs(self.row_ys, self.img_h)             # (NQ,R)
            if return_attn:
                hs, attn, shapes = dec
                pred = self.head(hs, prior_xs=prior, prior_ext=self.anchors.ext_prior())
                return pred, {"attn": attn, "shapes": shapes}
            return self.head(dec, prior_xs=prior, prior_ext=self.anchors.ext_prior())
        dec = self.decoder(feats, need_attn=return_attn)
        if return_attn:
            hs, attn, shapes = dec
            return self.head(hs), {"attn": attn, "shapes": shapes}
        return self.head(dec)

    @torch.no_grad()
    def predict(self, images: torch.Tensor, conf_thresh: float | None = 0.5,
                img_w: int = 800, img_h: int = 320) -> list[list[dict]]:
        self.eval()
        pred = self.forward(images)
        return decode_lanes(pred, layer=-1, conf_thresh=conf_thresh,
                            num_rows=self.num_rows, img_w=img_w, img_h=img_h)
