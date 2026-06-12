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
                 deformable: bool = False, n_points: int = 4, n_ref_points: int = 1,
                 ref_refine: bool = False, ref_refine_mode: str = "mlp",
                 ref_y_top: float = 0.15, ref_y_bottom: float = 0.95):
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained)
        self.fpn = FPN(self.backbone.out_channels, d_model)
        self.decoder = LaneDecoder(d_model=d_model, nhead=nhead, num_layers=num_layers,
                                   num_queries=num_queries, dim_ff=dim_ff,
                                   num_levels=len(self.backbone.out_channels),
                                   deformable=deformable, n_points=n_points,
                                   n_ref_points=n_ref_points, ref_refine=ref_refine,
                                   ref_refine_mode=ref_refine_mode)
        self.head = LaneHead(d_model, num_rows, residual_xs=use_anchors)
        self.num_rows = num_rows
        self.num_queries = num_queries
        self.img_h = img_h
        self.use_anchors = use_anchors
        self.deformable = deformable
        self.n_ref_points = n_ref_points
        self.ref_refine = ref_refine
        self.ref_refine_mode = ref_refine_mode
        self.ref_y_top = ref_y_top
        self.ref_y_bottom = ref_y_bottom
        if use_anchors:
            self.anchors = LaneAnchors(num_queries, d_model)
            self.register_buffer("row_ys", torch.tensor(make_row_ys(img_h, num_rows)))

    def forward(self, images: torch.Tensor, return_attn: bool = False):
        feats = self.fpn(self.backbone(images))
        if self.use_anchors:
            # P puntos de referencia a lo largo del carril (Paso 7); P=1 -> centro (modelo orig.)
            ref_predict = None
            if self.deformable:
                ref = self.anchors.reference_points_multi(self.n_ref_points, self.ref_y_top,
                                                          self.ref_y_bottom)
                ref_ys = self.anchors.ref_heights(self.n_ref_points, self.ref_y_top, self.ref_y_bottom)
                if self.ref_refine and self.ref_refine_mode == "xs":
                    ref_predict = self._make_ref_predict(ref_ys)
            else:
                ref, ref_ys = None, None
            dec = self.decoder(feats, query_pos=self.anchors.pos_embed(), reference_points=ref,
                               ref_ys=ref_ys, ref_predict=ref_predict, need_attn=return_attn)
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

    def _make_ref_predict(self, ref_ys):
        """Callable hs_l -> x del carril PREDICHO en las alturas `ref_ys` (b,NQ,n_ref).
        Reusa el xs que la cabeza ya predice (prior + delta), leído en las filas más cercanas a
        `ref_ys`. Es la señal del modo de refinamiento "xs" (supervisada directamente por la pérdida)."""
        norm_rows = (self.row_ys.to(ref_ys.device) / (self.img_h - 1)).float()    # (num_rows,) en [0,1]
        ref_row_idx = (norm_rows[None, :] - ref_ys[:, None]).abs().argmin(dim=1)   # (n_ref,)
        prior_xs = self.anchors.prior_xs(self.row_ys, self.img_h)                  # (NQ, num_rows)

        def ref_predict(hs_l):                                    # hs_l: (b,NQ,d)
            xs_full = prior_xs.unsqueeze(0) + self.head.xs(hs_l)  # (b,NQ,num_rows) = xs de la cabeza
            return xs_full[..., ref_row_idx]                      # (b,NQ,n_ref)
        return ref_predict

    @torch.no_grad()
    def predict(self, images: torch.Tensor, conf_thresh: float | None = 0.5,
                img_w: int = 800, img_h: int = 320) -> list[list[dict]]:
        self.eval()
        pred = self.forward(images)
        return decode_lanes(pred, layer=-1, conf_thresh=conf_thresh,
                            num_rows=self.num_rows, img_w=img_w, img_h=img_h)
