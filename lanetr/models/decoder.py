"""Decoder transformer tipo DETR para detección de carriles (Paso 3.2).

Idea: un conjunto FIJO de `num_queries` "fichas" (queries) que, capa a capa, se miran entre
sí (self-attention) y miran la pirámide de features del FPN (cross-attention), produciendo un
vector por query. Las cabezas (Paso 3.3) convertirán cada vector en (confianza + geometría).

De momento usa atención DENSA estándar (`nn.MultiheadAttention`). En el Paso 5 se sustituirá
por atención deformable para acelerar y estabilizar el entrenamiento.

La memoria es la concatenación de los 3 niveles del FPN aplanados, cada uno con su codificación
posicional 2D y un embedding de nivel. Para 800×320: 40·100 + 20·50 + 10·25 = 5250 tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .positional import PositionEmbeddingSine


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, dim_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    @staticmethod
    def _with_pos(t, pos):
        return t if pos is None else t + pos

    def forward(self, tgt, memory, query_pos=None, memory_pos=None, need_attn=False):
        # 1) self-attention entre queries
        q = k = self._with_pos(tgt, query_pos)
        sa, _ = self.self_attn(q, k, value=tgt, need_weights=False)
        tgt = self.norm1(tgt + self.dropout1(sa))
        # 2) cross-attention queries -> memoria (features del FPN)
        ca, attn = self.cross_attn(self._with_pos(tgt, query_pos),
                                   self._with_pos(memory, memory_pos),
                                   value=memory, need_weights=need_attn,
                                   average_attn_weights=True)
        tgt = self.norm2(tgt + self.dropout2(ca))
        # 3) feed-forward
        ff = self.linear2(self.dropout(self.act(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(ff))
        return tgt, attn


class LaneDecoder(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 6,
                 num_queries: int = 12, dim_ff: int = 1024, dropout: float = 0.1,
                 num_levels: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.num_levels = num_levels
        self.query_embed = nn.Embedding(num_queries, d_model)     # pos. de las queries
        self.level_embed = nn.Parameter(torch.zeros(num_levels, d_model))
        self.pos_enc = PositionEmbeddingSine(d_model // 2)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.level_embed, std=0.02)

    def _build_memory(self, feats):
        srcs, poss, shapes = [], [], []
        for lvl, f in enumerate(feats):
            b, c, h, w = f.shape
            shapes.append((h, w))
            pos = self.pos_enc(b, h, w, f.device, f.dtype)        # (b,c,h,w)
            src = f.flatten(2).transpose(1, 2)                    # (b,hw,c)
            pos = pos.flatten(2).transpose(1, 2) + self.level_embed[lvl].view(1, 1, -1)
            srcs.append(src)
            poss.append(pos)
        return torch.cat(srcs, dim=1), torch.cat(poss, dim=1), shapes

    def forward(self, feats, need_attn: bool = False):
        b = feats[0].shape[0]
        memory, memory_pos, shapes = self._build_memory(feats)
        query_pos = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # (b,N,d)
        tgt = torch.zeros_like(query_pos)

        outs, attns = [], []
        for layer in self.layers:
            tgt, attn = layer(tgt, memory, query_pos, memory_pos, need_attn=need_attn)
            outs.append(self.norm(tgt))
            if need_attn:
                attns.append(attn)
        hs = torch.stack(outs, dim=0)  # (num_layers, b, num_queries, d_model)
        if need_attn:
            return hs, attns, shapes
        return hs
