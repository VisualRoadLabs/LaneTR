"""Atención deformable multiescala (Deformable DETR), en PyTorch puro (Paso 5.3).

En vez de que cada query atienda a los ~5250 tokens del FPN (atención densa), atiende solo a
`n_points` puntos por nivel alrededor de un **punto de referencia** (la posición del ancla).
Cada query predice (a) los desplazamientos de esos puntos y (b) sus pesos. Los valores se
muestrean por interpolación bilineal (`F.grid_sample`) → coste independiente de la resolución.

Implementación sin kernels CUDA (usa `grid_sample`), así corre igual en Windows y en el A6000.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """Núcleo de la atención deformable (fallback PyTorch de Deformable DETR).

    value: (B, S, n_heads, head_dim);  value_spatial_shapes: lista de (H, W) por nivel;
    sampling_locations: (B, Lq, n_heads, n_levels, n_points, 2) en [0,1];
    attention_weights: (B, Lq, n_heads, n_levels, n_points).  -> (B, Lq, n_heads*head_dim).
    """
    B, _, n_heads, head_dim = value.shape
    _, Lq, _, n_levels, n_points, _ = sampling_locations.shape
    split_sizes = [int(H) * int(W) for H, W in value_spatial_shapes]
    value_list = value.split(split_sizes, dim=1)
    sampling_grids = 2 * sampling_locations - 1  # [0,1] -> [-1,1] para grid_sample
    sampling_value_list = []
    for lid, (H, W) in enumerate(value_spatial_shapes):
        H, W = int(H), int(W)
        # (B, H*W, n_heads, head_dim) -> (B*n_heads, head_dim, H, W)
        value_l = value_list[lid].flatten(2).transpose(1, 2).reshape(B * n_heads, head_dim, H, W)
        # (B, Lq, n_heads, n_points, 2) -> (B*n_heads, Lq, n_points, 2)
        grid_l = sampling_grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)
        # (B*n_heads, head_dim, Lq, n_points)
        sampled = F.grid_sample(value_l, grid_l, mode="bilinear", padding_mode="zeros",
                                align_corners=False)
        sampling_value_list.append(sampled)
    # pesos: (B, Lq, n_heads, n_levels, n_points) -> (B*n_heads, 1, Lq, n_levels*n_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(B * n_heads, 1, Lq, n_levels * n_points)
    # (B*n_heads, head_dim, Lq, n_levels, n_points) -> sum -> (B*n_heads, head_dim, Lq)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1)
    return output.view(B, n_heads * head_dim, Lq).transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    def __init__(self, d_model: int = 256, n_levels: int = 3, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model debe ser divisible por n_heads"
        self.d_model, self.n_levels, self.n_heads, self.n_points = d_model, n_levels, n_heads, n_points
        self.head_dim = d_model // n_heads
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2)
        grid = grid.repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid[:, :, i, :] *= (i + 1)
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight); nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight); nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query, reference_points, value, value_spatial_shapes, return_sampling=False):
        """query/(query+pos): (B, Lq, d);  reference_points: (B, Lq, n_levels, 2) en [0,1];
        value (memoria FPN): (B, S, d);  value_spatial_shapes: lista de (H, W). -> (B, Lq, d)."""
        B, Lq, _ = query.shape
        S = value.shape[1]
        value = self.value_proj(value).view(B, S, self.n_heads, self.head_dim)
        offsets = self.sampling_offsets(query).view(B, Lq, self.n_heads, self.n_levels, self.n_points, 2)
        attn = self.attention_weights(query).view(B, Lq, self.n_heads, self.n_levels * self.n_points)
        attn = attn.softmax(-1).view(B, Lq, self.n_heads, self.n_levels, self.n_points)

        normalizer = torch.tensor([[W, H] for H, W in value_spatial_shapes],
                                  dtype=query.dtype, device=query.device)  # (n_levels, 2) = [W,H]
        sampling_locations = (reference_points[:, :, None, :, None, :]
                              + offsets / normalizer[None, None, None, :, None, :])
        out = self.output_proj(ms_deform_attn_core_pytorch(value, value_spatial_shapes,
                                                           sampling_locations, attn))
        if return_sampling:
            return out, sampling_locations, attn
        return out
