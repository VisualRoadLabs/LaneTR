"""Matcher húngaro para LaneTR (Paso 4.2) — la novedad de la tesis.

Empareja, 1-a-1, cada query (predicción) con un carril GT, minimizando un coste que combina:

    coste(query, gt) = w_cls·coste_focal(conf)        ← ¿hay carril?
                     + w_iou·(1 − LaneIoU)            ← posición + forma  (LaneIoU = aportación)
                     + w_xy ·L1(xs)                   ← refuerzo de posición fila a fila
                     + w_ext·L1(start_y, length)      ← extensión vertical

Se resuelve con el algoritmo húngaro (`scipy.optimize.linear_sum_assignment`). Como hay más
queries que carriles (12 > ≤4), las queries sobrantes quedan SIN emparejar → clase "no-carril".
El emparejamiento 1-a-1 es lo que elimina la necesidad de NMS.

El coste NO necesita gradiente (la asignación se hace con `torch.no_grad`).
"""
from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment

from .lane_iou import IMG_H, IMG_W, LANE_WIDTH, lane_iou_pairwise


def focal_cost(conf_logits, alpha=0.25, gamma=2.0, eps=1e-12):
    """Coste de clasificación tipo focal (DETR/CLRerNet). conf_logits: (NQ,) -> (NQ,)."""
    p = conf_logits.sigmoid()
    neg = -(1 - p + eps).log() * (1 - alpha) * p.pow(gamma)
    pos = -(p + eps).log() * alpha * (1 - p).pow(gamma)
    return pos - neg  # coste de asignar la clase "carril" a cada query


def l1_xs_cost(pred_xs, gt_xs, gt_valid):
    """L1 medio entre xs predichas y GT, sobre las filas válidas del GT. -> (NQ, G)."""
    diff = (pred_xs[:, None, :] - gt_xs[None, :, :]).abs()  # (NQ,G,R)
    valid = gt_valid[None].expand(diff.shape[0], -1, -1)
    diff = diff.masked_fill(~valid, 0.0)
    return diff.sum(-1) / valid.sum(-1).clamp(min=1)


class HungarianMatcher:
    def __init__(self, w_cls=2.0, w_iou=2.0, w_xy=0.0, w_ext=0.5,
                 lane_width=LANE_WIDTH, img_w=IMG_W, img_h=IMG_H):
        self.w_cls, self.w_iou, self.w_xy, self.w_ext = w_cls, w_iou, w_xy, w_ext
        self.lane_width, self.img_w, self.img_h = lane_width, img_w, img_h

    def cost_components(self, pred, tgt) -> dict:
        """Devuelve las matrices de coste (NQ×G) por componente y el total."""
        NQ = pred["conf"].shape[0]
        G = tgt["xs"].shape[0]
        c_cls = focal_cost(pred["conf"])[:, None].expand(NQ, G)
        iou = lane_iou_pairwise(pred["xs"], tgt["xs"], tgt["valid"],
                                self.lane_width, self.img_w, self.img_h)
        c_iou = 1.0 - iou
        c_xy = l1_xs_cost(pred["xs"], tgt["xs"], tgt["valid"])
        c_ext = ((pred["start_y"][:, None] - tgt["start_y"][None, :]).abs()
                 + (pred["length"][:, None] - tgt["length"][None, :]).abs())
        total = (self.w_cls * c_cls + self.w_iou * c_iou
                 + self.w_xy * c_xy + self.w_ext * c_ext)
        return {"cls": c_cls, "iou": c_iou, "xy": c_xy, "ext": c_ext, "total": total}

    @torch.no_grad()
    def match_one(self, pred, tgt):
        """Empareja una imagen. Devuelve (query_idx, gt_idx) (long tensors, len = nº GT)."""
        G = tgt["xs"].shape[0]
        if G == 0:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty
        cost = self.cost_components(pred, tgt)["total"].detach().cpu().numpy()
        q, g = linear_sum_assignment(cost)
        dev = pred["conf"].device
        return (torch.as_tensor(q, dtype=torch.long, device=dev),
                torch.as_tensor(g, dtype=torch.long, device=dev))

    @torch.no_grad()
    def match_one_to_many(self, pred, tgt, k: int = 4):
        """Asignación UNO-A-MUCHOS (para capas auxiliares tempranas, estilo O2SFormer).

        Cada carril GT recibe sus `k` queries de menor coste; cada query se asigna como mucho
        a un GT (al de menor coste si fuera candidata de varios). Devuelve (query_idx, gt_idx),
        donde cada GT puede aparecer hasta `k` veces.
        """
        G = tgt["xs"].shape[0]
        dev = pred["conf"].device
        if G == 0:
            empty = torch.empty(0, dtype=torch.long, device=dev)
            return empty, empty
        cost = self.cost_components(pred, tgt)["total"]   # (NQ, G)
        NQ = cost.shape[0]
        topk = min(k, NQ)
        cand = torch.zeros(NQ, G, dtype=torch.bool, device=dev)
        for g in range(G):
            idx = torch.topk(cost[:, g], topk, largest=False).indices
            cand[idx, g] = True
        q_list, g_list = [], []
        for q in range(NQ):
            gs = torch.where(cand[q])[0]
            if len(gs) == 0:
                continue
            best_g = int(gs[cost[q, gs].argmin()])
            q_list.append(q)
            g_list.append(best_g)
        return (torch.tensor(q_list, dtype=torch.long, device=dev),
                torch.tensor(g_list, dtype=torch.long, device=dev))

    @torch.no_grad()
    def match(self, pred, targets):
        """Empareja un batch.
        `pred`: dict con tensores (B, NQ, ...). `targets`: lista de B dicts (cada uno con
        tensores (G, ...)). Devuelve lista de B tuplas (query_idx, gt_idx).
        """
        B = pred["conf"].shape[0]
        out = []
        for b in range(B):
            pred_b = {k: v[b] for k, v in pred.items()}
            out.append(self.match_one(pred_b, targets[b]))
        return out
