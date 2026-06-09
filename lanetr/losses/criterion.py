"""Criterion de LaneTR (Paso 4.3): la pérdida total de entrenamiento.

Para CADA capa del decoder (pérdidas auxiliares estilo DETR):
  1. matching húngaro → empareja queries con carriles GT (1-a-1).
  2. clasificación (focal): queries emparejadas → "carril" (1), el resto → "no-carril" (0).
  3. geometría sobre las parejas: LaneIoU + L1(xs) + L1(start_y, length) [+ L1(theta), opc.]
     [+ regularizador de suavidad opcional sobre las xs].

Devuelve un dict con cada término (para registrar/visualizar) y el `total`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lane_iou import IMG_H, IMG_W, LANE_WIDTH, lane_iou_loss, line_iou_loss
from .matcher import HungarianMatcher


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="sum"):
    """Focal loss binaria (sigmoide). logits/targets: misma forma."""
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t).pow(gamma)
    if alpha >= 0:
        a_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = a_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def prepare_targets(batch_targets, device) -> list[dict]:
    """Convierte los targets del dataset (numpy, de `encode_sample`) a tensores torch en
    `device`, con las claves que usan matcher y criterion."""
    out = []
    for t in batch_targets:
        out.append({
            "xs": torch.as_tensor(t["xs"], dtype=torch.float32, device=device),
            "valid": torch.as_tensor(t["valid"], dtype=torch.bool, device=device),
            "start_y": torch.as_tensor(t["start"][:, 1], dtype=torch.float32, device=device),
            "length": torch.as_tensor(t["length"], dtype=torch.float32, device=device),
            "theta": torch.as_tensor(t["theta"], dtype=torch.float32, device=device),
        })
    return out


class LaneCriterion(nn.Module):
    def __init__(self, matcher: HungarianMatcher | None = None, w_cls=2.0, w_iou=2.0,
                 w_xy=0.2, w_ext=0.5, w_theta=0.0, w_smooth=0.0,
                 lane_width=LANE_WIDTH, img_w=IMG_W, img_h=IMG_H, aux_layers=True,
                 focal_alpha=0.25, focal_gamma=2.0, aux_one_to_many=False, o2m_k=4,
                 geo_metric="laneiou"):
        super().__init__()
        # término geométrico: "laneiou" (tesis), "lineiou" (ablation), "distance" (L1 simple)
        self.geo_metric = geo_metric
        self.matcher = matcher or HungarianMatcher(w_cls=w_cls, w_iou=w_iou, w_ext=w_ext,
                                                   lane_width=lane_width, img_w=img_w, img_h=img_h,
                                                   geo_metric=geo_metric)
        self.w_cls, self.w_iou, self.w_xy = w_cls, w_iou, w_xy
        self.w_ext, self.w_theta, self.w_smooth = w_ext, w_theta, w_smooth
        self.lane_width, self.img_w, self.img_h = lane_width, img_w, img_h
        self.aux_layers = aux_layers
        self.focal_alpha, self.focal_gamma = focal_alpha, focal_gamma
        # asignación auxiliar uno-a-muchos en capas tempranas (one-to-one en la última)
        self.aux_one_to_many = aux_one_to_many
        self.o2m_k = o2m_k

    def _layer_loss(self, pred_l, targets, matches) -> dict:
        B, NQ = pred_l["conf"].shape
        device = pred_l["conf"].device

        labels = torch.zeros(B, NQ, device=device)
        z = torch.zeros((), device=device)
        iou_l, xy_l, ext_l, th_l, sm_l = z, z, z, z, z
        num = 0
        for b, (q, g) in enumerate(matches):
            if len(q) == 0:
                continue
            labels[b, q] = 1.0
            num += len(q)
            pxs, gxs = pred_l["xs"][b][q], targets[b]["xs"][g]
            gv = targets[b]["valid"][g]
            d = (pxs - gxs).abs().masked_fill(~gv, 0.0)
            l1_per_lane = (d.sum(-1) / gv.sum(-1).clamp(min=1)).sum()
            xy_l = xy_l + l1_per_lane
            if self.geo_metric == "laneiou":
                iou_l = iou_l + lane_iou_loss(pxs, gxs, gv, self.lane_width, self.img_w,
                                              self.img_h, reduction="sum")
            elif self.geo_metric == "lineiou":
                iou_l = iou_l + line_iou_loss(pxs, gxs, gv, reduction="sum")
            else:  # "distance": el término geométrico es la distancia L1 (ablation)
                iou_l = iou_l + l1_per_lane
            ext_l = ext_l + (pred_l["start_y"][b][q] - targets[b]["start_y"][g]).abs().sum()
            ext_l = ext_l + (pred_l["length"][b][q] - targets[b]["length"][g]).abs().sum()
            th_l = th_l + (pred_l["theta"][b][q] - targets[b]["theta"][g]).abs().sum()
            if self.w_smooth > 0:
                second = pxs[:, 2:] - 2 * pxs[:, 1:-1] + pxs[:, :-2]
                sm_l = sm_l + second.abs().mean(dim=-1).sum()
        n = max(num, 1)

        cls_l = sigmoid_focal_loss(pred_l["conf"], labels, self.focal_alpha,
                                   self.focal_gamma, reduction="sum") / n
        out = {
            "cls": self.w_cls * cls_l,
            "iou": self.w_iou * iou_l / n,
            "xy": self.w_xy * xy_l / n,
            "ext": self.w_ext * ext_l / n,
        }
        if self.w_theta > 0:
            out["theta"] = self.w_theta * th_l / n
        if self.w_smooth > 0:
            out["smooth"] = self.w_smooth * sm_l / n
        return out

    def _match_layer(self, pred_l, targets, one_to_many: bool):
        """Calcula el emparejamiento de una capa (uno-a-muchos o uno-a-uno)."""
        if one_to_many and hasattr(self.matcher, "match_one_to_many"):
            B = pred_l["conf"].shape[0]
            return [self.matcher.match_one_to_many({k: v[b] for k, v in pred_l.items()},
                                                   targets[b], self.o2m_k) for b in range(B)]
        return self.matcher.match(pred_l, targets)

    def forward(self, pred, targets) -> dict:
        """`pred`: dict de tensores (L,B,NQ,...). `targets`: lista de B dicts (tensores torch)."""
        L = pred["conf"].shape[0]
        last = L - 1
        layers = range(L) if self.aux_layers else [last]
        totals: dict[str, torch.Tensor] = {}
        for l in layers:
            pred_l = {k: v[l] for k, v in pred.items()}
            # uno-a-muchos en capas tempranas; uno-a-uno (sin NMS) en la última
            one_to_many = self.aux_one_to_many and (l != last)
            matches = self._match_layer(pred_l, targets, one_to_many)
            for k, v in self._layer_loss(pred_l, targets, matches).items():
                totals[k] = totals.get(k, 0.0) + v
        totals["total"] = sum(totals.values())
        return totals
