"""Visualizaciones por época durante el entrenamiento (Paso 6.x).

Sobre 3 imágenes FIJAS de CULane (las mismas en todos los entrenamientos, para comparar),
genera por época y POR CADA UNA DE LAS 3 FOTOS, en `viz/epoch_XXX/img{0,1,2}/`, 4 figuras:
  - gt_vs_pred.png : GT (blanco) vs predicción (color, conf>umbral) superpuestos.
  - attention.png  : dónde mira cada query (puntos deformables o heatmap denso).
  - anchors.png    : línea-prior de cada ancla (usadas vs no usadas) + barra de confianza.
  - matcher.png    : coste del matching húngaro + asignación.
"""
from __future__ import annotations

import colorsys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from PIL import Image

from .. import paths
from ..data import culane_annotation as ann
from ..data import target_encoding as TE
from ..data import transforms as T
from ..models.head import decode_lanes


def select_fixed_images(n: int = 3) -> list[str]:
    """3 imágenes de val deterministas y diversas (4, 3 y 2 carriles)."""
    lines = [l for l in (paths.list_dir() / "val_gt.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    picks: list[str] = []
    for w in (4, 3, 2):
        for l in lines:
            _, _, ex = ann.parse_gt_line(l)
            if ex and sum(ex) == w and l not in picks:
                picks.append(l)
                break
    i = 0
    while len(picks) < n and i < len(lines):
        if lines[i] not in picks:
            picks.append(lines[i])
        i += 1
    return picks[:n]


def _palette(n):
    return [colorsys.hsv_to_rgb(i / max(n, 1), 0.9, 1.0) for i in range(n)]


class EpochVisualizer:
    def __init__(self, device="cuda", img_w=800, img_h=320, num_rows=144,
                 conf_thresh=0.5, image_lines=None):
        self.device = device
        self.img_w, self.img_h, self.num_rows = img_w, img_h, num_rows
        self.conf_thresh = conf_thresh
        self.row_ys = TE.make_row_ys(img_h, num_rows)
        lines = image_lines or select_fixed_images(3)

        cr = T.CropResize(img_w, img_h, 270)
        rng = np.random.default_rng(0)
        self.samples = []
        imgs = []
        for line in lines:
            img_rel, seg, ex = ann.parse_gt_line(line)
            a = ann.load_annotation(img_rel, ex, seg)
            pil = Image.open(paths.image_path(img_rel)).convert("RGB")
            s = {"image": pil, "lanes": [l.points.copy() for l in a.lanes],
                 "slots": [l.slot for l in a.lanes], "existence": a.existence, "meta": {}}
            s = cr(s, rng)
            rgb = np.asarray(s["image"], dtype=np.uint8).copy()
            gt = TE.encode_sample(s["lanes"], s["slots"], self.row_ys, img_w, img_h)
            gt_lanes = [TE.decode_lane(gt["xs"][i], gt["valid"][i], self.row_ys, img_w)
                        for i in range(gt["xs"].shape[0])]
            t = T.Normalize()(s, rng)["image"]
            imgs.append(t)
            self.samples.append({"rgb": rgb, "gt": gt, "gt_lanes": gt_lanes, "rel": img_rel,
                                 "name": "/".join(img_rel.split("/")[-2:])})
        self.images = torch.stack(imgs, dim=0)  # (3,3,H,W) en CPU

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def visualize(self, model, matcher, epoch: int, out_root) -> list:
        model.eval()
        dev = next(model.parameters()).device
        images = self.images.to(dev)
        pred, attn_info = model(images, return_attn=True)
        lanes = decode_lanes(pred, conf_thresh=self.conf_thresh, num_rows=self.num_rows,
                             img_w=self.img_w, img_h=self.img_h)
        epdir = out_root / f"epoch_{epoch:03d}"
        saved = []
        for i in range(len(self.samples)):          # una figura de cada tipo POR CADA foto
            d = epdir / f"img{i}"
            d.mkdir(parents=True, exist_ok=True)
            saved.append(self._gt_vs_pred(i, lanes, pred, epoch, d))
            saved.append(self._anchors(i, model, pred, epoch, d))
            saved.append(self._attention(i, model, attn_info, pred, epoch, d))
            saved.append(self._matcher(i, model, matcher, pred, epoch, d, dev))
        return [s for s in saved if s]

    def _gt_vs_pred(self, i, lanes, pred, epoch, d):
        s = self.samples[i]
        pal = _palette(pred["conf"].shape[-1])
        fig, ax = plt.subplots(figsize=(6.5, 2.7))
        ax.imshow(s["rgb"]); ax.axis("off")
        ax.set_title(f"Época {epoch} — {s['name']}: GT (blanco) vs Pred  "
                     f"({len(lanes[i])} pred / {len(s['gt_lanes'])} GT, conf>{self.conf_thresh})",
                     fontsize=9)
        for gl in s["gt_lanes"]:
            if len(gl) >= 2:
                ax.plot(gl[:, 0], gl[:, 1], color="white", lw=4, alpha=0.85)
        for lane in lanes[i]:
            p = lane["points"]
            if len(p) >= 2:
                ax.plot(p[:, 0], p[:, 1], color=pal[lane["query"]], lw=2)
        return self._save(fig, d / "gt_vs_pred.png")

    def _anchors(self, i, model, pred, epoch, d):
        if not getattr(model, "use_anchors", False):
            return None
        conf = pred["conf"][-1, i].sigmoid().cpu().numpy()       # (NQ,) de esta foto
        used = conf >= self.conf_thresh
        prior = model.anchors.prior_xs(model.row_ys, self.img_h).cpu().numpy()
        NQ = prior.shape[0]
        pal = _palette(NQ)
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 3.0), gridspec_kw={"width_ratios": [1.3, 1]})
        axL.imshow(self.samples[i]["rgb"]); axL.axis("off")
        axL.set_title("Anclas (sólida=usada, punteada=no usada)", fontsize=9)
        for q in range(NQ):
            x = prior[q] * self.img_w
            axL.plot(x, self.row_ys, "-" if used[q] else ":", color=pal[q],
                     lw=2 if used[q] else 1, alpha=0.95 if used[q] else 0.35)
        axR.bar(range(NQ), conf, color=[pal[q] for q in range(NQ)])
        axR.axhline(self.conf_thresh, color="red", ls="--", lw=1, label=f"umbral {self.conf_thresh}")
        axR.set_xlabel("query"); axR.set_ylabel("confianza"); axR.set_ylim(0, 1)
        axR.set_title(f"{int(used.sum())}/{NQ} queries usadas", fontsize=9)
        axR.legend(fontsize=7)
        fig.suptitle(f"Época {epoch} — {self.samples[i]['name']}: anclas y uso de queries", fontsize=10)
        return self._save(fig, d / "anchors.png")

    def _attention(self, i, model, attn_info, pred, epoch, d):
        attn = attn_info["attn"]
        shapes = attn_info["shapes"]
        if not attn:
            return None
        rgb = self.samples[i]["rgb"]
        # solo las queries USADAS (conf>umbral) en esta foto; si ninguna, las 4 más confiadas
        conf = pred["conf"][-1, i].sigmoid().cpu().numpy()
        used = [q for q in range(model.num_queries) if conf[q] >= self.conf_thresh]
        if not used:
            used = list(np.argsort(-conf)[:4])
        show = sorted(used, key=lambda q: -conf[q])[:6]
        pal = _palette(len(show))
        if getattr(model, "deformable", False):
            sl = attn[-1][0][i]  # última capa, imagen i: (NQ,n_heads,n_levels,n_points,2)
            fig, ax = plt.subplots(figsize=(10, 3.2))
            ax.imshow(rgb); ax.axis("off")
            for c, q in enumerate(show):
                pts = sl[q].reshape(-1, 2).cpu().numpy()
                ax.scatter(pts[:, 0] * self.img_w, pts[:, 1] * self.img_h, s=10,
                           color=pal[c], alpha=0.7, label=f"q{q}")
            ax.legend(fontsize=7, ncol=6, loc="upper center")
        else:
            a = attn[-1][i]  # (NQ, S)
            h0, w0 = shapes[0]
            maps = a[:, : h0 * w0].reshape(model.num_queries, h0, w0).cpu().numpy()
            import cv2
            cols = len(show)
            fig, axes = plt.subplots(1, cols, figsize=(2.4 * cols, 2.2))
            axes = np.atleast_1d(axes)
            for c, q in enumerate(show):
                m = maps[q]; m = (m - m.min()) / (m.max() - m.min() + 1e-9)
                hm = cv2.resize((m * 255).astype(np.uint8), (self.img_w, self.img_h))
                hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)[:, :, ::-1]
                axes[c].imshow((0.5 * rgb + 0.5 * hm).astype(np.uint8)); axes[c].axis("off")
                axes[c].set_title(f"q{q}", fontsize=8)
        fig.suptitle(f"Época {epoch} — {self.samples[i]['name']}: atención de las queries usadas "
                     f"({len(show)}) — dónde mira cada una", fontsize=10)
        return self._save(fig, d / "attention.png")

    def _matcher(self, i, model, matcher, pred, epoch, d, dev):
        gt = self.samples[i]["gt"]
        if gt["xs"].shape[0] == 0:
            return None
        tgt = {"xs": torch.as_tensor(gt["xs"], device=dev),
               "valid": torch.as_tensor(gt["valid"], device=dev),
               "start_y": torch.as_tensor(gt["start"][:, 1], device=dev),
               "length": torch.as_tensor(gt["length"], device=dev)}
        pred_b = {"conf": pred["conf"][-1, i], "xs": pred["xs"][-1, i],
                  "start_y": pred["start_y"][-1, i], "length": pred["length"][-1, i]}
        comps = matcher.cost_components(pred_b, tgt)
        q_idx, g_idx = matcher.match_one(pred_b, tgt)
        total = comps["total"].detach().float().cpu().numpy()
        G = tgt["xs"].shape[0]
        pal = _palette(G)
        fig, (axI, axM) = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1.6, 1]})
        axI.imshow(self.samples[i]["rgb"]); axI.axis("off")
        axI.set_title("GT (blanco) y predicción emparejada (color)", fontsize=9)
        xs = pred["xs"][-1, i].detach().cpu().numpy()
        for qi, gi in zip(q_idx.tolist(), g_idx.tolist()):
            gl = self.samples[i]["gt_lanes"][gi]
            if len(gl) >= 2:
                axI.plot(gl[:, 0], gl[:, 1], color="white", lw=4, alpha=0.85)
            pp = TE.decode_lane(xs[qi], gt["valid"][gi], self.row_ys, self.img_w)
            if len(pp) >= 2:
                axI.plot(pp[:, 0], pp[:, 1], color=pal[gi], lw=2)
                axI.text(pp[0, 0], pp[0, 1] + 8, f"q{qi}->gt{gi}", color=pal[gi], fontsize=7)
        im = axM.imshow(total, aspect="auto", cmap="viridis")
        axM.set_title("Coste húngaro (rojo=asignado)", fontsize=9)
        axM.set_xlabel("GT"); axM.set_ylabel("query"); axM.set_xticks(range(G))
        for qi, gi in zip(q_idx.tolist(), g_idx.tolist()):
            axM.add_patch(Rectangle((gi - 0.5, qi - 0.5), 1, 1, fill=False, edgecolor="red", lw=2))
        fig.colorbar(im, ax=axM, fraction=0.046)
        fig.suptitle(f"Época {epoch} — {self.samples[i]['name']}: matching húngaro", fontsize=10)
        return self._save(fig, d / "matcher.png")

    @staticmethod
    def _save(fig, path):
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path
