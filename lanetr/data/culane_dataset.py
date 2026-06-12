"""Dataset / DataLoader de CULane para PyTorch (Paso 1C).

Devuelve, por muestra, la imagen ya recortada+redimensionada+normalizada como tensor, y los
carriles re-proyectados a ese espacio (todavía como polilíneas; la codificación a la
representación de filas-ancla del modelo será el Paso 1D).

Listas por split:
    train -> list/train_gt_new.txt   (FILTRADA: sin escenas de coche parado, Paso 1A)
    val   -> list/val_gt.txt
    test  -> list/test.txt
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .. import paths
from . import culane_annotation as ann
from . import transforms as T

_LIST_BY_SPLIT = {
    "train": "train_gt_new.txt",  # filtrada (Paso 1A)
    "train_full": "train_gt.txt",  # completa, por si se quiere comparar
    "val": "val_gt.txt",
    "test": "test.txt",
}


class CULaneDataset(Dataset):
    def __init__(self, split="train", img_w=800, img_h=320, cut_height=270,
                 augment=None, seed=None, list_file=None,
                 encode_targets=False, num_rows=144):
        if split not in _LIST_BY_SPLIT and list_file is None:
            raise ValueError(f"split desconocido: {split}; usa {list(_LIST_BY_SPLIT)} o list_file")
        self.split = split
        self.img_w, self.img_h, self.cut_height = img_w, img_h, cut_height
        self.seed = seed

        list_path = paths.list_dir() / (list_file or _LIST_BY_SPLIT[split])
        self.entries = [l for l in list_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.transforms = T.build_transforms(split, img_w, img_h, cut_height, augment,
                                             encode_targets=encode_targets, num_rows=num_rows)

    def __len__(self) -> int:
        return len(self.entries)

    def _rng(self, index: int) -> np.random.Generator:
        # reproducible si se fija seed; aleatorio (por época) si no.
        return np.random.default_rng(None if self.seed is None else self.seed + index)

    def __getitem__(self, index: int) -> dict:
        image_rel, seg_rel, existence = ann.parse_gt_line(self.entries[index])
        annotation = ann.load_annotation(image_rel, existence, seg_rel)

        img = Image.open(paths.image_path(image_rel)).convert("RGB")
        sample = {
            "image": img,
            "lanes": [lane.points.copy() for lane in annotation.lanes],
            "slots": [lane.slot for lane in annotation.lanes],
            "existence": annotation.existence,
            "meta": {
                "image_path": image_rel,
                "seg_path": seg_rel,
                "orig_size": img.size,  # (W, H)
                "index": index,
            },
        }
        sample = self.transforms(sample, self._rng(index))
        return sample


def collate_lanes(batch):
    """Apila las imágenes en un tensor (B,3,H,W) y mantiene los carriles como listas
    (longitud variable por imagen)."""
    out = {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "lanes": [b["lanes"] for b in batch],
        "slots": [b["slots"] for b in batch],
        "existence": [b["existence"] for b in batch],
        "meta": [b["meta"] for b in batch],
    }
    if "targets" in batch[0]:
        out["targets"] = [b["targets"] for b in batch]  # longitud variable -> lista
    return out


def build_dataloader(split="train", batch_size=8, shuffle=None, num_workers=0, seed=None,
                     curve_oversample=False, curve_alpha=4.0, curve_top_frac=0.1,
                     **ds_kwargs) -> DataLoader:
    if shuffle is None:
        shuffle = (split == "train")
    ds = CULaneDataset(split, seed=seed, **ds_kwargs)
    sampler = None
    if curve_oversample and split in ("train", "train_full"):
        # SOBRE-MUESTREO de curvas (Paso 7.3): el top `curve_top_frac` más curvo pesa ×(1+alpha).
        # Requiere list/train_curvature.npz (tools/compute_curvature.py), alineado a la lista.
        npz = paths.list_dir() / "train_curvature.npz"
        score = np.load(npz)["data"].astype(np.float64)
        if len(score) != len(ds.entries):
            raise ValueError(f"train_curvature.npz ({len(score)}) != dataset ({len(ds.entries)}); "
                             "regenera con tools/compute_curvature.py para este split")
        thr = np.quantile(score, 1.0 - curve_top_frac)
        w = 1.0 + curve_alpha * (score >= thr)                 # recto ×1, curvo ×(1+alpha)
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                                         num_samples=len(ds), replacement=True)
        shuffle = False                                        # mutuamente excluyente con sampler
    return DataLoader(ds, batch_size=batch_size, shuffle=(shuffle and sampler is None),
                      num_workers=num_workers, sampler=sampler, collate_fn=collate_lanes)
