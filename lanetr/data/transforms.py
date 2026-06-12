"""Transformaciones geométricas y de color para CULane.

Todas operan sobre un `sample` (dict) y transforman **a la vez** la imagen y los puntos de
los carriles, para que sigan alineados. Pipeline típico:

    train: CropResize -> RandomHorizontalFlip -> RandomAffine -> Normalize
    val:   CropResize -> Normalize

Espacio de trabajo: tras `CropResize` todo está en el espacio de la imagen final
(img_w × img_h, por defecto 800×320). La augmentación se aplica en ese espacio.

`sample` tiene las claves:
    image      : PIL.Image (RGB)  ->  tras Normalize pasa a torch.FloatTensor (3,H,W)
    lanes      : list[np.ndarray (N,2) float32]   puntos (x,y) en el espacio actual
    slots      : list[int]                         slot 0..3 por carril (paralelo a lanes)
    existence  : tuple[int,int,int,int] | None
    meta       : dict
"""
from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image

from . import target_encoding as TE

# Estadísticas de ImageNet (el backbone se preentrena con ellas).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_BILINEAR = Image.Resampling.BILINEAR


# --------------------------------------------------------------------------- #
def _affine_matrix(angle_deg, scale, tx, ty, cx, cy) -> np.ndarray:
    """Matriz 2x3 que rota+escala alrededor de (cx,cy) y traslada (tx,ty).
    Mapea punto ORIGINAL -> punto NUEVO."""
    a = math.radians(angle_deg)
    cos, sin = math.cos(a) * scale, math.sin(a) * scale
    return np.array([
        [cos, -sin, cx + tx - cos * cx + sin * cy],
        [sin,  cos, cy + ty - sin * cx - cos * cy],
    ], dtype=np.float64)


def _apply_matrix_to_lanes(lanes, M):
    out = []
    for pts in lanes:
        if len(pts) == 0:
            out.append(pts)
            continue
        homog = np.concatenate([pts, np.ones((len(pts), 1), np.float32)], axis=1)  # (N,3)
        out.append((homog @ M.T).astype(np.float32))  # (N,2)
    return out


# --------------------------------------------------------------------------- #
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample, rng):
        for t in self.transforms:
            sample = t(sample, rng)
        return sample


class CropResize:
    """Recorta la franja superior (cielo) y redimensiona al tamaño del modelo.

    region = filas [cut_height, H_orig] -> se redimensiona a (img_w, img_h).
    """

    def __init__(self, img_w=800, img_h=320, cut_height=270):
        self.img_w, self.img_h, self.cut_height = img_w, img_h, cut_height

    def __call__(self, sample, rng):
        img = sample["image"]
        W, H = img.size
        region_h = H - self.cut_height
        scale_x = self.img_w / W
        scale_y = self.img_h / region_h

        img = img.crop((0, self.cut_height, W, H)).resize((self.img_w, self.img_h), _BILINEAR)
        sample["image"] = img

        lanes = []
        for pts in sample["lanes"]:
            q = pts.copy()
            q[:, 0] = q[:, 0] * scale_x
            q[:, 1] = (q[:, 1] - self.cut_height) * scale_y
            lanes.append(q)
        sample["lanes"] = lanes
        sample["meta"]["scale"] = (scale_x, scale_y)
        sample["meta"]["img_size"] = (self.img_w, self.img_h)
        return sample


class RandomHorizontalFlip:
    """Espejo horizontal: invierte x, el orden de carriles y los flags de existencia."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample, rng):
        if rng.random() >= self.p:
            return sample
        W, _ = sample["image"].size
        sample["image"] = sample["image"].transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        lanes = []
        for pts in sample["lanes"]:
            q = pts.copy()
            q[:, 0] = (W - 1) - q[:, 0]
            lanes.append(q)

        # reordenar de izquierda a derecha por la x media e invertir la existencia
        order = sorted(range(len(lanes)), key=lambda i: float(lanes[i][:, 0].mean()))
        sample["lanes"] = [lanes[i] for i in order]
        if sample["existence"] is not None:
            flipped = tuple(sample["existence"][::-1])
            sample["existence"] = flipped
            sample["slots"] = [i for i, f in enumerate(flipped) if f == 1]
        else:
            sample["slots"] = [sample["slots"][i] for i in order]
        return sample


class RandomAffine:
    """Rotación/escala/traslación aleatoria en el espacio de la imagen final."""

    def __init__(self, degrees=6.0, scale=(0.9, 1.1), translate=(0.05, 0.05), p=0.5):
        self.degrees, self.scale, self.translate, self.p = degrees, scale, translate, p

    def __call__(self, sample, rng):
        if rng.random() >= self.p:
            return sample
        W, H = sample["image"].size
        angle = rng.uniform(-self.degrees, self.degrees)
        s = rng.uniform(self.scale[0], self.scale[1])
        tx = rng.uniform(-self.translate[0], self.translate[0]) * W
        ty = rng.uniform(-self.translate[1], self.translate[1]) * H

        M = _affine_matrix(angle, s, tx, ty, W / 2.0, H / 2.0)
        M3 = np.vstack([M, [0, 0, 1]])
        inv = np.linalg.inv(M3)
        data = (inv[0, 0], inv[0, 1], inv[0, 2], inv[1, 0], inv[1, 1], inv[1, 2])
        sample["image"] = sample["image"].transform((W, H), Image.Transform.AFFINE,
                                                     data, resample=_BILINEAR)
        sample["lanes"] = _apply_matrix_to_lanes(sample["lanes"], M)
        return sample


class Normalize:
    """Imagen PIL -> tensor (3,H,W) normalizado con estadísticas de ImageNet."""

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean, self.std = mean, std

    def __call__(self, sample, rng):
        arr = np.asarray(sample["image"], dtype=np.float32) / 255.0  # (H,W,3)
        arr = (arr - self.mean) / self.std
        sample["image"] = torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()
        return sample


class EncodeTargets:
    """Añade `sample['targets']`: la representación de filas-ancla que consume el modelo.

    Se ejecuta al final (usa `sample['lanes']`, que son arrays numpy en el espacio final).
    """

    def __init__(self, num_rows=TE.ROWS_DEFAULT, img_w=800, img_h=320):
        self.row_ys = TE.make_row_ys(img_h, num_rows)
        self.img_w, self.img_h = img_w, img_h

    def __call__(self, sample, rng):
        sample["targets"] = TE.encode_sample(sample["lanes"], sample["slots"],
                                             self.row_ys, self.img_w, self.img_h)
        return sample


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """Tensor normalizado (3,H,W) -> array uint8 (H,W,3) para visualizar."""
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * std + mean
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def build_transforms(split, img_w=800, img_h=320, cut_height=270, augment=None,
                     encode_targets=False, num_rows=TE.ROWS_DEFAULT):
    """Pipeline por split. `augment` por defecto = True solo en 'train'.
    Si `encode_targets`, añade la codificación a filas-ancla al final."""
    if augment is None:
        augment = (split == "train")
    ts = [CropResize(img_w, img_h, cut_height)]
    if augment:
        ts += [RandomHorizontalFlip(p=0.5), RandomAffine(degrees=6.0, scale=(0.9, 1.1),
                                                         translate=(0.05, 0.05), p=0.7)]
    ts += [Normalize()]
    if encode_targets:
        ts += [EncodeTargets(num_rows=num_rows, img_w=img_w, img_h=img_h)]
    return Compose(ts)
