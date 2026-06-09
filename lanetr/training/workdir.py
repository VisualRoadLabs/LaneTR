"""Directorio de trabajo, logging, ETA y medición de GPU para el entrenamiento (Paso 6.x).

Al entrenar se crea `work_dirs/<timestamp>/` con:
    config.yaml      copia de la config usada
    train.log        pérdidas (todas + total) + lr + ETA + época, por iteración
    eval.log         F1/P/R por evaluación
    gpu.log          uso de GPU (memoria pico + utilización media/máx) por época
    viz/epoch_XXX/   visualizaciones por época (GT vs pred, atención, anclas, matcher)
"""
from __future__ import annotations

import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

from .. import paths


def create_work_dir(cfg: dict, base: str = "work_dirs") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = paths.project_root() / base / f"{cfg.get('name', 'lanetr')}_{ts}"
    (work / "viz").mkdir(parents=True, exist_ok=True)
    (work / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                                      encoding="utf-8")
    return work


def get_logger(name: str, logfile: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def format_eta(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class ETA:
    """Estima el tiempo restante con una media móvil del tiempo por iteración."""

    def __init__(self, total_iters: int):
        self.total = total_iters
        self.avg = None
        self._t = time.time()

    def step(self, it: int) -> str:
        now = time.time()
        dt = now - self._t
        self._t = now
        self.avg = dt if self.avg is None else 0.9 * self.avg + 0.1 * dt
        return format_eta(self.avg * (self.total - it))


class GPUMonitor:
    """Mide memoria pico (torch) y utilización (%) por época. Robusto si no hay pynvml."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.enabled = device == "cuda"
        self._utils: list[float] = []
        try:
            import torch
            self.torch = torch
        except Exception:
            self.enabled = False

    def epoch_start(self):
        if self.enabled:
            self.torch.cuda.reset_peak_memory_stats()
            self._utils = []

    def _utilization(self):
        try:  # vía torch (usa pynvml por debajo)
            import torch
            return float(torch.cuda.utilization(0))
        except Exception:
            pass
        try:  # fallback: nvidia-smi
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            return float(out.stdout.strip().splitlines()[0])
        except Exception:
            return None

    def sample(self):
        if not self.enabled:
            return
        u = self._utilization()
        if u is not None:
            self._utils.append(u)

    def epoch_summary(self) -> dict:
        if not self.enabled:
            return {}
        mem_peak = self.torch.cuda.max_memory_allocated(0) / 1e9
        mem_reserved = self.torch.cuda.max_memory_reserved(0) / 1e9
        util_mean = sum(self._utils) / len(self._utils) if self._utils else None
        util_max = max(self._utils) if self._utils else None
        return {"mem_peak_gb": mem_peak, "mem_reserved_gb": mem_reserved,
                "util_mean": util_mean, "util_max": util_max}
