"""Resolución de rutas del proyecto y del dataset.

Lee la configuración de un fichero `.env` en la raíz del proyecto (sin depender de
`python-dotenv`, para que funcione sin instalar nada). Una variable de entorno real
tiene prioridad sobre el `.env`.

Uso típico:
    from lanetr import paths
    paths.dataset_dir()              # -> Path a la raíz de CULane
    paths.list_dir() / "train.txt"   # -> Path a una lista
    paths.image_path("/driver_23_30frame/.../00000.jpg")  # -> Path absoluto
"""
from __future__ import annotations

import os
from pathlib import Path

# raíz del proyecto = carpeta que contiene el paquete `lanetr`
_ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return _ROOT


def _load_dotenv() -> dict[str, str]:
    env: dict[str, str] = {}
    f = _ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


_ENV = _load_dotenv()


def get(key: str, default: str | None = None) -> str | None:
    """Variable de entorno real > .env > default."""
    return os.environ.get(key, _ENV.get(key, default))


def dataset_dir() -> Path:
    d = get("DATASET_DIR")
    if not d:
        raise RuntimeError(
            "DATASET_DIR no está definida. Añádela al .env (p. ej. DATASET_DIR=D:\\CULane) "
            "o expórtala como variable de entorno."
        )
    return Path(d)


def list_dir() -> Path:
    return dataset_dir() / "list"


def outputs_dir() -> Path:
    d = _ROOT / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def image_path(rel: str) -> Path:
    """Convierte una ruta relativa de las listas (p. ej. '/driver_.../00000.jpg')
    en una ruta absoluta dentro del dataset. Tolera separadores '/' y '\\'."""
    return dataset_dir() / str(rel).lstrip("/\\")
