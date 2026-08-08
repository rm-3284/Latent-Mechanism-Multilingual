"""Repository path helpers."""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return Latent-Mechanism-Multilingual repo root (parent of src/)."""
    # src/lib/paths.py -> parents[0]=lib, [1]=src, [2]=repo
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def repo_root_str() -> str:
    return str(repo_root())


def data_dir_str() -> str:
    return str(data_dir())
