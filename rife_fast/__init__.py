"""rife_fast — fast RIFE frame-interpolation pipeline (CUDA + TensorRT).

Layout (src layout, so ``pip install -e .`` works):

    RIFE_FAST/
      models/     flownet.pkl (+ optional IFNet_HDv3.py override)   [git-ignored]
      engines/    TensorRT engines, one per (W, H, batch, weights)  [git-ignored]
      rife_fast/  the package
"""
from __future__ import annotations

import os
from pathlib import Path

__version__ = "1.0.0"

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent                        # .../RIFE_FAST
MODELS = Path(os.environ.get("RIFE_MODELS", ROOT / "models"))
ENGINES = Path(os.environ.get("RIFE_ENGINES", ROOT / "engines"))

for _d in (MODELS, ENGINES):
    _d.mkdir(parents=True, exist_ok=True)
