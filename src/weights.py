"""Locating and loading the RIFE network + checkpoint.

The *architecture* file (``IFNet_HDv3.py``) ships with this package so you only
have to download the **weights** (``flownet.pkl``) from the upstream
Practical-RIFE repository.  If a newer RIFE version changes the architecture,
drop its ``IFNet_HDv3.py`` next to ``flownet.pkl`` in ``models/`` and it will be
used instead of the bundled one.

Upstream's ``IFNet_HDv3.py`` does ``from model.warplayer import warp``; we
register a tiny ``model`` package that points at this module, so no editing of
the downloaded file is needed.
"""
from __future__ import annotations

import importlib.util
import sys
import types

from src import MODELS


def model_dir() -> str:
    return str(MODELS)


def checkpoint_path() -> str:
    return str(MODELS / "flownet.pkl")


def _alias_model_package() -> None:
    """Make ``from model.warplayer import warp`` resolve to our implementation."""
    from src import warplayer

    if "model" not in sys.modules:
        pkg = types.ModuleType("model")
        pkg.__path__ = []                      # type: ignore[attr-defined]
        sys.modules["model"] = pkg
    sys.modules["model.warplayer"] = warplayer
    sys.modules["model"].warplayer = warplayer   # type: ignore[attr-defined]


def ifnet_source() -> str:
    """``models/IFNet_HDv3.py`` wins over the bundled copy (for new versions)."""
    override = MODELS / "IFNet_HDv3.py"
    if override.exists():
        return str(override)
    return str(__import__("src").PKG / "IFNet_HDv3.py")


def load_ifnet_class():
    """Import ``IFNet`` from the architecture file (by path, not by name)."""
    _alias_model_package()
    path = ifnet_source()
    spec = importlib.util.spec_from_file_location("src_ifnet", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "IFNet"):
        raise RuntimeError(f"{path} does not define IFNet")
    return mod.IFNet
