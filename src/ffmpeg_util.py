"""Locate ``ffmpeg`` / ``ffprobe`` and fail with an actionable message.

ffmpeg is a *system* dependency (the pip package of the same name is unrelated
and has no NVENC), so ``uv sync`` cannot install it.  A missing binary used to
surface as a bare ``FileNotFoundError: [Errno 2] ... 'ffmpeg'`` from a reader
thread, which is hard to connect to the real cause.  Besides ``PATH`` this
honours:

    FFMPEG=/path/to/ffmpeg        # file *or* directory
    FFPROBE=/path/to/ffprobe

Useful for static builds (e.g. ``ffmpeg-master-latest-linux64-gpl/bin``) that
are not on ``PATH``.  ``ffprobe`` falls back to the sibling of ``ffmpeg``.
"""
from __future__ import annotations

import functools
import os
import shutil


def _env_tool(env: str, name: str) -> str | None:
    spec = os.environ.get(env)
    if not spec:
        return None
    path = os.path.join(spec, name) if os.path.isdir(spec) else spec
    if os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    raise RuntimeError(
        f"{env}={spec!r} does not point to an executable {name} "
        f"(resolved to {path!r})")


def _install_hint(name: str) -> str:
    return (
        f"{name} not found in PATH.\n"
        f"  Install ffmpeg with NVENC support (e.g. apt install ffmpeg), or\n"
        f"  point {('FFMPEG' if name == 'ffmpeg' else 'FFPROBE')} at the binary "
        f"or its directory:\n"
        f"    export FFMPEG=/path/to/ffmpeg/bin\n"
        f"  Check with: ffmpeg -hide_banner -encoders | grep nvenc")


@functools.lru_cache(maxsize=None)
def ffmpeg() -> str:
    """Absolute path of the ffmpeg binary."""
    p = _env_tool("FFMPEG", "ffmpeg") or shutil.which("ffmpeg")
    if p:
        return p
    raise RuntimeError(_install_hint("ffmpeg"))


@functools.lru_cache(maxsize=None)
def ffprobe() -> str:
    """Absolute path of the ffprobe binary (defaults to ffmpeg's sibling)."""
    p = _env_tool("FFPROBE", "ffprobe") or shutil.which("ffprobe")
    if p:
        return p
    try:
        sibling = os.path.join(os.path.dirname(ffmpeg()), "ffprobe")
    except RuntimeError:
        sibling = ""
    if sibling and os.path.isfile(sibling) and os.access(sibling, os.X_OK):
        return sibling
    raise RuntimeError(_install_hint("ffprobe"))


def check() -> None:
    """Resolve both binaries now, so a missing tool fails before any work."""
    ffmpeg()
    ffprobe()
