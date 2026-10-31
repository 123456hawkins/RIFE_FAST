"""Dependency-free image input for the development tools.

``validate`` / ``quality`` / ``bench_gpu --psnr`` only need to read the two
bundled 8-bit PNGs, but ``opencv-python`` is an *optional* extra (``uv sync
--extra png``).  Requiring it made ``uv run python -m src.validate`` fail with
``ModuleNotFoundError: No module named 'cv2'`` right after a plain ``uv sync``.
A ~100 line PNG decoder (stdlib ``zlib`` + numpy) removes that dependency; the
optional OpenCV extra is no longer needed by anything in the repo.

Supported: 8-bit grayscale / RGB / grayscale+alpha / RGBA, non-interlaced.
Everything else raises a clear ``ValueError``.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _unfilter(raw: bytes, height: int, stride: int, bpp: int) -> np.ndarray:
    """Undo the per-scanline PNG filters (0 none, 1 sub, 2 up, 3 avg, 4 paeth)."""
    out = np.empty((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    pos = 0
    for y in range(height):
        ftype = raw[pos]
        pos += 1
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=pos)
        pos += stride
        if ftype == 0:                                   # none
            cur = line.copy()
        elif ftype == 2:                                 # up
            cur = (line.astype(np.uint16) + prev) & 0xFF
            cur = cur.astype(np.uint8)
        elif ftype == 1:                                 # sub (per channel)
            a = line.reshape(-1, bpp).astype(np.uint32)
            cur = (np.cumsum(a, axis=0) & 0xFF).astype(np.uint8).reshape(-1)
        elif ftype in (3, 4):                            # average / paeth
            cur = bytearray(line)
            if ftype == 3:
                for i in range(stride):
                    left = cur[i - bpp] if i >= bpp else 0
                    cur[i] = (cur[i] + ((left + int(prev[i])) >> 1)) & 0xFF
            else:
                for i in range(stride):
                    left = cur[i - bpp] if i >= bpp else 0
                    up = int(prev[i])
                    ul = int(prev[i - bpp]) if i >= bpp else 0
                    cur[i] = (cur[i] + _paeth(left, up, ul)) & 0xFF
        else:
            raise ValueError(f"unsupported PNG filter type {ftype}")
        out[y] = cur
        prev = out[y]
    return out


def read_png(path: str) -> np.ndarray:
    """Read an 8-bit PNG as ``[H, W, 3]`` uint8 RGB (alpha is dropped)."""
    with open(path, "rb") as f:
        data = f.read()
    if not data.startswith(_SIGNATURE):
        raise ValueError(f"{path}: not a PNG (this reader supports PNG only)")
    pos, idat = len(_SIGNATURE), []
    width = height = depth = ctype = interlace = None
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length                         # length + type + crc
        if kind == b"IHDR":
            width, height, depth, ctype, _, _, interlace = struct.unpack(
                ">IIBBBBB", chunk)
        elif kind == b"IDAT":
            idat.append(chunk)
        elif kind == b"IEND":
            break
    if width is None:
        raise ValueError(f"{path}: missing IHDR")
    if depth != 8:
        raise ValueError(f"{path}: only 8-bit PNGs are supported (got {depth}-bit)")
    if interlace:
        raise ValueError(f"{path}: interlaced PNGs are not supported")
    if ctype not in _CHANNELS:
        raise ValueError(f"{path}: unsupported PNG colour type {ctype}")

    channels = _CHANNELS[ctype]
    raw = zlib.decompress(b"".join(idat))
    stride = width * channels
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise ValueError(f"{path}: truncated PNG ({len(raw)} != {expected} bytes)")
    pix = _unfilter(raw, height, stride, channels).reshape(height, width, channels)

    if ctype == 2:
        return np.ascontiguousarray(pix)
    if ctype == 6:
        return np.ascontiguousarray(pix[:, :, :3])
    if ctype == 0:
        return np.ascontiguousarray(np.repeat(pix, 3, axis=2))
    # grayscale + alpha
    return np.ascontiguousarray(np.repeat(pix[:, :, :1], 3, axis=2))


def resize_rgb(rgb: np.ndarray, width: int, height: int,
               mode: str = "bilinear") -> np.ndarray:
    """Resize ``[H, W, 3]`` uint8 to ``(width, height)`` with torch."""
    if rgb.shape[1] == width and rgb.shape[0] == height:
        return rgb
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None]
    t = F.interpolate(t.float(), size=(height, width), mode=mode,
                      align_corners=False)
    return t[0].permute(1, 2, 0).round().clamp_(0, 255).to(torch.uint8).numpy()


if __name__ == "__main__":                              # tiny self-check
    import sys
    for p in sys.argv[1:]:
        a = read_png(p)
        print(f"{p}: {a.shape} {a.dtype} min={a.min()} max={a.max()}")
