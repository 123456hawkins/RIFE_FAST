"""Fused colour-conversion kernels for the frame pipeline.

The pipeline is NV12 end to end:

    nv12 -> rgb   (model input, K+1 frames)   the decoder already gives 4:2:0,
                                              this is the only RGB the model sees
    rgb  -> nv12  (K*(multi) output frames)   luma from the network, chroma from
                                              a 2x2 box of the interpolated RGB

Both are memory-bound elementwise work; the naive torch chains spend 0.2-0.7 ms
per chunk on ~12 kernels with several full-size temporaries.  These do it in one
pass each with a 3D grid (column block, row, frame) so the hot loop has no
integer div/mod.

Triton is optional: without it (or on CPU) the ``else`` branch below is used,
which is correct but slower on these passes.

Output layout of a chunk (``out`` is ``[K*multi, h*3//2, w] uint8``)::

    out[(n+1)*k]         = source frame k, copied verbatim (no colour math)
    out[(n+1)*k + j + 1] = mid frame: luma from ``mid[k*n + j]``, chroma from the
                           2x2 box average of the same interpolated RGB
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = torch.cuda.is_available()
except Exception:                                            # noqa: BLE001
    HAS_TRITON = False

# BT.601 limited range -- the matrix ffmpeg uses by default for yuv420p->rgb24,
# so a round trip reproduces the decoder's RGB to ~49 dB (the residual is chroma
# upsampling: nearest here vs bilinear in swscale).
_Y_R, _Y_G, _Y_B = 0.2568, 0.5041, 0.0979
_U_R, _U_G, _U_B = -0.1482, -0.2910, 0.4392
_V_R, _V_G, _V_B = 0.4392, -0.3678, -0.0714
_BLOCK = 512


def dst_index(pairs: int, multi: int, device) -> torch.Tensor:
    """mid-frame index -> output frame index (see the layout above)."""
    n = multi - 1
    k = torch.arange(pairs, device=device).view(pairs, 1)
    j = torch.arange(n, device=device).view(1, n)
    return ((n + 1) * k + j + 1).reshape(-1).to(torch.int32)


if HAS_TRITON:

    @triton.jit
    def _nv12_to_rgb_kernel(yuv, rgb, W: tl.constexpr, H: tl.constexpr,
                            PH: tl.constexpr, PW: tl.constexpr,
                            FS: tl.constexpr, PL: tl.constexpr,
                            BLOCK: tl.constexpr):
        col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row = tl.program_id(1)
        f = tl.program_id(2)
        m = col < W
        y = tl.load(yuv + f * FS + row * W + col, mask=m).to(tl.float32)
        uvp = f * FS + H * W + (row // 2) * W + (col // 2) * 2
        u = tl.load(yuv + uvp, mask=m).to(tl.float32) - 128.0
        v = tl.load(yuv + uvp + 1, mask=m).to(tl.float32) - 128.0
        yy = (y - 16.0) * 1.164
        r = tl.minimum(tl.maximum(yy + 1.596 * v, 0.0), 255.0) * (1.0 / 255.0)
        g = tl.minimum(tl.maximum(yy - 0.391 * u - 0.813 * v, 0.0), 255.0) * (1.0 / 255.0)
        b = tl.minimum(tl.maximum(yy + 2.018 * u, 0.0), 255.0) * (1.0 / 255.0)
        o = f * 3 * PL + row * PW + col
        tl.store(rgb + o, r.to(tl.float16), mask=m)
        tl.store(rgb + PL + o, g.to(tl.float16), mask=m)
        tl.store(rgb + 2 * PL + o, b.to(tl.float16), mask=m)

    @triton.jit
    def _rgb_to_y_kernel(mid, out, idx, W: tl.constexpr, H: tl.constexpr,
                         PH: tl.constexpr, PW: tl.constexpr,
                         OS: tl.constexpr, PL: tl.constexpr,
                         BLOCK: tl.constexpr):
        col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row = tl.program_id(1)
        f = tl.program_id(2)
        m = col < W
        p = f * 3 * PL + row * PW + col
        r = tl.load(mid + p, mask=m).to(tl.float32)
        g = tl.load(mid + PL + p, mask=m).to(tl.float32)
        b = tl.load(mid + 2 * PL + p, mask=m).to(tl.float32)
        y = r * 0.2568 + g * 0.5041 + b * 0.0979
        y = tl.minimum(tl.maximum(y * 255.0 + 16.0, 0.0), 255.0) + 0.5
        tl.store(out + tl.load(idx + f) * OS + row * W + col, y.to(tl.uint8), mask=m)

    @triton.jit
    def _rgb_to_uv_kernel(mid, out, idx, W: tl.constexpr, H: tl.constexpr,
                          PH: tl.constexpr, PW: tl.constexpr,
                          OS: tl.constexpr, PL: tl.constexpr,
                          BLOCK: tl.constexpr):
        col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)   # chroma column
        row = tl.program_id(1)                                 # chroma row
        f = tl.program_id(2)
        m = col < (W // 2)
        p = f * 3 * PL + (2 * row) * PW + 2 * col
        r = tl.zeros((BLOCK,), tl.float32)
        g = tl.zeros((BLOCK,), tl.float32)
        b = tl.zeros((BLOCK,), tl.float32)
        for dy in tl.static_range(2):
            for dx in tl.static_range(2):
                q = p + dy * PW + dx
                r += tl.load(mid + q, mask=m).to(tl.float32)
                g += tl.load(mid + PL + q, mask=m).to(tl.float32)
                b += tl.load(mid + 2 * PL + q, mask=m).to(tl.float32)
        r, g, b = r * 0.25, g * 0.25, b * 0.25
        u = (-0.1482 * r - 0.2910 * g + 0.4392 * b) * 255.0 + 128.0
        v = (0.4392 * r - 0.3678 * g - 0.0714 * b) * 255.0 + 128.0
        u = tl.minimum(tl.maximum(u, 0.0), 255.0) + 0.5
        v = tl.minimum(tl.maximum(v, 0.0), 255.0) + 0.5
        o = tl.load(idx + f) * OS + H * W + row * W + 2 * col
        tl.store(out + o, u.to(tl.uint8), mask=m)
        tl.store(out + o + 1, v.to(tl.uint8), mask=m)

    def _grid(w, h, f):
        return (triton.cdiv(w, _BLOCK), h, f)

    def nv12_to_rgb(dst, yuv, h, w):
        """``yuv`` [F, h*3/2, w] uint8 -> ``dst`` [F,3,PH,PW] fp16 in 0..1."""
        _nv12_to_rgb_kernel[_grid(w, h, dst.shape[0])](
            yuv, dst, w, h, dst.shape[2], dst.shape[3],
            yuv.shape[1] * yuv.shape[2], dst.shape[2] * dst.shape[3], _BLOCK)

    def rgb_to_nv12(out, mid, src, idx, n, h, w):
        """Fill ``out`` (see module docstring); ``src`` is [K+1, 1.5h, w] uint8."""
        k_frames = src.shape[0] - 1
        osh, osw = out.shape[1], out.shape[2]
        out.view(k_frames, n + 1, osh, osw)[:, 0].copy_(src[:k_frames])
        b = mid.shape[0]
        pl = mid.shape[2] * mid.shape[3]
        _rgb_to_y_kernel[_grid(w, h, b)](
            mid, out, idx, w, h, mid.shape[2], mid.shape[3], osh * osw, pl, _BLOCK)
        _rgb_to_uv_kernel[_grid(w // 2, h // 2, b)](
            mid, out, idx, w, h, mid.shape[2], mid.shape[3], osh * osw, pl, _BLOCK)

else:                                                        # pragma: no cover

    def nv12_to_rgb(dst, yuv, h, w):
        f = yuv.shape[0]
        y = yuv[:, :h].to(torch.float16).sub_(16.0)
        uv = yuv[:, h:].view(f, h // 2, w // 2, 2).to(torch.float16).sub_(128.0)
        u = uv[..., 0].repeat_interleave(2, 1).repeat_interleave(2, 2)
        v = uv[..., 1].repeat_interleave(2, 1).repeat_interleave(2, 2)
        dst[:, 0, :h, :w] = (y * 1.164 + v * 1.596).clamp_(0, 255).mul_(1 / 255)
        dst[:, 1, :h, :w] = (y * 1.164 - u * 0.391 - v * 0.813).clamp_(
            0, 255).mul_(1 / 255)
        dst[:, 2, :h, :w] = (y * 1.164 + u * 2.018).clamp_(0, 255).mul_(1 / 255)

    def rgb_to_nv12(out, mid, src, idx, n, h, w):
        k_frames = src.shape[0] - 1
        osh, osw = out.shape[1], out.shape[2]
        v = out.view(k_frames, n + 1, osh, osw)
        v[:, 0].copy_(src[:k_frames])
        rgb = mid.view(k_frames, n, 3, mid.shape[2], mid.shape[3])[:, :, :, :h, :w]
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        y = (r * _Y_R + g * _Y_G + b * _Y_B).mul_(255.0).add_(16.0)
        v[:, 1:, :h].copy_(y.clamp_(0, 255).add_(0.5).to(torch.uint8))
        m = rgb.reshape(k_frames, n, 3, h // 2, 2, w // 2, 2).float().mean((4, 6))
        u = (m[:, :, 0] * _U_R + m[:, :, 1] * _U_G + m[:, :, 2] * _U_B).mul_(
            255.0).add_(128.0)
        vv = (m[:, :, 0] * _V_R + m[:, :, 1] * _V_G + m[:, :, 2] * _V_B).mul_(
            255.0).add_(128.0)
        uv = torch.stack((u, vv), -1).clamp_(0, 255).add_(0.5).to(torch.uint8)
        v[:, 1:, h:].copy_(uv.view(k_frames, n, h // 2, w))
