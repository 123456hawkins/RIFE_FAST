"""Faster inference front-end for RIFE 4.25/4.26 (Practical-RIFE, IFNet_HDv3).

The expensive part of this model at 720p is *not* the convolutions: the
pyramid only does ~44 GFLOP while a 4090 can do ~330 TFLOPS fp16, and the
activation traffic is ~1.4 GB.  What actually costs time is

  * 18 full-resolution ``grid_sample`` warps (3-4 ch each, 1280x768),
  * a full-resolution ``torch.cat`` of 28 channels + a full-resolution
    bilinear resize for *every* pyramid block (4 x ~440 MB of traffic),
  * ~1600 eager dispatches per forward call (launch bound at 720p),
  * a second, duplicated ``Head`` encode for the frame shared by two pairs.

This module keeps the weights and the maths of the original network but
re-arranges *where* the work happens:

  mode="ref"      : literally the original forward (A/B baseline)
  mode="assemble" : assemble each block's input at the block's own working
                    resolution instead of at full resolution (the warps are
                    still full resolution, so this is numerically ~exact)
  mode="pyramid"  : additionally warp at the block's resolution and keep
                    flow/mask/feat inside the pyramid (fastest)

All modes support
  * a batch of independent (img0, img1) pairs,
  * several timesteps per pair in a single forward pass,
  * a cached ``f1`` (= ``f0`` of the next pair) to halve the Head cost.

The public entry point is :func:`interp`.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["IFNetFast", "interp", "warp", "load_flownet"]

# ---------------------------------------------------------------------------
# warp
# ---------------------------------------------------------------------------

_GRID_CACHE: dict = {}


def _base_grid(H: int, W: int, dtype, device):
    """Pixel -> normalised coordinates (align_corners=True), cached."""
    key = (H, W, dtype, device)
    g = _GRID_CACHE.get(key)
    if g is None or g[0].device != device:
        # built in float32 then cast: identical values, fewer dtype issues
        xs = torch.arange(W, device=device, dtype=torch.float32)
        ys = torch.arange(H, device=device, dtype=torch.float32)
        gx = (xs * (2.0 / max(W - 1, 1)) - 1.0).view(1, 1, W)
        gy = (ys * (2.0 / max(H - 1, 1)) - 1.0).view(1, H, 1)
        gx = gx.to(dtype)
        gy = gy.to(dtype)
        _GRID_CACHE[key] = (gx, gy)
        return gx, gy
    return g


def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward warp, equivalent to ``model.warplayer.warp`` (<=1e-5 fp32 diff).

    Differences to the original: the base grid is cached (the original builds
    ``linspace`` every time, and caches a *batch-expanded* copy), and the grid
    is produced with ``stack`` (contiguous NCHW-style NHWC grid) instead of
    ``cat(...).permute(...)``, which saves one full-resolution temporary pass
    and feeds ``grid_sample`` a contiguous grid.
    """
    H, W = img.shape[-2], img.shape[-1]
    gx, gy = _base_grid(H, W, flow.dtype, flow.device)
    sx = 2.0 / max(W - 1, 1)
    sy = 2.0 / max(H - 1, 1)
    grid = torch.stack((gx + flow[:, 0] * sx, gy + flow[:, 1] * sy), dim=-1)
    return F.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ds(t: torch.Tensor, s: float) -> torch.Tensor:
    """Bilinear down-sample by the integer factor ``s`` (s>=1)."""
    if s == 1:
        return t
    return F.interpolate(t, scale_factor=1.0 / s, mode="bilinear", align_corners=False)


def _up(t: torch.Tensor, s: float) -> torch.Tensor:
    if s == 1:
        return t
    return F.interpolate(t, scale_factor=s, mode="bilinear", align_corners=False)


def _up2(t: torch.Tensor) -> torch.Tensor:
    return F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)


def _fit(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Resize only when the exact size does not match (odd resolutions)."""
    if t.shape[-2] == h and t.shape[-1] == w:
        return t
    return F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)


def _tile(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """[B,1,1,1] -> [B,1,h,w] view (no copy)."""
    return t.expand(t.shape[0], 1, h, w)


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------


def _resconv_fast(self, x):
    """``relu(conv(x) * beta + x)`` after ``beta`` has been folded into conv."""
    return F.leaky_relu(self.conv(x) + x, 0.2, inplace=True)


def fold_beta(net: nn.Module) -> int:
    """Fold ``ResConv.beta`` into the conv weights.

    ``relu(conv(x) * beta + x)`` == ``relu(conv_beta(x) + x)`` when the conv
    weights/bias are pre-scaled by ``beta``, so the per-layer ``mul`` kernel
    disappears (~2 kernels per ResConv, 40 ResConvs per forward).

    .. warning:: call this *after* loading the checkpoint. Loading an unfolded
       ``state_dict`` into a folded model silently corrupts the output.
    """
    n = 0
    for m in net.modules():
        if type(m).__name__ != "ResConv" or getattr(m, "_beta_folded", False):
            continue
        beta = m.beta.detach().reshape(-1)          # (c,)
        m.conv.weight.data.mul_(beta.reshape(-1, 1, 1, 1))  # per *output* channel
        if m.conv.bias is not None:
            m.conv.bias.data.mul_(beta)
        m.beta.data.fill_(1.0)
        m.forward = _resconv_fast.__get__(m)
        m._beta_folded = True
        n += 1
    return n


class _Block(nn.Module):
    """``IFBlock.forward`` without the wasted resizes when ``scale == 1``.

    With ``align_corners=False`` and ``scale_factor=1`` the bilinear resize is
    an exact copy, so dropping it is numerically identical - but it saves a
    full read+write of the block input (28 channels at the block resolution,
    i.e. ~110 MB per call at 720p for the finest block).
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x, flow=None, scale=1):
        b = self.block
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear",
                              align_corners=False)
            if flow is not None:
                flow = F.interpolate(flow, scale_factor=1.0 / scale,
                                     mode="bilinear", align_corners=False) / scale
        if flow is not None:
            x = torch.cat((x, flow), 1)
        feat = b.conv0(x)
        feat = b.convblock(feat)
        tmp = b.lastconv(feat)
        if scale != 1:
            tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear",
                                align_corners=False)
            return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]
        return tmp[:, :4], tmp[:, 4:5], tmp[:, 5:]


class IFNetFast(nn.Module):
    """Wraps a trained ``IFNet_HDv3.IFNet`` and runs it more efficiently."""

    def __init__(self, net: nn.Module, mode: str = "pyramid"):
        super().__init__()
        assert mode in ("ref", "assemble", "pyramid"), mode
        self.net = net
        self.mode = mode
        self.wrapped = nn.ModuleList([
            _Block(net.block0), _Block(net.block1), _Block(net.block2),
            _Block(net.block3), _Block(net.block4),
        ])

    # -- expose the trainer-side blocks -----------------------------------
    @property
    def _blocks(self):
        return tuple(self.wrapped)

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        return self.net.encode(img[:, :3])

    # -- main forward ------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        timestep=0.5,
        scale_list: Sequence[float] = (16, 8, 4, 2, 1),
        f0: Optional[torch.Tensor] = None,
        f1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: [B,6,H,W] (img0 | img1). Returns merged frame [B,3,H,W].

        ``timestep`` may be a scalar or a tensor of shape [B] / [B,1,1,1]
        (one value per sample), which is what makes multi-frame batching
        possible: all intermediate timesteps of a pair share the same
        ``Head`` features but have independent flow fields.
        """
        if self.mode == "ref":
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.reshape(-1, 1, 1, 1)
            return self.net(x, timestep, scale_list)[2][-1]

        net = self.net
        B, _, H, W = x.shape
        img0, img1 = x[:, :3], x[:, 3:6]
        if isinstance(timestep, torch.Tensor):
            t = timestep.to(device=x.device, dtype=x.dtype).reshape(B, 1, 1, 1)
        else:
            t = torch.full((B, 1, 1, 1), float(timestep), dtype=x.dtype, device=x.device)
        if f0 is None:
            f0 = net.encode(img0)
        if f1 is None:
            f1 = net.encode(img1)

        blocks = self._blocks
        pyramid = self.mode == "pyramid"
        full_warp = self.mode == "assemble"
        flow = mask = feat = None

        for i, s in enumerate(scale_list):
            s = float(s)
            if flow is None:
                # assemble the first block's input at its working resolution:
                # bilinear resize is per-channel, so this is *exact*.
                i0, i1 = _ds(img0, s), _ds(img1, s)
                f0s, f1s = _ds(f0, s), _ds(f1, s)
                inp = torch.cat(
                    (i0, i1, f0s, f1s, _tile(t, i0.shape[-2], i0.shape[-1])), 1
                )
                flow, mask, feat = blocks[i](inp, None, scale=1)
                hh, ww = i0.shape[-2], i0.shape[-1]
                flow, mask, feat = _fit(flow, hh, ww), _fit(mask, hh, ww), _fit(feat, hh, ww)
                if not pyramid:  # keep the accumulated flow at full resolution
                    flow = _up(flow, s) * s
                continue

            if pyramid:  # flow lives at 1/prev_s -> 1/s, already in 1/s pixels
                flow = _up2(flow) * 2
                fl = flow
            else:  # flow lives at full resolution -> convert to 1/s pixels
                fl = _ds(flow, s) / s
            mask = _up2(mask)
            feat = _up2(feat)

            i0, i1 = _ds(img0, s), _ds(img1, s)
            f0s, f1s = _ds(f0, s), _ds(f1, s)
            h, w = i0.shape[-2], i0.shape[-1]
            if full_warp:
                # keep the (visible) warps at full resolution, only the
                # concatenation is done at the block resolution
                wi0 = _ds(warp(img0, flow[:, :2]), s)
                wi1 = _ds(warp(img1, flow[:, 2:4]), s)
                wf0 = _ds(warp(f0, flow[:, :2]), s)
                wf1 = _ds(warp(f1, flow[:, 2:4]), s)
            else:
                wi0 = warp(i0, fl[:, :2])
                wi1 = warp(i1, fl[:, 2:4])
                wf0 = warp(f0s, fl[:, :2])
                wf1 = warp(f1s, fl[:, 2:4])
            inp = torch.cat((wi0, wi1, wf0, wf1, _tile(t, h, w), mask, feat), 1)
            fd, m0, feat = blocks[i](inp, fl, scale=1)
            fd, m0, feat = _fit(fd, h, w), _fit(m0, h, w), _fit(feat, h, w)
            flow = flow + fd if pyramid else flow + _up(fd, s) * s
            mask = m0

        # final resolution (scale_list[-1] is 1/scale)
        s_last = float(scale_list[-1])
        if s_last != 1:
            flow = _up(flow, s_last) * s_last
            mask = _up(mask, s_last)
        mask = torch.sigmoid(mask)
        return warp(img0, flow[:, :2]) * mask + warp(img1, flow[:, 2:4]) * (1 - mask)


# ---------------------------------------------------------------------------
# convenience: N pairs x T timesteps in one call
# ---------------------------------------------------------------------------


def interp(
    net: IFNetFast,
    img0: torch.Tensor,
    img1: torch.Tensor,
    ts: Sequence[float],
    scale: float = 1.0,
    f0: Optional[torch.Tensor] = None,
    f1: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Interpolate ``len(ts)`` frames between every (img0[k], img1[k]).

    img0/img1 : [B,3,H,W] float, same shape.
    ts        : timesteps in (0,1), e.g. [0.5] for 2x or [.25,.5,.75] for 4x.
    Returns   : [B*len(ts),3,H,W] ordered pair-major (k0t0,k0t1,...,k1t0,...).
    """
    n = len(ts)
    B = img0.shape[0]
    scale_list = [16.0 / scale, 8.0 / scale, 4.0 / scale, 2.0 / scale, 1.0 / scale]
    # f0/f1 are per-pair [B,4,H,W]; every timestep of a pair shares them
    if f0 is not None and f0.shape[0] == B:
        f0 = f0.repeat_interleave(n, 0)
    if f1 is not None and f1.shape[0] == B:
        f1 = f1.repeat_interleave(n, 0)
    if n == 1:
        x = torch.cat((img0, img1), 1)
        return net(x, float(ts[0]), scale_list, f0=f0, f1=f1)
    I0 = img0.repeat_interleave(n, 0)
    I1 = img1.repeat_interleave(n, 0)
    t = torch.tensor(ts, dtype=img0.dtype, device=img0.device).repeat(B)
    out = net(torch.cat((I0, I1), 1), t, scale_list, f0=f0, f1=f1)
    return out.view(B * n, *out.shape[1:])


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_flownet(model_dir: str | None = None, device=None, fp16: bool = False,
                 mode: str = "pyramid", fold: bool = True):
    """Load ``flownet.pkl`` into a fresh IFNet and wrap it with IFNetFast.

    ``train_log_dir`` defaults to ``rife_fast/weights``; pass a path to use a
    checkpoint that lives elsewhere (e.g. the upstream ``train_log/``).
    """

    import torch

    from rife_fast import MODELS
    from rife_fast.weights import load_ifnet_class

    model_dir = model_dir or str(MODELS)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = os.path.join(model_dir, "flownet.pkl")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"{ckpt} not found — download flownet.pkl from "
            "https://github.com/hzwer/Practical-RIFE (train_log/) into models/")
    net = load_ifnet_class()()
    sd = torch.load(ckpt, map_location="cpu")
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    # checkpoints also carry unused training-only heads (teacher / caltime)
    own = net.state_dict()
    sd = {k: v for k, v in sd.items() if k in own}
    missing = [k for k in own if k not in sd]
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} tensors: {missing[:4]}")
    net.load_state_dict(sd, strict=True)
    net.eval()
    if fold:
        fold_beta(net)
    net = net.to(device)
    if fp16:
        net = net.half()
    return IFNetFast(net, mode=mode)
