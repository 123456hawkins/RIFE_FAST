"""Numerical self-check. Run this after swapping in a new checkpoint.

Runs on CPU or CUDA and compares the fast paths against the stock forward:

  warp     custom back-warp vs. warplayer                    (<=1e-5 in fp32)
  ref      IFNetFast(ref) vs. the stock RIFE_HDv3.inference   (must be 0.0)
  assemble vs. ref                                           (>=120 dB)
  pyramid  vs. ref                                           (>=40 dB)
  fold     folded beta vs. unfolded                          (>=80 dB)
  batch    batched vs. per-sample loop

Usage:  uv run python -m rife_fast.validate
"""
import os
import time

import cv2
import numpy as np
import torch

from rife_fast import ROOT
from rife_fast.net import interp, load_flownet, warp  # noqa: E402
from rife_fast.warplayer import warp as warp_ref  # noqa: E402


def stock_inference(ifnet, img0, img1, timestep=0.5, scale=1.0):
    """What upstream ``RIFE_HDv3.Model.inference`` does, minus the training code."""
    scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
    _flow, _mask, merged = ifnet(torch.cat((img0, img1), 1), timestep, scale_list)
    return merged[-1]   # upstream RIFE_HDv3.Model.inference returns merged[-1]

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
# this is a bit-exactness check: TF32 would blur it (~1e-3)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
DTYPE = torch.float16 if DEV.type == "cuda" else torch.float32


def load_pair(a, b):
    def rd(p):
        im = cv2.imread(os.path.join(ROOT, p), cv2.IMREAD_COLOR)[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))).float()[None] / 255.0

    return rd(a).to(DEV), rd(b).to(DEV)


def psnr(a, b):
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def main():
    # ---- 1. warp equivalence -------------------------------------------
    x = torch.rand(2, 3, 64, 96, device=DEV)
    fl = (torch.rand(2, 2, 64, 96, device=DEV) - 0.5) * 20
    d = (warp(x, fl) - warp_ref(x, fl)).abs().max().item()
    print(f"[warp]   max|fast-orig| = {d:.3e}  (fp32 CUDA is bit-exact; cpu/fp16 may differ)")

    # ---- 2. load both models -------------------------------------------
    # fold=False so that "ref" is literally the original arithmetic
    fast = load_flownet(None, DEV, fp16=False, mode="ref", fold=False)

    i0, i1 = load_pair("assets/I0_0.png", "assets/I0_1.png")
    print(f"pair: {tuple(i0.shape)}  device={DEV}")

    ref = stock_inference(fast.net, i0, i1, 0.5, 1.0)
    fast.mode = "ref"
    r2 = fast(torch.cat((i0, i1), 1), 0.5, [16, 8, 4, 2, 1])
    print(f"[ref]    max|IFNetFast(ref)-stock inference| = {(ref-r2).abs().max().item():.3e}")

    out = {}
    for mode in ("assemble", "pyramid"):
        fast.mode = mode
        out[mode] = fast(torch.cat((i0, i1), 1), 0.5, [16, 8, 4, 2, 1])
        print(
            f"[{mode:7s}] PSNR vs ref = {psnr(out[mode], ref):6.2f} dB   "
            f"max|d| = {(out[mode]-ref).abs().max().item():.4f}"
        )

    # ---- 2b. cost of folding beta into the conv weights -----------------
    folded = load_flownet(None, DEV, fp16=False,
                          mode="pyramid", fold=True)
    unf = load_flownet(None, DEV, fp16=False,
                       mode="pyramid", fold=False)
    xin = torch.cat((i0, i1), 1)
    a = folded(xin, 0.5, [16., 8., 4., 2., 1.])
    b = unf(xin, 0.5, [16., 8., 4., 2., 1.])
    print(f"[fold]   PSNR folded vs unfolded = {psnr(a, b):6.2f} dB   "
          f"max|d| = {(a-b).abs().max().item():.4f}")

    # ---- 3. batching equivalence ---------------------------------------
    fast.mode = "pyramid"
    i0b = i0.repeat(2, 1, 1, 1)
    i1b = i1.repeat(2, 1, 1, 1)
    b = interp(fast, i0b, i1b, [0.25, 0.5, 0.75], scale=1.0)
    s = torch.cat([interp(fast, i0, i1, [0.25, 0.5, 0.75], 1.0)] * 2, 0)
    print(f"[batch]  max|batched-single| = {(b-s).abs().max().item():.3e}")

    # ---- 4. rough CPU timing -------------------------------------------
    print("\ntiming (device=%s, not representative of a 4090):" % DEV)
    for mode in ("ref", "assemble", "pyramid"):
        fast.mode = mode
        for _ in range(1):
            fast(torch.cat((i0, i1), 1), 0.5, [16, 8, 4, 2, 1])
        t0 = time.perf_counter()
        n = 3
        for _ in range(n):
            fast(torch.cat((i0, i1), 1), 0.5, [16, 8, 4, 2, 1])
        print(f"  {mode:8s}: {(time.perf_counter()-t0)/n*1e3:7.1f} ms/call")


if __name__ == "__main__":
    main()
