"""Quality A/B: stock forward vs the net.py modes (device-agnostic).

    uv run python -m src.tools.quality
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from src import ROOT
from src.imgio import read_png, resize_rgb
from src.net import load_flownet  # noqa: E402

torch.set_grad_enabled(False)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rd(path, size):
    im = resize_rgb(read_png(path), size[0], size[1], mode="bicubic")
    return torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))).float()[None] / 255.


def psnr(a, b):
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def main():
    # mode="ref" is literally the upstream forward; fold=False so the baseline
    # keeps the per-ResConv beta multiply (the checkpoints store module.* keys,
    # so don't hand-load IFNet here -- load_flownet handles that).
    ref_net = load_flownet(None, DEV, fp16=False, mode="ref", fold=False)
    scales = [16.0, 8.0, 4.0, 2.0, 1.0]
    for W, H in [(448, 256), (1280, 720)]:
        PH = ((H - 1) // 128 + 1) * 128
        PW = ((W - 1) // 128 + 1) * 128
        i0 = F.pad(rd(os.path.join(ROOT, "assets", "I0_0.png"), (W, H)).to(DEV),
                   (0, PW - W, 0, PH - H))
        i1 = F.pad(rd(os.path.join(ROOT, "assets", "I0_1.png"), (W, H)).to(DEV),
                   (0, PW - W, 0, PH - H))
        ref = ref_net(torch.cat((i0, i1), 1), 0.5, scales)[..., :H, :W]
        print(f"\n{W}x{H} (padded {PW}x{PH})")
        for mode in ("assemble", "pyramid"):
            net = load_flownet(None, DEV, fp16=False, mode=mode)
            out = net(torch.cat((i0, i1), 1), 0.5, scales)[..., :H, :W]
            d = (out - ref).abs()
            print(f"  {mode:9s} fp32: PSNR {psnr(out, ref):6.2f} dB  "
                  f"max|d| {d.max():.4f}  px>2/255 {float((d > 2/255).float().mean())*100:.3f}%")
        if DEV.type == "cuda":
            net = load_flownet(None, DEV, fp16=True, mode="pyramid")
            out = net(torch.cat((i0, i1), 1).half(), 0.5, scales)
            out = out.float()[..., :H, :W]
            d = (out - ref).abs()
            print(f"  pyramid   fp16: PSNR {psnr(out, ref):6.2f} dB  "
                  f"max|d| {d.max():.4f}  px>2/255 {float((d > 2/255).float().mean())*100:.3f}%")


if __name__ == "__main__":
    main()
