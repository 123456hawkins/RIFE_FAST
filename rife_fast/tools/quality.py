"""Quality A/B: stock forward vs the net.py modes (device-agnostic).

    python -m rife_fast.quality
"""
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from rife_fast import MODELS, ROOT
from rife_fast.net import load_flownet  # noqa: E402

torch.set_grad_enabled(False)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rd(path, size):
    im = cv2.resize(cv2.imread(path), size, interpolation=cv2.INTER_CUBIC)
    im = im[:, :, ::-1]
    return torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))).float()[None] / 255.


def psnr(a, b):
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def main():
    for W, H in [(448, 256), (1280, 720)]:
        PH = ((H - 1) // 128 + 1) * 128
        PW = ((W - 1) // 128 + 1) * 128
        i0 = F.pad(rd(os.path.join(ROOT, "demo/I0_0.png"), (W, H)).to(DEV),
                   (0, PW - W, 0, PH - H))
        i1 = F.pad(rd(os.path.join(ROOT, "demo/I0_1.png"), (W, H)).to(DEV),
                   (0, PW - W, 0, PH - H))
        from rife_fast.weights import load_ifnet_class
        m = load_ifnet_class()()
        _sd = torch.load(str(MODELS / "flownet.pkl"), map_location="cpu")
        _own = m.state_dict()
        m.load_state_dict({k: v for k, v in _sd.items() if k in _own})
        m.eval().float().to(DEV)
        ref = m.inference(i0, i1, 0.5, 1.0)[..., :H, :W]
        print(f"\n{W}x{H} (padded {PW}x{PH})")
        for mode in ("assemble", "pyramid"):
            net = load_flownet(None, DEV, fp16=False, mode=mode)
            out = net(torch.cat((i0, i1), 1), 0.5, [16., 8., 4., 2., 1.])[..., :H, :W]
            d = (out - ref).abs()
            print(f"  {mode:9s} fp32: PSNR {psnr(out, ref):6.2f} dB  "
                  f"max|d| {d.max():.4f}  px>2/255 {float((d > 2/255).float().mean())*100:.3f}%")
        if DEV.type == "cuda":
            net = load_flownet(None, DEV, fp16=True, mode="pyramid")
            out = net(torch.cat((i0, i1), 1).half(), 0.5, [16., 8., 4., 2., 1.])
            out = out.float()[..., :H, :W]
            d = (out - ref).abs()
            print(f"  pyramid   fp16: PSNR {psnr(out, ref):6.2f} dB  "
                  f"max|d| {d.max():.4f}  px>2/255 {float((d > 2/255).float().mean())*100:.3f}%")


if __name__ == "__main__":
    main()
