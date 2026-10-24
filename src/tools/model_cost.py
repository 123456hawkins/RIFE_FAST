"""Cost model of one RIFE 4.25 forward at 720p.

Reports, for the stock forward and for the two restructured variants:

  1. conv FLOPs, broken down per pyramid block
  2. activation traffic of the conv/relu layers (sum of in+out bytes)
  3. traffic of the *assembly* ops only (cat / interpolate / grid_sample)
  4. number of aten dispatches (= kernel launches, the launch-overhead proxy)

Usage: uv run python -m src.model_cost [--h 720] [--w 1280] [--mode pyramid]
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from torch.utils.flop_counter import FlopCounterMode

from src.net import load_flownet  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h", type=int, default=720)
    p.add_argument("--w", type=int, default=1280)
    p.add_argument("--modes", default="ref,assemble,pyramid")
    p.add_argument("--bytes", type=int, default=4)
    args = p.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(min(32, os.cpu_count() or 8))

    PH, PW = ((args.h - 1) // 128 + 1) * 128, ((args.w - 1) // 128 + 1) * 128
    x = torch.zeros(1, 6, PH, PW)
    scales = [16.0, 8.0, 4.0, 2.0, 1.0]

    GRID_N = [0]
    _g = F.grid_sample
    F.grid_sample = lambda *a, **k: (GRID_N.__setitem__(0, GRID_N[0] + 1), _g(*a, **k))[1]

    acc = {}


    def nb(t):
        return t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0


    _cat, _int = torch.cat, F.interpolate


    def cat(ts, *a, **k):
        out = _cat(ts, *a, **k)
        acc["cat"] = acc.get("cat", 0) + sum(nb(t) for t in ts) + nb(out)
        return out


    def interp(t, *a, **k):
        out = _int(t, *a, **k)
        acc["interpolate"] = acc.get("interpolate", 0) + nb(t) + nb(out)
        return out


    torch.cat, F.interpolate = cat, interp

    rows = []
    for mode in args.modes.split(","):
        net = load_flownet(None, torch.device("cpu"),
                           fp16=False, mode=mode)

        # -- conv FLOPs per top-level block ---------------------------------
        fc = FlopCounterMode(display=False)
        with fc:
            net(x, torch.tensor([0.5]), scales)
        counts = fc.get_flop_counts()
        total_flops = max(sum(v.values()) for v in counts.values())

        # -- module activation traffic --------------------------------------
        acc.clear()
        GRID_N[0] = 0
        traffic = [0]

        def hook(mod, inp, out, _t=traffic):
            _t[0] += sum(nb(t) for t in inp)
            _t[0] += nb(out) if isinstance(out, torch.Tensor) else sum(nb(t) for t in out)

        hooks = [m.register_forward_hook(hook) for _, m in net.net.named_modules()
                 if not list(m.children())]
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            net(x, torch.tensor([0.5]), scales)
        for h in hooks:
            h.remove()
        ndisp = sum(e.count for e in prof.key_averages() if e.self_cpu_time_total > 0)
        rows.append((mode, total_flops, traffic[0], sum(acc.values()), ndisp, GRID_N[0]))

        print(f"\n=== mode={mode} ===")
        print(f"conv FLOPs        : {total_flops/1e9:7.2f} GFLOP")
        print(f"layer traffic     : {traffic[0]/1e9:7.3f} GB")
        print(f"assembly traffic  : {sum(acc.values())/1e9:7.3f} GB  "
              + "  ".join(f"{k}={v/1e9:.3f}" for k, v in sorted(acc.items())))
        print(f"aten dispatches   : {ndisp}")
        print(f"grid_sample calls : {GRID_N[0]}")

    print("\n=== summary (fp32 bytes) ===")
    print(f"{'mode':10s} {'GFLOP':>8s} {'layer GB':>9s} {'asm GB':>8s} {'total GB':>9s} "
          f"{'disp':>6s} {'warp':>5s}")
    for mode, fl, tr, asm, nd, ng in rows:
        print(f"{mode:10s} {fl/1e9:8.2f} {tr/1e9:9.3f} {asm/1e9:8.3f} {(tr+asm)/1e9:9.3f} "
              f"{nd:6d} {ng:5d}")
    print("\nRoofline at 1.0 TB/s and 2 us/dispatch (4090):")
    for mode, fl, tr, asm, nd, ng in rows:
        tot = tr + asm
        print(f"  {mode:10s} memory {tot/1e12*1e3:5.2f} ms | dispatch {nd*2e-6*1e3:5.2f} ms | "
              f"fp16 memory {tot/2/1e12*1e3:5.2f} ms")



if __name__ == "__main__":
    main()
