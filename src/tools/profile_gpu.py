"""Where do the remaining milliseconds go?  Run this on the 4090.

    uv run python -m src.tools.profile_gpu
    uv run python -m src.tools.profile_gpu --mode pyramid --fp16
    uv run python -m src.tools.profile_gpu --channels-last
    uv run python -m src.tools.profile_gpu --compile reduce-overhead

It prints
  1. the top CUDA kernels by self time, grouped into conv / warp / resize /
     concat / elementwise buckets,
  2. micro-benchmarks of the individual building blocks (a single 720p
     ``warp``, one ``Head`` encode, one block),
  3. the share of the forward spent in the encoder.

Use it after ``bench_gpu.py``: the bench tells you *which config* is fastest,
this tells you *what to attack next*.
"""
import argparse
import re
import time

import torch

from src.net import load_flownet, warp  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

CATS = [
    ("conv", r"conv|scudnn|cutlass|implicit_gemm|cudnn|winograd|gemm"),
    ("warp/grid_sample", r"grid_sampler|grid_sample"),
    ("resize", r"upsample|interpolate|resize"),
    ("concat", r"cat|CatArray"),
    ("pixel_shuffle", r"pixel_shuffle"),
    ("elementwise", r"elementwise|vectorized|activation|leaky|sigmoid|copy|fill|add|mul"),
]


def cat_of(name):
    n = name.lower()
    for c, pat in CATS:
        if re.search(pat, n):
            return c
    return "other"


def bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h", type=int, default=720)
    p.add_argument("--w", type=int, default=1280)
    p.add_argument("--mode", default="pyramid", choices=["ref", "assemble", "pyramid"])
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--fp32", dest="fp16", action="store_false")
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--compile", default="off",
                   choices=["off", "default", "reduce-overhead", "max-autotune"])
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    ph, pw = ((args.h - 1) // 64 + 1) * 64, ((args.w - 1) // 64 + 1) * 64
    net = load_flownet(None, DEV,
                       fp16=args.fp16, mode=args.mode)
    if args.channels_last:
        net = net.to(memory_format=torch.channels_last)
    if args.compile != "off":
        net = torch.compile(net, mode=args.compile, dynamic=False)

    dt = next(net.parameters()).dtype
    x = torch.rand(1, 6, ph, pw, device=DEV, dtype=dt)
    if args.channels_last:
        x = x.to(memory_format=torch.channels_last)
    scales = [16.0, 8.0, 4.0, 2.0, 1.0]
    t = torch.tensor([0.5], device=DEV, dtype=dt)

    if args.compile != "off":
        t0 = time.perf_counter()
        net(x, t, scales)
        torch.cuda.synchronize()
        print(f"compile warmup {time.perf_counter()-t0:.0f}s")

    full = bench(lambda: net(x, t, scales), args.iters)
    print(f"\n=== {args.w}x{args.h} padded {pw}x{ph}  mode={args.mode} "
          f"dtype={str(dt).replace('torch.','')}  channels_last={args.channels_last} "
          f"compile={args.compile} ===")
    print(f"forward: {full:.2f} ms  ({1000/full:.0f} pairs/s)\n")

    # ---- 1. kernel breakdown -------------------------------------------
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            net(x, t, scales)
        if DEV.type == "cuda":
            torch.cuda.synchronize()

    per_iter = 5.0
    agg = {}
    for e in prof.events():
        # only real CUDA kernels: aten:: rows report the *same* device time as
        # the kernels they launch, so counting both doubles the total
        if getattr(e, "device_type", None) != torch.autograd.DeviceType.CUDA:
            continue
        if getattr(e, "self_device_time_total", 0) > 0:
            agg[e.key] = agg.get(e.key, 0.0) + e.self_device_time_total / per_iter

    total = sum(agg.values())
    print(f"GPU kernel time: {total/1e3:.2f} ms/iter")
    print(f"{'kernel':58s} {'ms':>7s} {'%':>6s}")
    cats = {}
    for name, us in sorted(agg.items(), key=lambda kv: -kv[1])[:18]:
        print(f"{name[:58]:58s} {us/1e3:7.3f} {100*us/total:5.1f}%")
        c = cat_of(name)
        cats[c] = cats.get(c, 0) + us
    print("\nby category:")
    for c, us in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {c:18s} {us/1e3:6.3f} ms  {100*us/total:5.1f}%")
    gaps = full - total / 1e3
    print(f"\nnon-kernel (launch gaps / CPU / sync): {gaps:.2f} ms  "
          f"({100*gaps/full:.0f}%)")

    # ---- 2. building blocks --------------------------------------------
    print("\nbuilding blocks (isolated):")
    img3 = torch.rand(1, 3, ph, pw, device=DEV, dtype=dt)
    img4 = torch.rand(1, 4, ph, pw, device=DEV, dtype=dt)
    flow = torch.randn(1, 2, ph, pw, device=DEV, dtype=dt)
    print(f"  warp 3ch @full res : {bench(lambda: warp(img3, flow))*1e3:6.1f} us")
    print(f"  warp 4ch @full res : {bench(lambda: warp(img4, flow))*1e3:6.1f} us")
    print(f"  Head encode x1     : {bench(lambda: net.encode(img3))*1e3:6.1f} us")
    x6 = torch.cat((img3, img3), 1)
    w = torch.randn(96, 6, 3, 3, device=DEV, dtype=dt)
    b = torch.randn(96, device=DEV, dtype=dt)
    print(f"  conv 6->96 s2      : "
          f"{bench(lambda: torch.nn.functional.conv2d(x6, w, b, 2, 1))*1e3:6.1f} us")
    print(f"  cat 4x3ch->12ch    : "
          f"{bench(lambda: torch.cat((img3, img3, img3, img3), 1))*1e3:6.1f} us")
    print(f"  resize 28ch /4     : "
          f"{bench(lambda: torch.nn.functional.interpolate(torch.cat((img4,)*7, 1), scale_factor=0.25, mode='bilinear', align_corners=False))*1e3:6.1f} us")

    if DEV.type == "cuda":
        print(f"\npeak alloc: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
