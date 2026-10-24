"""Measure every optimisation on *your* GPU, one variable at a time.

    uv run python -m rife_fast.bench_gpu                 # 720p, 2x
    uv run python -m rife_fast.bench_gpu -m 8            # 8x (7 timesteps/pair)
    uv run python -m rife_fast.bench_gpu --compile       # include torch.compile
    uv run python -m rife_fast.bench_gpu --psnr          # + quality check

Columns:
    ms/pair   wall time per input frame pair (multi-1 frames produced)
    pairs/s   throughput; out-fps = pairs/s * multi
    speedup   relative to "orig fp32 K=1" (the stock inference_video.py path)

Nothing here touches the video I/O, so it isolates the model.
"""

import argparse
import os
import time

import torch

from rife_fast import MODELS, ROOT
from rife_fast.net import load_flownet  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)


def timeit(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


class Graph:
    """Minimal CUDA graph wrapper around a fixed-shape call."""

    def __init__(self, fn, *args):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.out = fn(*args)
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.out = fn(*args)
        self.args = args

    def __call__(self, *args):
        for a, b in zip(self.args, args):
            a.copy_(b)
        self.g.replay()
        return self.out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--multi", type=int, default=2)
    p.add_argument("--h", type=int, default=720)
    p.add_argument("--w", type=int, default=1280)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--batch", default="1,2,4,8", help="comma list, e.g. 1,2,4,8")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--psnr", action="store_true")
    p.add_argument("--channels-last", action="store_true",
                   help="also measure the pyramid forward in NHWC")
    args = p.parse_args()

    m = args.multi
    ts = [(i + 1) / m for i in range(m - 1)]
    Ks = [int(v) for v in str(args.batch).split(",")]
    ph, pw = ((args.h - 1) // 64 + 1) * 64, ((args.w - 1) // 64 + 1) * 64
    x = torch.rand(1, 6, ph, pw, device=DEV)
    print(f"{args.w}x{args.h} -> padded {pw}x{ph}  multi={m}  device="
          f"{torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'}")
    print(f"{'config':44s} {'ms/pair':>9s} {'pairs/s':>9s} {'out-fps':>9s} {'x':>6s}")

    # ---- stock model (raw IFNet, i.e. what RIFE_HDv3.Model wraps) --------
    from rife_fast.weights import load_ifnet_class
    stock = load_ifnet_class()()
    _sd = torch.load(str(MODELS / "flownet.pkl"), map_location="cpu")
    _own = stock.state_dict()
    stock.load_state_dict({k: v for k, v in _sd.items() if k in _own})
    stock.eval().to(DEV)
    SCALES = [16, 8, 4, 2, 1]

    def stock_infer(x0, x1, t=0.5):
        return stock(torch.cat((x0, x1), 1), t, SCALES)[2][3]

    base = None

    def report(name, dt):
        nonlocal base
        pairs = 1.0 / dt
        if base is None:
            base = dt
        print(f"{name:44s} {dt*1e3:9.2f} {pairs:9.1f} {pairs*m:9.1f} "
              f"{base/dt:5.2f}x")

    def orig(x0, x1, mul=1):
        return [stock_infer(x0, x1, (i + 1) / (mul + 1) if mul > 1 else 0.5)
                for i in range(mul)]

    x0, x1 = x[:, :3], x[:, 3:]
    dt = timeit(lambda: orig(x0, x1, m - 1), args.iters)
    report(f"orig fp32 eager (stock) multi={m}", dt)

    from rife_fast import warplayer as _wp
    _wp.backwarp_tenGrid.clear()  # cache is keyed by shape only
    torch.set_default_dtype(torch.float16)  # what --fp16 does in the stock script
    stock.half()
    x0h, x1h = x0.half(), x1.half()
    report(f"orig fp16        multi={m}",
           timeit(lambda: orig(x0h, x1h, m - 1), args.iters))
    stock.float()
    torch.set_default_dtype(torch.float32)
    _wp.backwarp_tenGrid.clear()

    stock.to(memory_format=torch.channels_last)
    x0c = x0.contiguous(memory_format=torch.channels_last)
    x1c = x1.contiguous(memory_format=torch.channels_last)
    report("orig fp32 channels_last",
           timeit(lambda: orig(x0c, x1c, m - 1), args.iters))
    stock.to(memory_format=torch.contiguous_format)

    # ---- net.py variants, K = 1 ----------------------------------------
    nets = {}
    for mode in ("assemble", "pyramid"):
        nets[(mode, "fp32")] = load_flownet(None, DEV,
                                            fp16=False, mode=mode)
        nets[(mode, "fp16")] = load_flownet(None, DEV,
                                            fp16=True, mode=mode)

    x0, x1 = x[:, :3], x[:, 3:]

    def run(net, K=1, graph=None, a=None, b=None, head_cache=False):
        dt = net_dtype(net)
        a = x0 if a is None else a
        b = x1 if b is None else b
        I0 = a.expand(K, -1, -1, -1).to(dt)
        I1 = b.expand(K, -1, -1, -1).to(dt)
        t = torch.tensor(ts, dtype=dt, device=DEV).repeat(K)
        n = len(ts)
        f0 = f1 = None
        if head_cache:  # one encode per distinct frame instead of one per pair
            feats = net.encode(a.expand(K + 1, -1, -1, -1).to(dt))
            f0 = feats[:K].repeat_interleave(n, 0)
            f1 = feats[1:K + 1].repeat_interleave(n, 0)
        xin = torch.cat((I0.repeat_interleave(n, 0), I1.repeat_interleave(n, 0)), 1)
        if graph is not None:
            return graph(xin, t)
        return net(xin, t, [16, 8, 4, 2, 1], f0=f0, f1=f1)

    # beta fold on/off (eager only; TRT/inductor fold it anyway)
    net_nf = load_flownet(None, DEV, fp16=True,
                          mode="pyramid", fold=False)
    report("pyramid fp16 K=1 (beta NOT folded)",
           timeit(lambda: run(net_nf, 1), args.iters))

    outs = {}
    for (mode, prec), net in nets.items():
        dt = timeit(lambda n=net: run(n), args.iters)
        report(f"net.py {mode:8s} {prec:4s} K=1", dt)
        if mode == "pyramid" and prec == "fp16":
            outs["fast"] = run(net)

    # ---- batching --------------------------------------------------------
    for K in Ks:
        if K == 1:
            continue  # already measured above
        for prec in ("fp32", "fp16"):
            net = nets[("pyramid", prec)]
            if K > 1 and prec == "fp32":
                continue
            dt = timeit(lambda n=net, k=K: run(n, k), args.iters)
            # dt is one *chunk* (K pairs); the table is per pair
            report(f"net.py pyramid {prec:4s} K={K} (batch {K*(m-1)})", dt / K)
        net = nets[("assemble", "fp16")]
        dt = timeit(lambda n=net, k=K: run(n, k), args.iters)
        report(f"net.py assemble fp16 K={K} (batch {K*(m-1)})", dt / K)

    net = nets[("pyramid", "fp16")]
    for K in Ks:
        if K == 1:
            continue
        dt = timeit(lambda n=net, k=K: run(n, k, head_cache=True), args.iters)
        report(f"pyramid fp16 K={K} + headcache", dt / K)

    # ---- CUDA graph ------------------------------------------------------
    if DEV.type == "cuda":
        for K in Ks:
            net = nets[("pyramid", "fp16")]
            B = K * len(ts)
            dtp = net_dtype(net)
            xs = torch.zeros(B, 6, ph, pw, device=DEV, dtype=dtp)
            tt = torch.zeros(B, device=DEV, dtype=dtp)
            I0 = x0.expand(K, -1, -1, -1).half()
            I1 = x1.expand(K, -1, -1, -1).half()
            xs.copy_(torch.cat((I0.repeat_interleave(len(ts), 0),
                                I1.repeat_interleave(len(ts), 0)), 1))
            tt.copy_(torch.tensor(ts, dtype=dtp, device=DEV).repeat(K))
            g = Graph(lambda a, b: net(a, b, [16, 8, 4, 2, 1]), xs, tt)
            dt = timeit(lambda: g(xs, tt), args.iters)
            report(f"net.py pyramid fp16 K={K} + cudagraph", dt / K)

    # ---- channels_last (NHWC) -------------------------------------------
    if args.channels_last:
        net = load_flownet(None, DEV, fp16=True,
                           mode="pyramid").to(memory_format=torch.channels_last)
        n = len(ts)
        for K in Ks:
            a = x0.expand(K, -1, -1, -1).half().contiguous(memory_format=torch.channels_last)
            b = x1.expand(K, -1, -1, -1).half().contiguous(memory_format=torch.channels_last)
            xin = torch.cat((a.repeat_interleave(n, 0), b.repeat_interleave(n, 0)), 1)
            xin = xin.contiguous(memory_format=torch.channels_last)
            t = torch.tensor(ts, dtype=torch.float16, device=DEV).repeat(K)
            dt = timeit(lambda: net(xin, t, [16, 8, 4, 2, 1]), args.iters)
            report(f"pyramid fp16 K={K} + channels_last", dt / K)

    # ---- torch.compile ---------------------------------------------------
    if args.compile and DEV.type == "cuda":
        for mode in ("pyramid",):
            for comp_mode in ("default", "reduce-overhead", "max-autotune"):
                net = load_flownet(None, DEV,
                                   fp16=True, mode=mode)
                try:
                    net = torch.compile(net, mode=comp_mode, dynamic=False)
                    t0 = time.perf_counter()
                    run(net, 1)
                    torch.cuda.synchronize()
                    print(f"  (compile {comp_mode} warmup {time.perf_counter()-t0:.0f}s)")
                    dt = timeit(lambda n=net: run(n, 1), args.iters)
                    report(f"net.py {mode} fp16 K=1 compile={comp_mode}", dt)
                except Exception as e:  # noqa: BLE001
                    print(f"  compile={comp_mode} failed: {type(e).__name__}: {e}")

    if args.psnr:
        # real content matters: random noise is a pathological case for any
        # multi-scale model, so use the demo pair instead.
        import math

        import cv2
        import numpy as np

        def rd(path):
            im = cv2.resize(cv2.imread(path), (pw, ph))[:, :, ::-1]
            return torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1)))[None]

        i0 = rd(os.path.join(ROOT, "assets", "I0_0.png")).to(DEV).float() / 255.0
        i1 = rd(os.path.join(ROOT, "assets", "I0_1.png")).to(DEV).float() / 255.0
        ref = stock_infer(i0, i1, 0.5)
        for prec in ("fp32", "fp16"):
            net = nets[("pyramid", prec)]
            out = run(net, 1, a=i0, b=i1)
            mse = torch.mean((out.float() - ref.float()) ** 2).item()
            print(f"pyramid {prec} vs stock fp32: PSNR "
                  f"{10*math.log10(1.0/max(mse,1e-12)):.2f} dB")

    if DEV.type == "cuda":
        print(f"peak allocated: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")


def net_dtype(net):
    return next(net.parameters()).dtype


if __name__ == "__main__":
    main()
