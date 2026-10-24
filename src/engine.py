"""TensorRT path for the RIFE pyramid forward.

TensorRT 10 has a *native* ``GridSample`` op (LINEAR / align_corners / CLAMP),
so ``F.grid_sample(..., padding_mode='border', align_corners=True)`` maps 1:1
and no custom plugin is needed.  What TRT buys us over PyTorch eager:

  * conv + LeakyReLU fused, no NCHW<->NHWC conversion kernels
    (PyTorch/cuDNN spends ~22% of the kernel time on those at 720p),
  * one tactic per layer tuned for the exact static shape,
  * a single ``execute_async_v3`` launch + our own CUDA graph.

The engine is **split in two**:

  ``<name>.head.engine``  [K+1,3,PH,PW] -> [K+1,4,PH,PW]   (the Head encoder)
  ``<name>.body.engine``  i0,i1,f0,f1,t  -> out            (the IFNet pyramid)

Why: the fused export calls ``encode()`` once for ``img0`` and once for ``img1``
of *every* sample, i.e. ``2*B`` single-frame encodes per forward where ``K+1``
would do (see docs/optimization.md §3.4a; measured -13% with B=4, K=4).
Both engines are then captured into **one** CUDA graph, so there is still only
one kernel launch per chunk, and at ``multi == 2`` the body's ``i0/i1/f0/f1``
inputs are bound *by pointer* onto the input/feature buffers (``n == 1`` makes
each of them a contiguous slice), so no copy happens at all.

Usage
-----
    pip install tensorrt onnx            # onnx only needed for export
    uv run python -m src.engine export --h 720 --w 1280 --pairs 4 --multi 2
    uv run python -m src.engine build --onnx ... --engine ...
    uv run python -m src.engine bench --h 720 --w 1280 --pairs 4 --multi 2

    # then the normal video pipeline, but with the engine:
    uv run rife-fast -i in.mp4 -m 2 --trt-engine auto --encoder nvenc

Notes
-----
* Shapes are **static** (K, multi, H, W).  Rebuild when they change; that is
  why the file name carries them.
* Export in fp16 and build with the strongly-typed graph (TRT >= 11); on older
  TRT pass ``--fp16`` to the builder.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from src import ENGINES, MODELS
from src.net import IFNetFast, load_flownet  # noqa: E402


def weights_tag() -> str:
    """Short hash of the checkpoint: engines embed the weights, so the tag
    changes (and the engine is rebuilt) as soon as you swap flownet.pkl."""
    import hashlib
    path = str(MODELS / "flownet.pkl")
    if not os.path.exists(path):
        return "noweights"
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def engine_name(w: int, h: int, pairs: int, multi: int, scale: float = 1.0,
                mode: str = "pyramid") -> str:
    """Base name (without ``.head.engine`` / ``.body.engine``).

    Everything that is baked into the exported graph shows up in the name:
    ``scale`` (pyramid scale list) and ``mode`` (pyramid vs assemble vs ref)
    are only spelled out when they differ from the default, so existing engines
    keep their names.
    """
    tag = "" if float(scale) == 1.0 else f"_s{scale:g}"
    if mode != "pyramid":
        tag += f"_{mode}"
    return f"rife_{w}x{h}_k{pairs}m{multi}{tag}_{weights_tag()}"


def engine_files(w: int, h: int, pairs: int, multi: int, scale: float = 1.0,
                 mode: str = "pyramid"):
    base = os.path.join(str(ENGINES), engine_name(w, h, pairs, multi, scale, mode))
    return f"{base}.head.engine", f"{base}.body.engine"


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_LOGGER = None


def trt_logger():
    """One shared TRT logger (a second instance triggers a spurious warning)."""
    global _LOGGER
    if _LOGGER is None:
        import tensorrt as trt
        _LOGGER = trt.Logger(trt.Logger.WARNING)
    return _LOGGER


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


class _HeadWrap(torch.nn.Module):
    def __init__(self, net: IFNetFast):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net.encode(x)


class _BodyWrap(torch.nn.Module):
    """img0/img1 pairs (B samples) + their Head features -> merged frame."""

    def __init__(self, net: IFNetFast, scale: float):
        super().__init__()
        self.net = net
        self.scale_list = [16.0 / scale, 8.0 / scale, 4.0 / scale,
                           2.0 / scale, 1.0 / scale]

    def forward(self, i0, i1, f0, f1, t):
        return self.net(torch.cat((i0, i1), 1), t, self.scale_list, f0=f0, f1=f1)


def export(h: int, w: int, pairs: int, multi: int, scale: float = 1.0,
           mode: str = "pyramid", fp16: bool = True, pad_mult: int = 64):
    """Write the Head and IFNet ONNX files; returns ``(head_onnx, body_onnx)``.

    ``pairs`` is the number of frame pairs per forward (``--batch-pairs``) and
    ``multi`` the interpolation factor, so the body sees ``pairs*(multi-1)``
    samples while the head only sees ``pairs+1`` frames.
    """
    ph, pw = ((h - 1) // pad_mult + 1) * pad_mult, \
        ((w - 1) // pad_mult + 1) * pad_mult
    n = multi - 1
    batch = pairs * n
    net = load_flownet(None, DEV, fp16=fp16, mode=mode).eval()
    dt = torch.float16 if fp16 else torch.float32

    x = torch.rand(pairs + 1, 3, ph, pw, device=DEV, dtype=dt)
    i0 = torch.rand(batch, 3, ph, pw, device=DEV, dtype=dt)
    i1 = torch.rand(batch, 3, ph, pw, device=DEV, dtype=dt)
    f0 = torch.rand(batch, 4, ph, pw, device=DEV, dtype=dt)
    f1 = torch.rand(batch, 4, ph, pw, device=DEV, dtype=dt)
    t = torch.full((batch,), 0.5, device=DEV, dtype=dt)

    head_onnx, body_onnx = engine_files(pw, ph, pairs, multi, scale, mode)
    head_onnx = head_onnx.replace(".engine", ".onnx")
    body_onnx = body_onnx.replace(".engine", ".onnx")
    kw = dict(opset_version=17, do_constant_folding=True)
    with torch.no_grad():
        _export(_HeadWrap(net), (x,), head_onnx, ["x"], ["f"], **kw)
        _export(_BodyWrap(net, scale), (i0, i1, f0, f1, t), body_onnx,
                ["i0", "i1", "f0", "f1", "t"], ["o"], **kw)
    print(f"wrote {head_onnx}  ({os.path.getsize(head_onnx) / 2**20:.1f} MiB)")
    print(f"wrote {body_onnx}  ({os.path.getsize(body_onnx) / 2**20:.1f} MiB)  "
          f"head [{pairs + 1},3,{ph},{pw}] body [{batch},3,{ph},{pw}]")

    try:  # make sure GridSample survived the export
        from collections import Counter

        import onnx
        ops = Counter(nn.op_type for nn in onnx.load(body_onnx).graph.node)
        print("onnx ops:", ", ".join(f"{k}x{v}" for k, v in ops.most_common(8)))
        if "GridSample" not in ops:
            print("WARNING: no GridSample node -> the warp was decomposed into "
                  "Gather/Mul/Add; TRT can still run it but slower.")
    except ImportError:
        print("pip install onnx  # to see the op histogram")
    return head_onnx, body_onnx


def _export(mod, args, path, in_names, out_names, **kw):
    try:
        torch.onnx.export(mod, args, path, input_names=in_names,
                          output_names=out_names, dynamo=False, **kw)
    except (TypeError, RuntimeError) as e:   # legacy exporter removed
        print(f"legacy exporter failed ({type(e).__name__}), retrying with dynamo")
        torch.onnx.export(mod, args, path, input_names=in_names,
                          output_names=out_names, dynamo=True, **kw)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def build(onnx_path: str, engine_path: str, fp16: bool = True,
          workspace_gb: int = 4):
    import tensorrt as trt

    logger = trt_logger()
    builder = trt.Builder(logger)
    # explicit batch is the only mode since TRT 10, the flag was removed later
    ndcf = getattr(trt, "NetworkDefinitionCreationFlag", None)
    flags = 0
    if ndcf is not None and hasattr(ndcf, "EXPLICIT_BATCH"):
        flags |= 1 << int(ndcf.EXPLICIT_BATCH)   # TRT <= 10.x only
    if ndcf is not None and hasattr(ndcf, "STRONGLY_TYPED") and fp16:
        flags |= 1 << int(ndcf.STRONGLY_TYPED)   # honour the graph dtypes
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("TRT parse error:", parser.get_error(i))
            sys.exit(1)
    cfg = builder.create_builder_config()
    pool = getattr(trt, "MemoryPoolType", None)
    if pool is not None:
        cfg.set_memory_pool_limit(pool.WORKSPACE, workspace_gb << 30)
    else:
        cfg.max_workspace_size = workspace_gb << 30
    dtypes = {network.get_input(i).dtype for i in range(network.num_inputs)}
    if getattr(trt.BuilderFlag, "FP16", None) is not None:      # TRT <= 10.x
        if fp16 and builder.platform_has_fast_fp16:
            cfg.set_flag(trt.BuilderFlag.FP16)
    elif trt.float16 not in dtypes:
        print(f"warning: {onnx_path} is not fp16 ({dtypes})")
    if hasattr(cfg, "builder_optimization_level"):
        cfg.builder_optimization_level = 5

    t0 = time.perf_counter()
    ser = builder.build_serialized_network(network, cfg)
    if ser is None:
        sys.exit("TRT build failed (see errors above)")
    with open(engine_path, "wb") as f:
        f.write(ser)
    print(f"built {engine_path} in {time.perf_counter() - t0:.0f}s "
          f"({os.path.getsize(engine_path) / 2**20:.1f} MiB)")


def build_pair(head_onnx: str, body_onnx: str, head_engine: str,
               body_engine: str, fp16: bool = True):
    build(head_onnx, head_engine, fp16)
    build(body_onnx, body_engine, fp16)


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------


class _Ctx:
    """One TRT engine + its buffers, bound by tensor name."""

    def __init__(self, path: str):
        import tensorrt as trt
        self.trt = trt
        logger = trt_logger()
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            sys.exit(f"could not deserialize {path}")
        self.ctx = self.engine.create_execution_context()
        self.names = {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            self.names[n] = (tuple(self.engine.get_tensor_shape(n)),
                             torch.float16 if self.engine.get_tensor_dtype(n)
                             == trt.float16 else torch.float32,
                             self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.inputs = [n for n, v in self.names.items() if v[2]]
        self.outputs = [n for n, v in self.names.items() if not v[2]]
        self.buf = {n: torch.zeros(s, device=DEV, dtype=d)
                    for n, (s, d, _) in self.names.items()}
        for n in self.buf:                      # bind everything, override later
            self.bind(n)

    def bind(self, name, ptr=None):
        self.ctx.set_tensor_address(name, ptr if ptr is not None
                                    else self.buf[name].data_ptr())

    def run(self, stream=None):
        ok = self.ctx.execute_async_v3(
            stream if stream is not None else torch.cuda.current_stream().cuda_stream)
        if ok is False:                         # TRT swallows the reason; say so
            raise RuntimeError(f"execute_async_v3 failed for a {len(self.names)}-"
                               f"tensor engine ({', '.join(self.names)})")


class TRTNet:
    """Drop-in ``IFNetFast``-like callable: ``net(x, t) -> merged``.

    ``x`` is ``[K+1,3,H,W]`` fp16 (the K+1 distinct frames of the chunk),
    ``t`` is ``[B]`` with ``B = K*(multi-1)``.  Returns ``[B,3,H,W]``.

    The Head runs once over the K+1 frames (instead of 2*B times), and both
    engines are captured into a single CUDA graph.  With ``multi == 2`` the
    body's ``i0/i1/f0/f1`` are the K-frame slices of ``x``/``feats``, so they
    are bound by pointer and cost nothing.
    """

    fixed_batch = True  # the engine has a static batch dimension
    takes_frames = True  # __call__ takes the K+1 distinct frames, not i0/i1

    def __init__(self, head_engine: str, body_engine: str, use_graph: bool = True):
        self.head = _Ctx(head_engine)
        self.body = _Ctx(body_engine)
        x_shape, _, _ = self.head.names["x"]
        f_shape, _, _ = self.head.names["f"]
        o_shape, _, _ = self.body.names["o"]
        self.pairs = x_shape[0] - 1
        K = self.pairs
        self.batch = o_shape[0]
        self.n = self.batch // K                     # timesteps per pair
        self.multi = self.n + 1
        self.in_shape = tuple(x_shape)
        self.out_shape = tuple(o_shape)
        self.dtype = self.head.buf["x"].dtype
        self.t = self.body.buf["t"]

        # the body's per-sample buffers: aliased at n == 1, gathered otherwise
        self.rep = self.n > 1
        if self.rep:
            idx = torch.arange(self.batch, device=DEV) // self.n
            self.idx0, self.idx1 = idx, idx + 1
            for n_ in ("i0", "i1"):
                self.body.buf[n_] = torch.zeros(self.body.names[n_][0], device=DEV,
                                                dtype=self.dtype)
            for n_ in ("f0", "f1"):
                self.body.buf[n_] = torch.zeros(self.body.names[n_][0], device=DEV,
                                                dtype=self.dtype)
            for n_ in ("i0", "i1", "f0", "f1"):
                self.body.bind(n_)
        else:
            x0 = self.head.buf["x"].data_ptr()
            f0 = self.head.buf["f"].data_ptr()
            o3 = x_shape[1] * x_shape[2] * x_shape[3] * self.dtype.itemsize
            o4 = f_shape[1] * f_shape[2] * f_shape[3] * self.dtype.itemsize
            self.body.bind("i0", x0)
            self.body.bind("i1", x0 + o3)
            self.body.bind("f0", f0)
            self.body.bind("f1", f0 + o4)
        self.head.bind("x")
        self.head.bind("f")

        self.x = self.head.buf["x"]
        self.feats = self.head.buf["f"]
        self.out = self.body.buf["o"]
        self.graph = None
        if use_graph:
            try:
                self._capture()
            except Exception as e:  # noqa: BLE001 - capture is an optimisation
                print(f"CUDA graph capture failed ({type(e).__name__}: {e}); "
                      "falling back to two plain execute_async_v3 calls")
                self.graph = None

    # -- graph -----------------------------------------------------------
    def _launch(self):
        s = torch.cuda.current_stream().cuda_stream
        if self.rep:                       # gather the pair inputs (no dep on head)
            torch.index_select(self.x, 0, self.idx0, out=self.body.buf["i0"])
            torch.index_select(self.x, 0, self.idx1, out=self.body.buf["i1"])
        self.head.run(s)
        if self.rep:                       # gather the features the head just made
            torch.index_select(self.feats, 0, self.idx0, out=self.body.buf["f0"])
            torch.index_select(self.feats, 0, self.idx1, out=self.body.buf["f1"])
        self.body.run(s)

    def _capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._launch()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=s):
            self._launch()

    # -- io --------------------------------------------------------------
    @torch.inference_mode()
    def __call__(self, x, t, scale_list=None, f0=None, f1=None):
        self.x.copy_(x)
        self.t.copy_(t.reshape(-1))
        if self.graph is not None:
            self.graph.replay()
        else:
            self._launch()
        return self.out

    def encode(self, img):  # the Head lives inside the graph
        raise RuntimeError("head caching is not available with --trt-engine")


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


def bench(h: int, w: int, pairs: int, multi: int, iters: int = 100):
    ph, pw = ((h - 1) // 64 + 1) * 64, ((w - 1) // 64 + 1) * 64
    head, body = engine_files(pw, ph, pairs, multi)
    net = TRTNet(head, body)
    x = torch.rand(*net.in_shape, device=DEV, dtype=net.dtype)
    t = torch.full((net.batch,), 0.5, device=DEV, dtype=net.dtype)
    for _ in range(10):
        net(x, t)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        net(x, t)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    print(f"TRT {w}x{h} K={pairs} x{multi} batch={net.batch}: {dt * 1e3:.2f} ms/chunk  "
          f"{net.batch / dt:.0f} samples/s  {pairs / dt:.0f} pairs/s")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--h", type=int, default=720)
    e.add_argument("--w", type=int, default=1280)
    e.add_argument("--pairs", type=int, default=4)
    e.add_argument("--multi", type=int, default=2)
    e.add_argument("--scale", type=float, default=1.0)
    e.add_argument("--mode", default="pyramid")
    e.add_argument("--fp32", dest="fp16", action="store_false",
                   help="export fp32 (default is fp16 for TRT>=11)")
    b = sub.add_parser("build")
    b.add_argument("--onnx", required=True)
    b.add_argument("--engine", default=None)
    b.add_argument("--no-fp16", dest="fp16", action="store_false")
    c = sub.add_parser("bench")
    c.add_argument("--h", type=int, default=720)
    c.add_argument("--w", type=int, default=1280)
    c.add_argument("--pairs", type=int, default=4)
    c.add_argument("--multi", type=int, default=2)
    c.add_argument("--iters", type=int, default=100)
    a = p.parse_args()

    os.makedirs(ENGINES, exist_ok=True)
    if a.cmd == "export":
        export(a.h, a.w, a.pairs, a.multi, a.scale, a.mode, a.fp16)
    elif a.cmd == "build":
        eng = a.engine or os.path.splitext(a.onnx)[0] + ".engine"
        build(a.onnx, eng, a.fp16)
    else:
        bench(a.h, a.w, a.pairs, a.multi, a.iters)


if __name__ == "__main__":
    main()
