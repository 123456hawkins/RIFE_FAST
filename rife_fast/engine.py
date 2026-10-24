"""TensorRT path for the RIFE pyramid forward.

TensorRT 10 has a *native* ``GridSample`` op (LINEAR / align_corners / CLAMP),
so ``F.grid_sample(..., padding_mode='border', align_corners=True)`` maps 1:1
and no custom plugin is needed.  What TRT buys us over PyTorch eager:

  * conv + LeakyReLU fused, no NCHW<->NHWC conversion kernels
    (PyTorch/cuDNN spends ~22% of the kernel time on those at 720p),
  * one tactic per layer tuned for the exact static shape,
  * a single ``execute_async_v3`` launch + our own CUDA graph.

Usage
-----
    pip install tensorrt onnx            # onnx only needed for export
    uv run python -m rife_fast.engine export --h 720 --w 1280 --batch 4
    uv run python -m rife_fast.engine build  --onnx rife_720p_b4.onnx --engine rife.engine
    uv run python -m rife_fast.engine bench  --engine rife.engine --h 720 --w 1280 --batch 4

    # then the normal video pipeline, but with the engine:
    uv run rife-fast -i in.mp4 -m 2 \
        --trt-engine rife.engine --batch-pairs 4 --encoder nvenc

Notes
-----
* Shapes are **static** (batch, H, W).  Rebuild the engine when the
  resolution/batch changes; that is why the file name carries them.
* Export in fp32, build with fp16 (``--fp16``) — TRT then keeps the fp32
  weights for the layers where fp16 would overflow.
* ``--batch`` here is *samples per forward* = ``batch_pairs * (multi-1)``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from rife_fast import ENGINES, MODELS
from rife_fast.net import IFNetFast, load_flownet  # noqa: E402


def weights_tag() -> str:
    """Short hash of the checkpoint: engines embed the weights, so the tag
    changes (and the engine is rebuilt) as soon as you swap flownet.pkl."""
    import hashlib
    path = str(MODELS / "flownet.pkl")
    if not os.path.exists(path):
        return "noweights"
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def engine_name(w: int, h: int, batch: int) -> str:
    return f"rife_{w}x{h}_b{batch}_{weights_tag()}.engine"

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


class _Wrap(torch.nn.Module):
    """img0, img1, t -> merged frame (only what TRT needs)."""

    def __init__(self, net: IFNetFast, scale: float):
        super().__init__()
        self.net = net
        self.scale_list = [16.0 / scale, 8.0 / scale, 4.0 / scale,
                           2.0 / scale, 1.0 / scale]

    def forward(self, img0, img1, t):
        return self.net(torch.cat((img0, img1), 1), t, self.scale_list)


def export(h: int, w: int, batch: int, scale: float, out: str,
           mode: str = "pyramid", fp16: bool = True):
    ph, pw = ((h - 1) // 64 + 1) * 64, ((w - 1) // 64 + 1) * 64
    net = load_flownet(None, DEV, fp16=fp16, mode=mode)
    net.eval()
    wrap = _Wrap(net, scale).eval()
    dt = torch.float16 if fp16 else torch.float32
    img0 = torch.rand(batch, 3, ph, pw, device=DEV, dtype=dt)
    img1 = torch.rand(batch, 3, ph, pw, device=DEV, dtype=dt)
    t = torch.full((batch,), 0.5, device=DEV, dtype=dt)
    with torch.no_grad():
        ref = wrap(img0, img1, t)
    kw = dict(opset_version=17, input_names=["img0", "img1", "t"],
              output_names=["out"], do_constant_folding=True)
    try:
        torch.onnx.export(wrap, (img0, img1, t), out, dynamo=False, **kw)
    except (TypeError, RuntimeError) as e:  # legacy exporter removed
        print(f"legacy exporter failed ({type(e).__name__}), retrying with dynamo")
        torch.onnx.export(wrap, (img0, img1, t), out, dynamo=True, **kw)
    print(f"wrote {out}  ({os.path.getsize(out)/2**20:.1f} MiB)  "
          f"shapes: [{batch},3,{ph},{pw}]")

    try:  # op inventory: make sure GridSample survived the export
        from collections import Counter

        import onnx
        ops = Counter(n.op_type for n in onnx.load(out).graph.node)
        print("onnx ops:", ", ".join(f"{k}x{v}" for k, v in ops.most_common(12)))
        if "GridSample" not in ops:
            print("WARNING: no GridSample node -> the exporter decomposed the "
                  "warp into Gather/Mul/Add; TRT can still run it but slower.")
    except ImportError:
        print("pip install onnx  # to see the op histogram")

    # optional numeric check
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out, providers=["CUDAExecutionProvider"])
        got = sess.run(None, {"img0": img0.cpu().numpy(),
                              "img1": img1.cpu().numpy(),
                              "t": t.cpu().numpy()})[0]
        diff = abs(got - ref.cpu().numpy()).max()
        print(f"onnxruntime check: max|onnx-torch| = {diff:.2e} (fp32)")
    except ImportError:
        print("onnxruntime not installed, skipping the numeric check")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def build(onnx_path: str, engine_path: str, fp16: bool = True,
          workspace_gb: int = 4):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    print(f"TensorRT {trt.__version__}")
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
    print(f"network input dtypes: {dtypes}")
    if getattr(trt.BuilderFlag, "FP16", None) is not None:      # TRT <= 10.x
        if fp16 and builder.platform_has_fast_fp16:
            cfg.set_flag(trt.BuilderFlag.FP16)
            print("fp16 flag enabled")
    elif trt.float16 in dtypes:
        print("fp16 comes from the strongly-typed graph (TRT >= 11)")
    if hasattr(cfg, "builder_optimization_level"):
        cfg.builder_optimization_level = 5

    t0 = time.perf_counter()
    ser = builder.build_serialized_network(network, cfg)
    if ser is None:
        sys.exit("TRT build failed (see errors above)")
    with open(engine_path, "wb") as f:
        f.write(ser)
    print(f"built {engine_path} in {time.perf_counter()-t0:.0f}s "
          f"({os.path.getsize(engine_path)/2**20:.1f} MiB)")


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------


class TRTNet:
    """Drop-in ``IFNetFast``-like callable backed by a TRT engine.

    ``__call__(x, t, scale_list=None, f0=None, f1=None) -> merged``
    ``x`` is [B,6,H,W] (img0|img1), ``t`` is [B].  The head features are
    computed inside the engine, so ``f0/f1`` are ignored.
    """

    fixed_batch = True  # the engine has a static batch dimension

    def __init__(self, engine_path: str, shape, use_graph: bool = True):
        import tensorrt as trt

        self._trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            sys.exit(f"could not deserialize {engine_path}")
        self.ctx = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i)
                      for i in range(self.engine.num_io_tensors)]
        self.shapes = shape
        # static device buffers
        self.buf = {}
        for n in self.names:
            s = tuple(self.engine.get_tensor_shape(n))
            self.buf[n] = torch.zeros(s, device=DEV,
                                      dtype=torch.float16 if self.engine.get_tensor_dtype(n) == trt.float16
                                      else torch.float32)
        for n in self.names:
            self.ctx.set_tensor_address(n, self.buf[n].data_ptr())
        self.in0, self.in1, self.t = (self.buf["img0"], self.buf["img1"],
                                      self.buf["t"])
        self.out = self.buf["out"]
        self.dtype = self.in0.dtype
        self.stream = torch.cuda.Stream()
        self.graph = None
        if use_graph:
            try:
                self._capture()
            except Exception as e:  # noqa: BLE001 - capture is an optimisation
                print(f"CUDA graph capture failed ({type(e).__name__}: {e}); "
                      "falling back to plain execute_async_v3")
                self.graph = None

    def _exec(self):
        self.ctx.execute_async_v3(self.stream.cuda_stream)

    def _capture(self):
        s = self.stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._exec()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=s):
            self._exec()

    @torch.inference_mode()
    def __call__(self, x, t, scale_list=None, f0=None, f1=None):
        self.in0.copy_(x[:, :3])
        self.in1.copy_(x[:, 3:6])
        self.t.copy_(t.reshape(-1))
        if self.graph is not None:
            self.graph.replay()
        else:
            self._exec()
        return self.out

    def encode(self, img):  # head-cache path is not supported
        raise RuntimeError("head caching is not available with --trt-engine")


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


def bench(engine_path: str, h: int, w: int, batch: int, iters: int = 100):
    ph, pw = ((h - 1) // 64 + 1) * 64, ((w - 1) // 64 + 1) * 64
    net = TRTNet(engine_path, (batch, 6, ph, pw))
    x = torch.rand(batch, 6, ph, pw, device=DEV)
    t = torch.full((batch,), 0.5, device=DEV)
    for _ in range(10):
        net(x, t)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        net(x, t)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    print(f"TRT {w}x{h} batch={batch}: {dt*1e3:.2f} ms/chunk  "
          f"{batch/dt:.0f} samples/s")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--h", type=int, default=720)
    e.add_argument("--w", type=int, default=1280)
    e.add_argument("--batch", type=int, default=4)
    e.add_argument("--scale", type=float, default=1.0)
    e.add_argument("--mode", default="pyramid")
    e.add_argument("--fp32", dest="fp16", action="store_false",
                   help="export fp32 (default is fp16 for TRT>=11)")
    e.add_argument("--out", default=None)
    b = sub.add_parser("build")
    b.add_argument("--onnx", required=True)
    b.add_argument("--engine", default=None)
    b.add_argument("--no-fp16", dest="fp16", action="store_false")
    c = sub.add_parser("bench")
    c.add_argument("--engine", required=True)
    c.add_argument("--h", type=int, default=720)
    c.add_argument("--w", type=int, default=1280)
    c.add_argument("--batch", type=int, default=4)
    c.add_argument("--iters", type=int, default=100)
    a = p.parse_args()

    if a.cmd == "bench" and not os.path.exists(a.engine):
        sys.exit(f"engine not found: {a.engine}  (run the `build` step first)")

    os.makedirs(ENGINES, exist_ok=True)
    if a.cmd == "export":
        out = a.out or os.path.join(
            str(ENGINES), engine_name(a.w, a.h, a.batch).replace(".engine", ".onnx"))
        export(a.h, a.w, a.batch, a.scale, out, a.mode, a.fp16)
    elif a.cmd == "build":
        eng = a.engine or os.path.join(
            str(ENGINES), os.path.splitext(os.path.basename(a.onnx))[0] + ".engine")
        build(a.onnx, eng, a.fp16)
    else:
        bench(a.engine, a.h, a.w, a.batch, a.iters)


if __name__ == "__main__":
    main()
