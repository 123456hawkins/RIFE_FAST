#!/usr/bin/env python3
"""Fast RIFE interpolation CLI (720p / 1080p / 2K, RTX 30/40).

日常只需要：输入、输出、倍率，偶尔加 --codec / --quality。

    python -m rife_fast -i in.mp4 -m 2 -o out.mp4              # 默认（最快）
    python -m rife_fast -i in.mp4 -m 2 --codec h265 --quality 28   # 更小
    python -m rife_fast -i in.mp4 -m 4 -o out4x.mp4

想更快：加 `--trt-engine auto`（首次会自动构建引擎，约 2 分钟）。
其余参数都在"高级参数"里，一般不用动。详见 README.md。
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn.functional as F

from rife_fast import ENGINES
from rife_fast.net import load_flownet  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def net_dtype(net) -> torch.dtype:
    d = getattr(net, "dtype", None)
    if isinstance(d, torch.dtype):
        return d
    return next(net.parameters()).dtype


def auto_batch_pairs(multi, ph, pw):
    """Pick the batch size that measured fastest for this resolution/multiplier.

    720p: 2x->4, 4x->2, 8x->1 | 1080p: 2x->4, 4x->2 | 2K+: 2x->2, else 1
    (bigger batches stop helping once the GPU is saturated; measured on a 4090)
    """
    n = multi - 1
    px = ph * pw
    if px > 3.0e6:                    # 2K and up
        return 2 if n == 1 else 1
    if px > 1.5e6:                    # 1080p
        return 4 if n == 1 else 2
    return 4 if n <= 1 else (2 if n <= 3 else 1)


def pad_size(h, w, mult=64):
    return ((h - 1) // mult + 1) * mult, ((w - 1) // mult + 1) * mult


# ---------------------------------------------------------------------------
# ffmpeg I/O
# ---------------------------------------------------------------------------


def ffprobe(path):
    if not os.path.exists(path):
        sys.exit(f"input not found: {path}")
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,r_frame_rate,avg_frame_rate", "-of", "json", path],
            stderr=subprocess.PIPE,
        ).decode()
    except subprocess.CalledProcessError as e:
        sys.exit(f"ffprobe failed on {path}: {e.stderr.decode().strip()}")
    if '"streams"' not in out:
        sys.exit(f"no video stream in {path}")
    st = json.loads(out)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    rate = st.get("r_frame_rate", "0/0")
    if rate in ("0/0", "0"):
        rate = st.get("avg_frame_rate", "0/0")
    num, den = (int(x) for x in rate.split("/"))
    return w, h, (num / den if den else 0.0)


def ffprobe_frames(path, fps):
    """Total frame count (falls back to duration*fps for containers without it)."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=nb_frames:format=duration", "-of", "json", path]).decode()
        j = json.loads(out)
        n = j.get("streams", [{}])[0].get("nb_frames", "")
        if str(n).isdigit() and int(n) > 0:
            return int(n)
        dur = float(j.get("format", {}).get("duration", 0.0))
        return int(round(dur * fps)) if dur > 0 else None
    except Exception:
        return None


def ffprobe_bitrate(path):
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=bit_rate",
             "-of", "csv=p=0", path]).decode().strip()
        return float(out) if out and out != "N/A" else 0.0
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def has_audio(path):
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index", "-of", "csv=p=0", path]
        ).decode().strip()
        return bool(out)
    except (subprocess.CalledProcessError, IndexError):
        return False


class FrameReader(threading.Thread):
    """Decode with ffmpeg into a queue of HWC uint8 arrays."""

    def __init__(self, path, w, h, q):
        super().__init__(daemon=True)
        self.w, self.h, self.q = w, h, q
        self.nbytes = w * h * 3
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-map", "0:v:0",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE, bufsize=self.nbytes * 8,
        )

    def run(self):
        try:
            while True:
                buf = self.proc.stdout.read(self.nbytes)
                if not buf or len(buf) < self.nbytes:
                    break
                self.q.put(np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3).copy())
        finally:
            self.q.put(None)
            self.proc.stdout.close()
            self.proc.wait()


class FrameWriter(threading.Thread):
    """Encode chunks of HWC uint8 frames with ffmpeg, or dump PNGs."""

    def __init__(self, path, w, h, fps, q, pix_fmt="rgb24", venc_args=None,
                 png_dir=None, null=False):
        super().__init__(daemon=True)
        self.q, self.png_dir, self.written = q, png_dir, 0
        self.proc = None
        self.null = null                # measure the pipeline, encode nothing
        if png_dir:
            os.makedirs(png_dir, exist_ok=True)
            return
        if self.null:
            return
        v = venc_args or ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20"]
        # keep nv12 all the way: forcing yuv420p makes ffmpeg insert an
        # nv12->i420 swscale per frame (measured 965 -> 1121 fps at 720p)
        if pix_fmt == "nv12":
            pix = "nv12"
        else:
            pix = "yuv420p" if (w % 2 == 0 and h % 2 == 0) else "yuv444p"
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", pix_fmt, "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
               "-i", "-", "-an", *v, "-pix_fmt", pix, path]
        bpp = 1.5 if pix_fmt == "nv12" else 3
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     bufsize=int(w * h * bpp * 4))

    def run(self):
        import cv2
        while True:
            chunk = self.q.get()
            if chunk is None:
                break
            free = None
            if isinstance(chunk, tuple):  # (pinned array, cuda event, slot, pool)
                chunk, ev, slot, pool = chunk
                _t = time.perf_counter()
                if ev is not None:
                    ev.synchronize()
                _WST["wait"] += time.perf_counter() - _t
                free = (slot, pool)
            if self.null:
                self.written += len(chunk)
            elif self.png_dir:
                for fr in chunk:
                    cv2.imwrite(os.path.join(self.png_dir, f"{self.written:07d}.png"),
                                fr[:, :, ::-1])
                    self.written += 1
            else:
                _t = time.perf_counter()
                self.proc.stdin.write(memoryview(np.ascontiguousarray(chunk)))
                _WST["write"] += time.perf_counter() - _t
                self.written += len(chunk)
            if free is not None:
                free[1].put(free[0])
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()


# ---------------------------------------------------------------------------
# model / runner
# ---------------------------------------------------------------------------


def video_encoder_args(args, w=None, h=None):
    """ffmpeg -c:v ... for the chosen codec.

    Notes on the NVENC rate control: ``-cq`` is *not* x264's CRF.  ``-tune ll``
    disables B-frames and lookahead (fine for streaming, expensive for files);
    ``-cq 18`` on noisy 720p can easily hit 20+ Mbit/s.  For files use
    ``-tune hq`` + B-frames + ``-cq 24..26``, and AQ for dark/flat areas.
    """
    codec = args.codec
    if codec == "auto":
        # measured on a 4090: NVENC's HEVC beats H.264 by -9% bitrate at 1080p
        # and -25% at 1440p, but is 8% *worse* at 720p (CTU 64 is too coarse
        # for small frames).  Pick by resolution.
        codec = "h265" if (h or 0) > 1080 else "h264"
    if codec == "x264":
        return ["-c:v", "libx264", "-preset", args.x264_preset,
                "-crf", str(args.cq)]
    if codec == "x265":
        return ["-c:v", "libx265", "-preset", args.x264_preset,
                "-crf", str(args.cq), "-tag:v", "hvc1"]
    is_hevc = codec in ("h265", "hevc")
    enc = "hevc_nvenc" if is_hevc else "h264_nvenc"
    # h264 -> speed preset, h265 -> compression preset, unless overridden
    preset = args.nvenc_preset or ("p3" if is_hevc else "p1")
    tune = args.tune or ("hq" if is_hevc else "ull")
    bf = args.nvenc_bf if args.nvenc_bf is not None else (3 if is_hevc else 0)
    br = str(args.bitrate)
    if br not in ("0", "0k", "0K"):
        # predictable file size: VBR with a target + a cap
        mbps = _parse_rate(br)
        v = ["-c:v", enc, "-preset", preset, "-tune", tune,
             "-rc", "vbr", "-b:v", br, "-maxrate", f"{int(mbps*1.5)}M",
             "-bufsize", f"{int(mbps*3)}M", "-bf", str(bf)]
    else:
        # constqp = true constant quality (no rate-control overhead, ~2% faster),
        # long GOP + no scenecut = fewer expensive IDR frames
        v = ["-c:v", enc, "-preset", preset, "-tune", tune,
             "-rc", "constqp", "-qp", str(args.cq), "-bf", str(bf),
             "-g", "600", "-no-scenecut", "1", "-multipass", "0"]
    if tune == "hq":   # lookahead/AQ cost speed; only for the quality preset
        v += ["-rc-lookahead", "20", "-spatial_aq", "1", "-temporal_aq", "1"]
    if is_hevc:
        v += ["-tag:v", "hvc1"]
    return v


def _parse_rate(s):
    s = str(s).strip()
    mult = {"k": 1e-3, "K": 1e-3, "m": 1.0, "M": 1.0}.get(s[-1], None)
    return float(s[:-1]) * mult if mult else float(s) / 1e6


def build_net(args):
    net = load_flownet(args.model, DEV, fp16=args.fp16, mode=args.mode)
    if args.channels_last:
        net = net.to(memory_format=torch.channels_last)
        net._channels_last = True
    if args.compile != "off" and DEV.type == "cuda":
        net = torch.compile(net, mode=args.compile, dynamic=False)
    return net


class GraphRunner:
    """Capture one forward pass of ``net`` into a CUDA graph and replay it.

    ``with_feats`` also captures the ``f0/f1`` encoder-feature inputs, so the
    ``Head`` (which is computed outside the graph, once per chunk) is excluded.
    """

    def __init__(self, net, x_shape, scale_list, t_len, dtype, with_feats=True):
        self.net, self.scale_list = net, scale_list
        self.x = torch.zeros(x_shape, device=DEV, dtype=dtype)
        self.t = torch.zeros(t_len, device=DEV, dtype=dtype)
        self.f0 = self.f1 = None
        if with_feats:
            fshape = (x_shape[0], 4, x_shape[2], x_shape[3])
            self.f0 = torch.zeros(fshape, device=DEV, dtype=dtype)
            self.f1 = torch.zeros(fshape, device=DEV, dtype=dtype)
        def call():
            return net(self.x, self.t, scale_list=scale_list,
                       f0=self.f0, f1=self.f1)

        self._call = call
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):  # warmup on a side stream
            for _ in range(3):
                self.out = call()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = call()

    @torch.inference_mode()
    def __call__(self, x, t, f0=None, f1=None):
        self.x.copy_(x)
        self.t.copy_(t)
        if self.f0 is not None:
            self.f0.copy_(f0)
            self.f1.copy_(f1)
        self.graph.replay()
        return self.out


_WST = {"wait": 0.0, "write": 0.0}
_BT601 = None


def rgb_to_nv12(x: torch.Tensor) -> torch.Tensor:
    """[F,3,H,W] uint8/fp16 RGB (NCHW, GPU) -> [F, H*3//2, W] uint8 NV12.

    Works straight on NCHW: the previous version went through
    ``permute(0,2,3,1).reshape(-1,3)`` which forces a full NHWC transpose copy
    (44 MB per chunk at 720p) before the 3x3 colour matrix.
    """
    global _BT601
    nb, _, H, W = x.shape
    if _BT601 is None or _BT601.device != x.device:
        _BT601 = torch.tensor(                         # rows: Y,U,V  cols: R,G,B
            [[0.2568, -0.1482, 0.4392],
             [0.5041, -0.2910, -0.3678],
             [0.0979, 0.4392, -0.0714]], device=x.device, dtype=torch.float16)
    f = x.to(torch.float16)
    # yuv[c] = sum_r f[r] * M[r, c]
    r, g, b = f[:, 0], f[:, 1], f[:, 2]
    m = _BT601
    y = r * m[0, 0] + g * m[1, 0] + b * m[2, 0] + 16.0
    u = r * m[0, 1] + g * m[1, 1] + b * m[2, 1] + 128.0
    v = r * m[0, 2] + g * m[1, 2] + b * m[2, 2] + 128.0
    y = y.clamp_(0, 255).to(torch.uint8)
    # chroma in fp32: the 2x2 average of a 0..255 range loses ~2 dB in fp16
    uv = F.avg_pool2d(torch.stack((u, v), 1).float(), 2)       # [nb,2,H/2,W/2]
    uv = uv.clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).reshape(nb, H // 2, W)
    return torch.cat((y.reshape(nb, H, W), uv), dim=1)


def scale_list_of(scale):
    return [16.0 / scale, 8.0 / scale, 4.0 / scale, 2.0 / scale, 1.0 / scale]


@torch.inference_mode()
def chunk_forward(net, frames_u8_gpu, n_real, ts, scale, h, w, ph, pw, runner, K,
                  head_cache=True, nv12=False):
    """[K+1,H,W,3] uint8 on GPU -> [n_real*(n+1),H,W,3] uint8 on CPU."""
    dtype = net_dtype(net)
    # NHWC uint8 -> NCHW is a permute (free); its memory layout *is* channels_last
    x = frames_u8_gpu.to(dtype).div_(255.0).permute(0, 3, 1, 2)
    if pw != w or ph != h:
        x = F.pad(x, (0, pw - w, 0, ph - h))
    cl = getattr(net, "_channels_last", False) or getattr(
        getattr(net, "_orig_mod", net), "_channels_last", False)
    if cl:
        x = x.contiguous(memory_format=torch.channels_last)
    n = len(ts)
    # a CUDA graph / TRT engine has a fixed batch: always feed K pairs, then
    # keep only the frames that belong to real pairs
    fixed = runner is not None or getattr(net, "fixed_batch", False)
    n_batch = K if fixed else n_real
    i0 = x[:n_batch].repeat_interleave(n, 0)
    i1 = x[1:n_batch + 1].repeat_interleave(n, 0)
    # one batched Head over the K+1 distinct frames instead of 2K encodes
    feats = net.encode(x[:n_batch + 1]) if head_cache else None
    f0 = feats[:n_batch].repeat_interleave(n, 0) if head_cache else None
    f1 = feats[1:n_batch + 1].repeat_interleave(n, 0) if head_cache else None
    t = torch.tensor(ts, dtype=dtype, device=DEV).repeat(n_batch)
    xin = torch.cat((i0, i1), 1)
    if runner is not None:
        mid = runner(xin, t, f0, f1)
    else:
        mid = net(xin, t, scale_list_of(scale), f0=f0, f1=f1)
    mid = mid[: n_real * n]
    mid = mid.mul_(255.0).clamp_(0, 255).to(torch.uint8)[:, :, :h, :w]  # NCHW
    mid = mid.view(n_real, n, 3, h, w)
    left = frames_u8_gpu[:n_real].permute(0, 3, 1, 2).unsqueeze(1)      # NCHW
    rgb = torch.cat((left, mid), 1).reshape(-1, 3, h, w)                # GPU, uint8
    if not nv12:
        return rgb.permute(0, 2, 3, 1)
    return rgb_to_nv12(rgb)


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


def bench(args):
    m = args.multi
    ts = [(i + 1) / m for i in range(m - 1)]
    ph, pw = pad_size(args.height, args.width, args.pad_mult)
    net = build_net(args)
    dtype = net_dtype(net)
    print(f"bench {args.width}x{args.height} (padded {pw}x{ph}) multi={m} "
          f"mode={args.mode} fp16={int(args.fp16)} compile={args.compile} "
          f"device={DEV}")
    ks = ([args.batch_pairs] if args.batch_pairs
          else sorted({1, auto_batch_pairs(m, ph, pw)}))
    for K in ks:
        B = K * len(ts)
        runner = None
        if args.cudagraph and DEV.type == "cuda":
            runner = GraphRunner(net, (B, 6, ph, pw), scale_list_of(args.scale), B, dtype,
                                 with_feats=args.head_cache)
        frames = torch.randint(0, 255, (K + 1, args.height, args.width, 3),
                               dtype=torch.uint8, device=DEV)
        for _ in range(3):
            chunk_forward(net, frames, K, ts, args.scale, args.height, args.width,
                          ph, pw, runner, K, args.head_cache)
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            chunk_forward(net, frames, K, ts, args.scale, args.height, args.width,
                          ph, pw, runner, K, args.head_cache)
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.iters
        pairs = K / dt
        print(f"  batch_pairs={K:<3d} samples/fwd={B:<3d} : {dt*1e3:7.2f} ms/chunk  "
              f"{pairs:7.1f} pairs/s  {pairs*m:7.1f} out-fps  "
              f"({dt/K*1e3:6.2f} ms/pair)")


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------


def ensure_engine(args, batch, h, w, ph, pw):
    """--trt-engine auto: one engine per (resolution, batch), built on demand."""
    os.makedirs(ENGINES, exist_ok=True)
    from rife_fast.engine import engine_name
    name = os.path.join(str(ENGINES), engine_name(pw, ph, batch))
    if os.path.exists(name):
        return name
    if not torch.cuda.is_available():
        sys.exit("--trt-engine auto needs CUDA")
    from rife_fast.engine import build, export
    print(f"[trt] {name} not found, exporting + building "
          f"({pw}x{ph}, batch {batch}, ~2 min)")
    onnx = name.replace(".engine", ".onnx")
    export(h, w, batch, args.scale, onnx, args.mode, fp16=True)
    build(onnx, name, fp16=True)
    return name


def run_video(args):
    m = args.multi
    ts = [(i + 1) / m for i in range(m - 1)]
    w, h, fps = ffprobe(args.input)
    out_fps = args.fps or fps * m
    ph, pw = pad_size(h, w, args.pad_mult)
    print(f"input {w}x{h} @ {fps:.3f} fps -> {m}x, padded {pw}x{ph}, out {out_fps:.3f} fps")

    K = args.batch_pairs or auto_batch_pairs(m, ph, pw)
    if args.batch_pairs is None:
        print(f"auto batch-pairs = {K} (batch {K*len(ts)})")
    head_cache = args.head_cache
    runner = None
    if args.trt_engine == "auto":
        args.trt_engine = ensure_engine(args, K * len(ts), h, w, ph, pw)
    if args.trt_engine:
        from rife_fast.engine import TRTNet
        net = TRTNet(args.trt_engine, (K * len(ts), 6, ph, pw))
        head_cache = False  # the encoder lives inside the engine
        print(f"TensorRT engine: {args.trt_engine}")
    else:
        net = build_net(args)
        dtype = net_dtype(net)
        if args.cudagraph and DEV.type == "cuda":
            try:
                runner = GraphRunner(net, (K * len(ts), 6, ph, pw),
                                     scale_list_of(args.scale), K * len(ts),
                                     dtype, with_feats=head_cache)
                print("CUDA graph captured")
            except Exception as e:  # noqa: BLE001 - capture is only an optimisation
                print(f"CUDA graph capture failed ({type(e).__name__}); "
                      "continuing without it")
                runner = None

    out_path = args.output
    png_dir = None
    if args.png:
        png_dir = args.output or "vid_out"
        out_path = None
    elif not out_path:
        base, _ = os.path.splitext(args.input)
        out_path = f"{base}_{m}X_{int(round(out_fps))}fps.mp4"

    # NV12 through the pipe: no CPU swscale, half the D2H + pipe bytes
    use_nv12 = (not args.png and args.pix_fmt != "rgb24"
                and h % 2 == 0 and w % 2 == 0)
    rq, wq = queue.Queue(maxsize=8), queue.Queue(maxsize=4)
    reader = FrameReader(args.input, w, h, rq)
    writer = FrameWriter(out_path, w, h, out_fps, wq, png_dir=png_dir,
                         pix_fmt="nv12" if use_nv12 else "rgb24",
                         venc_args=None if (args.png or args.encoder == "null")
                         else video_encoder_args(args, w, h),
                         null=(args.encoder == "null"))
    if use_nv12:
        print("piping NV12 (GPU colour conversion, no CPU swscale)")
    reader.start()
    writer.start()

    # pinned staging + a copy stream so the D2H of chunk n overlaps chunk n+1
    n_out = K * (len(ts) + 1)
    frame_shape = (h + h // 2, w) if use_nv12 else (h, w, 3)
    pool = queue.Queue()
    bufs = []
    copy_stream = None
    if DEV.type == "cuda":
        copy_stream = torch.cuda.Stream()
        for i in range(6):
            bufs.append(torch.empty((n_out,) + frame_shape, dtype=torch.uint8,
                                    pin_memory=True))
            pool.put(i)
    pin_in = (torch.empty((K + 1, h, w, 3), dtype=torch.uint8, pin_memory=True)
              if DEV.type == "cuda" else None)

    stats = dict(read=0.0, pin=0.0, h2d=0.0, fwd=0.0, bufwait=0.0)
    want_stats = bool(os.environ.get("RIFE_STATS"))
    total_frames = ffprobe_frames(args.input, fps)
    total_pairs = (total_frames - 1) if total_frames else None
    from tqdm import tqdm
    pbar = tqdm(total=total_pairs, unit="pair", desc="interp",
                dynamic_ncols=True, smoothing=0.05,
                disable=None,   # auto: bar in a TTY, plain lines in a log
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    window, done, eof = [], 0, False
    t_start = pbar_t = time.perf_counter()

    def fill():
        nonlocal eof
        while not eof and len(window) < K + 1:
            f = rq.get()
            if f is None:
                eof = True
                break
            window.append(f)

    while True:
        _t0 = time.perf_counter()
        fill()
        if len(window) < 2:
            break
        _t1 = time.perf_counter()
        n_real = min(len(window) - 1, K)
        chunk = window[: n_real + 1]
        if n_real < K:  # pad the batch to the fixed graph shape
            chunk = chunk + [chunk[-1]] * (K - n_real)
        if pin_in is not None:
            # copy frame by frame: np.stack would add a second full-size memcpy
            for _i, _fr in enumerate(chunk):
                pin_in[_i].copy_(torch.from_numpy(_fr))
            _t2 = time.perf_counter()
            gpu = pin_in[: len(chunk)].to(DEV, non_blocking=True)
        else:
            gpu = torch.from_numpy(np.stack(chunk))
            _t2 = time.perf_counter()
        _t3 = time.perf_counter()
        _ev0 = torch.cuda.Event(True) if want_stats else None
        if _ev0 is not None:
            _ev0.record()
        out_gpu = chunk_forward(net, gpu, n_real, ts, args.scale, h, w, ph, pw,
                                runner, K, head_cache, use_nv12)
        _t4 = time.perf_counter()
        if _ev0 is not None:
            _ev1 = torch.cuda.Event(True)
            _ev1.record()
            _ev1.synchronize()
            stats["gpu_fwd"] = stats.get("gpu_fwd", 0.0) + _ev0.elapsed_time(_ev1) / 1e3
        if copy_stream is not None:
            slot = pool.get()  # back-pressure: waits for the writer to free one
            _t5 = time.perf_counter()
            if want_stats:
                stats["read"] += _t1 - _t0
                stats["pin"] += _t2 - _t1
                stats["h2d"] += _t3 - _t2
                stats["fwd"] += _t4 - _t3
                stats["bufwait"] += _t5 - _t4
            buf = bufs[slot]
            # the wait must be issued *outside* the stream context, otherwise
            # current_stream() is copy_stream itself and the wait is a no-op
            copy_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(copy_stream):
                buf[: out_gpu.shape[0]].copy_(out_gpu, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(copy_stream)
            wq.put((buf.numpy()[: out_gpu.shape[0]], ev, slot, pool))
        else:
            wq.put(out_gpu.cpu().numpy())
        done += n_real
        window = window[n_real:]
        pbar.update(n_real)
        now = time.perf_counter()
        if now - pbar_t > 0.25:
            pbar_t = now
            el = max(now - t_start, 1e-6)
            pbar.set_postfix_str(f"{done/el:6.1f} pair/s  "
                                 f"{done*m/el:6.0f} fps out")

    if window:  # last frame
        if copy_stream is not None:
            slot = pool.get()
            buf = bufs[slot]
            last = torch.from_numpy(window[0])
            if use_nv12:
                last = rgb_to_nv12(last.to(DEV).permute(2, 0, 1)[None])[0].cpu()
            buf[0].copy_(last)
            wq.put((buf.numpy()[:1], None, slot, pool))
        else:
            wq.put(window[:1])
    pbar.close()
    wq.put(None)
    writer.join()
    reader.join()
    el = time.perf_counter() - t_start
    print(f"\ndone: {done} pairs in {el:.1f}s ({done/el:.1f} pairs/s, "
          f"{done*m/el:.1f} out-fps)")
    if want_stats:
        nch = max(1, round(done / K))
        print(f"writer: wait {_WST['wait']/nch*1e3:6.2f} ms/chunk  "
              f"pipe {_WST['write']/nch*1e3:6.2f} ms/chunk")
        print("stage breakdown (ms/chunk):")
        for k, v in stats.items():
            print(f"  {k:8s} {v/nch*1e3:7.2f}  ({100*v/el:5.1f}% of wall)")
        print(f"  {'wall':8s} {el/nch*1e3:7.2f}")

    if out_path and os.path.exists(out_path):
        sz = os.path.getsize(out_path)
        dur = max(done / max(fps, 1e-6), 1e-6)
        out_br = sz * 8 / dur
        print(f"output : {sz/1e6:7.1f} MB  {out_br/1e6:5.2f} Mbps  "
              f"({out_br/(w*h*out_fps):.3f} bit/pixel)")
        src_br = ffprobe_bitrate(args.input)
        if src_br and out_br > 1.8 * src_br:
            print(f"  !! {out_br/src_br:.1f}x the source bitrate "
                  f"({src_br/1e6:.2f} Mbps). File will be ~{out_br/src_br:.1f}x "
                  f"the source size.\n     lower it with --cq "
                  f"{args.cq + 4} / --cq {args.cq + 8}, or cap it with "
                  f"--bitrate {max(2, round(src_br/1e6))}M")

    if out_path and args.audio and has_audio(args.input):
        tmp = out_path + ".mux.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", out_path, "-i", args.input,
             "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy", "-shortest", tmp],
            check=False,
        )
        if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            shutil.move(tmp, out_path)
            print("audio muxed")


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # -- 基本：日常只需要这几个 ----------------------------------------
    g = p.add_argument_group("基本参数")
    g.add_argument("-i", "--input", "--video", dest="input")
    g.add_argument("-o", "--output")
    g.add_argument("-m", "--multi", type=int, default=2,
                   help="插帧倍率：2 / 4 / 8（默认 2）")
    g.add_argument("--exp", type=int, default=None, help="倍率 = 2**exp（等价 -m）")
    g.add_argument("--codec", default="auto", choices=["auto", "h264", "h265", "x264", "x265"],
                   help="auto（默认）：≤1080p 用 h264，>1080p 用 h265（实测）")
    g.add_argument("--quality", "--cq", dest="cq", type=int, default=22,
                   help="画质档 18~30：越小越清晰、文件越大。默认 22（人眼不可闻）；"
                        "20=视觉无损，26=平衡，30=小体积")
    g.add_argument("--bitrate", default="0",
                   help="固定码率，如 4M；给了就不用 --quality")
    g.add_argument("--fps", type=float, default=None, help="输出帧率（默认 源fps×倍率）")
    g.add_argument("--png", action="store_true", help="输出 PNG 序列而不是视频")
    g.add_argument("--no-audio", dest="audio", action="store_false", help="不合并音轨")

    # -- 高级：一般不用动 ----------------------------------------------
    a = p.add_argument_group("高级参数（一般不用改）")
    a.add_argument("--trt-engine", default=None,
                   help="'auto' = 按分辨率/批次/权重自动找或建引擎（推荐）")
    a.add_argument("--batch-pairs", type=int, default=None,
                   help="每次前向的帧对数；默认按 分辨率×倍率 自动选")
    a.add_argument("--scale", type=float, default=1.0, help="内部工作分辨率倍数（4K 用 0.5）")
    a.add_argument("--mode", default="pyramid", choices=["ref", "assemble", "pyramid"],
                   help="pyramid=最快(默认) / assemble=与原实现逐位一致 / ref=原始实现")
    a.add_argument("--fp32", dest="fp16", action="store_false",
                   help="关闭 fp16（默认 CUDA 上自动开启）")
    a.add_argument("--no-cudagraph", dest="cudagraph", action="store_false",
                   help="关闭 CUDA Graph（默认开启）")
    a.add_argument("--compile", default="off",
                   choices=["off", "default", "reduce-overhead", "max-autotune"],
                   help="torch.compile（与 --cudagraph 二选一）")
    a.add_argument("--channels-last", action="store_true", help="NHWC（实测无收益）")
    a.add_argument("--encoder", default="auto", choices=["auto", "nvenc", "x264", "null"],
                   help="null = 只测流水线不编码")
    a.add_argument("--pix-fmt", default="auto", choices=["auto", "nv12", "rgb24"],
                   help="nv12 = GPU 做 RGB→YUV（默认）")
    a.add_argument("--nvenc-preset", default=None, help="默认 h264→p1, h265→p3")
    a.add_argument("--tune", default=None, choices=["hq", "ll", "ull", "lossless"],
                   help="默认 h264→ll, h265→hq")
    a.add_argument("--nvenc-bf", type=int, default=None, help="B 帧数，默认 h264→0, h265→3")
    a.add_argument("--x264-preset", default="medium")
    a.add_argument("--model", default=None, help="权重目录（默认 models/）")
    a.add_argument("--pad-mult", type=int, default=64)
    a.add_argument("--no-head-cache", dest="head_cache", action="store_false")
    a.add_argument("--bench", action="store_true", help="只测模型吞吐，不读写视频")
    a.add_argument("--iters", type=int, default=20)
    a.add_argument("--width", type=int, default=1280, help="--bench only")
    a.add_argument("--height", type=int, default=720, help="--bench only")
    args = p.parse_args()

    if args.exp is not None:
        args.multi = 2 ** args.exp
    if args.multi < 2:
        p.error("--multi must be >= 2")

    if DEV.type != "cuda":
        args.fp16 = False          # half precision on CPU is slower, not faster
        args.cudagraph = False
    torch.set_grad_enabled(False)
    if DEV.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    if args.compile != "off" and args.cudagraph:
        print("note: torch.compile(reduce-overhead) already uses CUDA graphs; "
              "disabling --cudagraph")
        args.cudagraph = False
    if args.bench:
        bench(args)
        return
    if not args.input:
        p.error("--input is required (or use --bench)")
    run_video(args)


if __name__ == "__main__":
    main()
