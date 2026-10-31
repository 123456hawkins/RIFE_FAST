#!/usr/bin/env python3
"""Fast RIFE interpolation CLI (720p / 1080p / 2K, RTX 30/40).

日常只需要：输入、输出文件夹、倍率，偶尔加 --codec / --quality。

    python -m src -i in.mp4 -m 2 -o out/               # out/in_2X.mp4
    python -m src -i a.mp4 b.mp4 -m 2 -o out/          # 多文件排队处理
    python -m src -i in.mp4 -m 2 --codec h265 --quality 28 -o out/
    python -m src -i in.mp4 -m 4 -o out/               # out/in_4X.mp4

默认就用 TensorRT（首次每个 分辨率×倍率 组合会自动构建两个引擎，约 2~4 分钟，
之后直接复用）。想对比 PyTorch 路径或引擎建不起来时加 `--no-trt`。
其余参数都在"高级参数"里，一般不用动。详见 README.md。
"""

from __future__ import annotations

import argparse
import fcntl
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

from src import ENGINES, ffmpeg_util
from src.net import load_flownet  # noqa: E402

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
        raise RuntimeError(f"input not found: {path}")
    try:
        out = subprocess.check_output(
            [ffmpeg_util.ffprobe(), "-v", "error", "-select_streams", "v:0",
             "-show_entries",
             "stream=width,height,r_frame_rate,avg_frame_rate", "-of", "json", path],
            stderr=subprocess.PIPE,
        ).decode()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffprobe failed on {path}: {e.stderr.decode().strip()}") from e
    if '"streams"' not in out:
        raise RuntimeError(f"no video stream in {path}")
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
            [ffmpeg_util.ffprobe(), "-v", "error", "-select_streams", "v:0",
             "-show_entries",
             "stream=nb_frames:format=duration", "-of", "json", path]).decode()
        j = json.loads(out)
        n = j.get("streams", [{}])[0].get("nb_frames", "")
        if str(n).isdigit() and int(n) > 0:
            return int(n)
        dur = float(j.get("format", {}).get("duration", 0.0))
        return int(round(dur * fps)) if dur > 0 else None
    except Exception:
        return None


def has_audio(path):
    try:
        out = subprocess.check_output(
            [ffmpeg_util.ffprobe(), "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", path]
        ).decode().strip()
        return bool(out)
    except (subprocess.CalledProcessError, IndexError):
        return False


class FrameReader(threading.Thread):
    """Decode with ffmpeg into a ring of buffers of ``[h*3//2, w]`` NV12 frames.

    Two modes:

    * with ``free_q``/``bufs`` (CUDA path): ffmpeg writes **straight into a
      pinned buffer** (``os.readv``), so a frame costs one kernel copy instead
      of three (``read()`` -> ``bytes`` -> ``numpy.copy()`` -> pinned).  That
      was ~4x the CPU of the whole rest of the pipeline, mostly page faults on
      the freshly mmap'd 3 MB objects.
    * without (CPU fallback): the plain ``read``+``copy`` path.

    NV12 (not rgb24) because the decoder hands us 4:2:0 anyway: the model input
    is converted on the GPU and the *source* frames of the output go to the
    encoder untouched (bit-exact, and half the PCIe traffic of rgb24).
    """

    def __init__(self, path, w, h, q, pad=None, free_q=None, bufs=None,
                 start_frame=None, end_frame=None, fps=None):
        super().__init__(daemon=True)
        self.w, self.h, self.q = w, h, q
        self.free_q, self.bufs = free_q, bufs
        self.nbytes = w * h * 3 // 2
        vf = [] if pad is None else ["-vf", f"pad={pad[0]}:{pad[1]}"]
        cmd = [ffmpeg_util.ffmpeg(), "-v", "error", "-nostdin"]
        if start_frame:
            # input-side seek is frame accurate for transcoding: ffmpeg seeks to
            # the preceding keyframe, decodes and drops until the target frame
            cmd += ["-ss", f"{start_frame / fps:.9f}"]
        cmd += ["-i", path, "-map", "0:v:0", *vf]
        if end_frame is not None:
            cmd += ["-frames:v", str(int(end_frame) - int(start_frame or 0))]
        # no duplication/dropping: each decoded frame is one output frame
        cmd += ["-vsync", "0", "-f", "rawvideo", "-pix_fmt", "nv12", "-"]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            bufsize=0 if bufs is not None else self.nbytes * 8,
        )

    def run(self):
        try:
            if self.bufs is None:
                while True:
                    buf = self.proc.stdout.read(self.nbytes)
                    if not buf or len(buf) < self.nbytes:
                        break
                    self.q.put(np.frombuffer(buf, np.uint8).reshape(
                        self.h + self.h // 2, self.w).copy())
                return
            fd = self.proc.stdout.fileno()
            n = self.nbytes
            while True:
                slot, ev = self.free_q.get()
                if slot is None:
                    break
                if ev is not None:
                    # never reuse a slot whose H2D may still be in flight; poll
                    # instead of synchronize(), which would busy-wait (an event
                    # that old is ready ~always, so the sleep is a safety net)
                    while not ev.query():
                        time.sleep(0.0005)
                mv = memoryview(self.bufs[slot].numpy()).cast("B")
                got = 0
                while got < n:
                    r = os.readv(fd, [mv[got:]])
                    if r == 0:
                        break
                    got += r
                if got < n:
                    self.free_q.put((slot, None))
                    break
                self.q.put(slot)
        finally:
            self.q.put(None)
            self.proc.stdout.close()
            self.proc.wait()


class FrameWriter(threading.Thread):
    """Encode chunks of HWC uint8 frames with ffmpeg."""

    def __init__(self, path, w, h, fps, q, pix_fmt="rgb24", venc_args=None,
                 null=False):
        super().__init__(daemon=True)
        self.q, self.written = q, 0
        self.proc = None
        self.null = null                # measure the pipeline, encode nothing
        if self.null:
            return
        v = venc_args or ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20"]
        # keep nv12 all the way: forcing yuv420p makes ffmpeg insert an
        # nv12->i420 swscale per frame (measured 965 -> 1121 fps at 720p)
        if pix_fmt == "nv12":
            pix = "nv12"
        else:
            pix = "yuv420p" if (w % 2 == 0 and h % 2 == 0) else "yuv444p"
        cmd = [ffmpeg_util.ffmpeg(), "-v", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", pix_fmt, "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
               "-i", "-", "-an", *v, "-pix_fmt", pix, path]
        bpp = 1.5 if pix_fmt == "nv12" else 3
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     bufsize=int(w * h * bpp * 4))

    def run(self):
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

    Quality is always controlled by ``--quality`` (CRF for x264/x265,
    constant-QP for NVENC), never by a fixed bitrate.  ``-cq`` is *not* x264's
    CRF, but constant-QP still gives a predictable quality/size trade-off and
    avoids the rate-control overhead of VBR (~2% faster).
    """
    codec = args.codec
    if getattr(args, "encoder", "auto") == "x264":
        codec = "x264"          # --encoder x264 is an explicit override
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


def mux_audio(input_path, out_path):
    """Copy the source audio track into ``out_path`` (no-op without audio)."""
    if not has_audio(input_path):
        return False
    tmp = out_path + ".mux.mp4"
    r = subprocess.run(
        [ffmpeg_util.ffmpeg(), "-v", "error", "-y", "-i", out_path,
         "-i", input_path, "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy",
         "-shortest", tmp],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        msg = r.stderr.decode(errors="replace").strip().splitlines()
        print(f"warning: audio mux failed ({msg[-1] if msg else 'ffmpeg error'})")
        return False
    shutil.move(tmp, out_path)
    print("audio muxed")
    return True


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
_SCRATCH: dict = {}


def _scratch(key, shape, dtype, zero=True):
    """One persistent buffer per role, re-allocated when the shape changes.

    Keeping the same tensor alive across chunks keeps its address stable (the
    forward pass reads/writes it every chunk) and avoids allocator churn; the
    key deliberately ignores the shape so a queue with mixed resolutions does
    not accumulate buffers ("only one shape at a time").
    """
    buf = _SCRATCH.get(key)
    if buf is None or tuple(buf.shape) != tuple(shape) or buf.dtype != dtype:
        buf = (torch.zeros if zero else torch.empty)(shape, device=DEV, dtype=dtype)
        _SCRATCH[key] = buf
    return buf


_T_CACHE: dict = {}


def _timesteps(ts, n_batch, dtype):
    """The [B] timestep tensor, built once per (ts, batch, dtype).

    ``torch.tensor(python_list, device='cuda')`` copies through pageable memory,
    which is a *synchronising* H2D: it stalls the compute stream and busy-waits
    for it (~0.3 ms measured, plus it kills the CPU/GPU overlap).  The
    timesteps never change for a given run, so build them once.
    """
    key = (tuple(ts), n_batch, dtype)
    t = _T_CACHE.get(key)
    if t is None:
        t = torch.tensor(list(ts) * n_batch, dtype=dtype, device=DEV)
        _T_CACHE[key] = t
    return t


_IDX: dict = {}


def _dst_index(K, m):
    from src.kernels import dst_index
    idx = _IDX.get((K, m))
    if idx is None:
        idx = dst_index(K, m, DEV)
        _IDX[(K, m)] = idx
    return idx


def warmup_kernels(h, w, ph, pw, K, m, dtype):
    """Compile + run the Triton kernels once, outside the timed loop."""
    if DEV.type != "cuda":
        return
    from src import kernels
    yuv = torch.zeros(K + 1, h + h // 2, w, dtype=torch.uint8, device=DEV)
    x = _scratch("x", (K + 1, 3, ph, pw), dtype)
    kernels.nv12_to_rgb(x, yuv, h, w)
    out = _scratch("out0", (K * m, h + h // 2, w), torch.uint8, zero=False)
    mid = torch.zeros(K * (m - 1), 3, x.shape[2], x.shape[3], device=DEV, dtype=dtype)
    kernels.rgb_to_nv12(out, mid, yuv, _dst_index(K, m), m - 1, h, w)
    torch.cuda.synchronize()


def scale_list_of(scale):
    return [16.0 / scale, 8.0 / scale, 4.0 / scale, 2.0 / scale, 1.0 / scale]


@torch.inference_mode()
def chunk_forward(net, yuv_gpu, n_real, ts, scale, h, w, ph, pw, runner, K,
                  head_cache=True, nv12=True, out_slot=0):
    """[K+1, h*3//2, w] uint8 NV12 on GPU -> [n_real*(n+1), h*3//2, w] NV12 on GPU.

    Source frames of the output are the decoder's own planes (no colour math at
    all, bit-exact); the interpolated frames get luma from the network output and
    chroma from the 2x2 box average of the same interpolated RGB.

    (Blending the two *source* chroma planes instead looked equivalent on paper
    -- RGB->YUV is linear -- but the interpolated RGB is motion compensated, not
    a blend, so luma and chroma edges disagree and the bitrate went up 44%.)
    """
    from src import kernels

    dtype = net_dtype(net)
    n = len(ts)
    x = _scratch("x", (K + 1, 3, ph, pw), dtype)
    kernels.nv12_to_rgb(x, yuv_gpu, h, w)

    cl = getattr(net, "_channels_last", False) or getattr(
        getattr(net, "_orig_mod", net), "_channels_last", False)
    if cl:
        x = x.contiguous(memory_format=torch.channels_last)

    # a CUDA graph / TRT engine has a fixed batch: always feed K pairs, then
    # keep only the frames that belong to real pairs
    fixed = runner is not None or getattr(net, "fixed_batch", False)
    n_batch = K if fixed else n_real
    takes_frames = getattr(net, "takes_frames", False)
    t = _timesteps(ts, n_batch, dtype)
    if takes_frames:
        # Head+Body TRT: hand over the K+1 distinct frames, the graph slices them
        mid = net(x[: K + 1], t)
    else:
        i0 = x[:n_batch].repeat_interleave(n, 0)
        i1 = x[1:n_batch + 1].repeat_interleave(n, 0)
        # one batched Head over the K+1 distinct frames instead of 2K encodes
        feats = net.encode(x[:n_batch + 1]) if head_cache else None
        f0 = feats[:n_batch].repeat_interleave(n, 0) if feats is not None else None
        f1 = feats[1:n_batch + 1].repeat_interleave(n, 0) if feats is not None else None
        xin = torch.cat((i0, i1), 1)
        if runner is not None:
            mid = runner(xin, t, f0, f1)
        else:
            mid = net(xin, t, scale_list_of(scale), f0=f0, f1=f1)

    m = len(ts) + 1
    out = _scratch(f"out{out_slot}", (K * m, h + h // 2, w), torch.uint8,
                   zero=False)
    kernels.rgb_to_nv12(out, mid, yuv_gpu, _dst_index(K, m), n, h, w)
    out = out[: n_real * m]
    if nv12:
        return out
    rgb = torch.empty(out.shape[0], h, w, 3, device=DEV, dtype=torch.uint8)
    tmp = _scratch("rgb", (out.shape[0], 3, ph, pw), dtype)
    kernels.nv12_to_rgb(tmp, out.contiguous(), h, w)
    rgb.copy_(tmp[:, :, :h, :w].mul_(255.0).to(torch.uint8).permute(0, 2, 3, 1))
    return rgb


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
        frames = torch.randint(0, 255, (K + 1, args.height + args.height // 2,
                                       args.width),
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


def engine_pair(spec):
    """``rife.engine`` -> (``rife.head.engine``, ``rife.body.engine``)."""
    base = os.path.splitext(spec)[0]
    return base + ".head.engine", base + ".body.engine"


def ensure_engine(args, K, m, h, w, ph, pw):
    """--trt-engine auto: Head+Body engines per (resolution, K, multi).

    Two GPU workers processing one video share ``engines/``.  They must not
    build the same plan twice: an advisory lock serialises the build, the
    second worker waits and then reuses the result.  The engine file itself is
    written atomically (``build`` uses write+rename).
    """
    os.makedirs(ENGINES, exist_ok=True)
    from src.engine import build, engine_files, export, gpu_tag
    head, body = engine_files(pw, ph, K, m, args.scale, args.mode, args.model)
    if os.path.exists(head) and os.path.exists(body):
        return head, body
    if not torch.cuda.is_available():
        sys.exit("--trt-engine auto needs CUDA")
    lock_path = os.path.splitext(body)[0] + ".lock"
    with open(lock_path, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"[trt] another process is building "
                  f"{os.path.basename(body)}; waiting...", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(head) and os.path.exists(body):
            return head, body
        print(f"[trt] {os.path.basename(body)} not found, exporting + building "
              f"({pw}x{ph}, K={K}, {m}x, {gpu_tag()}, ~2-5 min)", flush=True)
        head_onnx, body_onnx = export(h, w, K, m, args.scale, args.mode,
                                      fp16=True, pad_mult=args.pad_mult,
                                      model_dir=args.model)
        build(head_onnx, head, fp16=True)
        build(body_onnx, body, fp16=True)
    return head, body


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".m4v", ".avi", ".webm", ".flv", ".ts"}


def output_name(input_path, multi):
    """``a/b/clip.mp4`` -> ``clip_2X.mp4`` (the marker is the only suffix)."""
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return f"{stem}_{multi}X.mp4"


def plan_jobs(args):
    """Validate the queue and map every input to its output file."""
    inputs = [os.path.abspath(os.path.expanduser(p)) for p in args.input]
    for p in inputs:
        if not os.path.isfile(p):
            sys.exit(f"input not found: {p}")
    out_dir = os.path.abspath(os.path.expanduser(args.output or "."))
    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        sys.exit(f"-o/--output must be a directory: {out_dir}")
    if (not os.path.exists(out_dir)
            and os.path.splitext(out_dir)[1].lower() in VIDEO_EXTS):
        sys.exit(f"-o/--output is now an output directory, not a file: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    jobs = [(p, os.path.join(out_dir, output_name(p, args.multi))) for p in inputs]
    seen = {}
    input_set = set(inputs)
    for src, dst in jobs:
        if dst in seen:
            sys.exit(f"output collision: {src} and {seen[dst]} both map to {dst}")
        if dst in input_set:
            sys.exit(f"output would overwrite a queued input: {dst}")
        seen[dst] = src
    return jobs


class Runtime:
    """Model + graph/engine for the active (resolution, batch) shape.

    Queued files usually share a resolution.  Building the network, the TRT
    engine and capturing the CUDA graph is expensive, so keep them alive between
    files and only rebuild when the shape changes (one shape at a time keeps
    VRAM flat).
    """

    def __init__(self, args):
        self.args = args
        self._torch_net = None
        self._key = None
        self.net = None
        self.runner = None
        self.head_cache = args.head_cache

    def _torch(self):
        if self._torch_net is None:
            self._torch_net = build_net(self.args)
        return self._torch_net

    def get(self, K, m, h, w, ph, pw):
        batch = K * (m - 1)
        key = (batch, ph, pw)
        if key == self._key and self.net is not None:
            return self.net, self.runner, self.head_cache

        args = self.args
        spec, net, paths = args.trt_engine, None, None
        if spec:
            try:
                if spec == "auto":
                    spec = ensure_engine(args, K, m, h, w, ph, pw)
                paths = spec if isinstance(spec, tuple) else engine_pair(spec)
                from src.engine import TRTNet
                net = TRTNet(*paths, use_graph=args.cudagraph)
                if net.in_shape != (K + 1, 3, ph, pw):
                    raise RuntimeError(
                        f"{os.path.basename(paths[0])} takes {net.in_shape}, this "
                        f"input needs {(K + 1, 3, ph, pw)}")
            except Exception as e:  # noqa: BLE001 - TRT is an optimisation
                print(f"[trt] {type(e).__name__}: {e}")
                print("[trt] falling back to the PyTorch path (15~30% slower); "
                      "pass --no-trt to silence this")
                net = None
        if net is not None:
            head_cache = False  # the Head lives inside the graph
            runner = None
            print(f"TensorRT engine: {os.path.basename(paths[1])} "
                  f"(head {net.pairs + 1} frames -> {net.batch} samples)")
        else:
            net = self._torch()
            head_cache = args.head_cache
            runner = None
            if args.cudagraph and DEV.type == "cuda":
                try:
                    runner = GraphRunner(net, (batch, 6, ph, pw),
                                         scale_list_of(args.scale), batch,
                                         net_dtype(net), with_feats=head_cache)
                    print(f"CUDA graph captured ({pw}x{ph}, batch {batch})")
                except Exception as e:  # noqa: BLE001 - capture is only an optimisation
                    print(f"CUDA graph capture failed ({type(e).__name__}); "
                          "continuing without it")
                    runner = None
        self.net, self.runner, self.head_cache, self._key = (
            net, runner, head_cache, key)
        return net, runner, head_cache


def run_video(args, input_path, out_path, runtime, start_frame=None,
              end_frame=None, skip_first=False):
    """Interpolate one input file (or one frame range of it).

    ``start_frame``/``end_frame`` are used by the multi-GPU path to split a
    single video into contiguous pair ranges; ``end_frame`` is exclusive.
    ``skip_first`` drops the leading source frame, because the previous segment
    already emitted it (segments concatenate into exactly the same frame
    sequence as a single-process run).
    """
    m = args.multi
    ts = [(i + 1) / m for i in range(m - 1)]
    w, h, fps = ffprobe(input_path)
    if fps <= 0:
        raise RuntimeError(f"could not read the frame rate of {input_path}")
    gpu_tag = os.environ.get("RIFE_GPU_TAG")
    tag = f"[gpu {gpu_tag}] " if gpu_tag is not None else ""
    seg = (start_frame is not None or end_frame is not None)
    total = None
    if seg:
        total = ffprobe_frames(input_path, fps)
        start_frame = int(start_frame or 0)
        if end_frame is None:
            if total is None:
                raise RuntimeError(f"cannot segment {input_path}: unknown length")
            end_frame = total
        end_frame = int(end_frame)
        if not (0 <= start_frame < end_frame):
            raise RuntimeError(f"bad segment {start_frame}:{end_frame}")
    # 4:2:0 (NV12) cannot represent odd sizes; pad by one pixel if needed
    pad = None
    if (w & 1) or (h & 1):
        pw_, ph_ = w + (w & 1), h + (h & 1)
        print(f"{tag}note: {w}x{h} is odd, padding to {pw_}x{ph_} (4:2:0 needs "
              "even dimensions); the output will have the padded size")
        pad, (w, h) = (pw_, ph_), (pw_, ph_)
    out_fps = fps * m
    ph, pw = pad_size(h, w, args.pad_mult)
    span = (f" frames {start_frame}..{end_frame - 1} (of {total})"
            if seg else "")
    print(f"{tag}input {w}x{h} @ {fps:.3f} fps -> {m}X{span}, padded {pw}x{ph}, "
          f"out {out_fps:.3f} fps")

    K = args.batch_pairs or auto_batch_pairs(m, ph, pw)
    if args.batch_pairs is None:
        print(f"auto batch-pairs = {K} (batch {K*len(ts)})")
    net, runner, head_cache = runtime.get(K, m, h, w, ph, pw)
    warmup_kernels(h, w, ph, pw, K, m, net_dtype(net))

    # NV12 through the pipe: no CPU swscale, half the D2H + pipe bytes
    use_nv12 = args.pix_fmt != "rgb24" and h % 2 == 0 and w % 2 == 0
    if not use_nv12:
        print("note: opaque output requested -- converting NV12 back to rgb24 "
              "(slower, only useful for debugging)")
    if args.encoder != "null" and os.path.exists(out_path):
        os.remove(out_path)          # never report/mux a stale file from a failed run
    _WST["wait"] = _WST["write"] = 0.0
    # input ring: the decoder writes into these pinned buffers, the GPU copies
    # from them directly, so a frame costs one memcpy instead of three
    in_bufs = in_free = None
    if DEV.type == "cuda":
        in_bufs = [torch.empty((h + h // 2, w), dtype=torch.uint8, pin_memory=True)
                   for _ in range(24)]
        in_free = queue.Queue()
        for _i in range(len(in_bufs)):
            in_free.put((_i, None))
    yuv_gpu = _scratch("yuv", (K + 1, h + h // 2, w), torch.uint8, zero=False)
    rq, wq = queue.Queue(maxsize=8), queue.Queue(maxsize=4)
    reader = FrameReader(input_path, w, h, rq, pad, in_free, in_bufs,
                         start_frame=start_frame, end_frame=end_frame, fps=fps)
    writer = FrameWriter(out_path, w, h, out_fps, wq,
                         pix_fmt="nv12" if use_nv12 else "rgb24",
                         venc_args=(None if args.encoder == "null"
                                    else video_encoder_args(args, w, h)),
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

    stats = dict(read=0.0, h2d=0.0, fwd=0.0, bufwait=0.0)
    want_stats = bool(os.environ.get("RIFE_STATS"))
    total_frames = ((end_frame - start_frame) if seg
                    else ffprobe_frames(input_path, fps))
    total_pairs = max(total_frames - 1, 0) if total_frames else None
    from tqdm import tqdm
    want_progress = os.environ.get("RIFE_PROGRESS", "1") != "0"
    # workers inherit a terminal shared with their siblings; stacked tqdm bars
    # from separate processes fight over the cursor, so workers print lines
    use_bar = (want_progress and sys.stdout.isatty()
               and not os.environ.get("RIFE_WORKER"))
    pbar = tqdm(total=total_pairs, unit="pair", desc="interp",
                dynamic_ncols=True, smoothing=0.05,
                disable=not use_bar,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    window, done, eof = [], 0, False
    out_slot = 0
    skip_pending = bool(skip_first)
    t_start = pbar_t = last_log = time.perf_counter()

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
        if in_bufs is not None:
            for _i, _slot in enumerate(chunk):     # one H2D per frame, no CPU copy
                yuv_gpu[_i].copy_(in_bufs[_slot], non_blocking=True)
            gpu = yuv_gpu
            _t2 = time.perf_counter()
        else:
            gpu = torch.from_numpy(np.stack(chunk))
            _t2 = time.perf_counter()
        _ev0 = torch.cuda.Event(True) if want_stats else None
        if _ev0 is not None:
            _ev0.record()
        # ping-pong output buffers: chunk n's D2H reads slot n%2 while chunk
        # n+1 writes slot (n+1)%2, so no cross-stream wait is needed
        out_slot ^= 1
        out_gpu = chunk_forward(net, gpu, n_real, ts, args.scale, h, w, ph, pw,
                                runner, K, head_cache, use_nv12, out_slot)
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
                stats["h2d"] += _t2 - _t1
                stats["fwd"] += _t4 - _t2
                stats["bufwait"] += _t5 - _t4
            buf = bufs[slot]
            # the wait must be issued *outside* the stream context, otherwise
            # current_stream() is copy_stream itself and the wait is a no-op
            copy_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(copy_stream):
                buf[: out_gpu.shape[0]].copy_(out_gpu, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(copy_stream)
            first = 1 if skip_pending else 0
            skip_pending = False
            if first >= out_gpu.shape[0]:       # nothing to emit from this chunk
                pool.put(slot)
            else:
                wq.put((buf.numpy()[first: out_gpu.shape[0]], ev, slot, pool))
        else:
            first = 1 if skip_pending else 0
            skip_pending = False
            wq.put(out_gpu.cpu().numpy()[first:])
        done += n_real
        if in_free is not None:                    # hand the used slots back
            ev = torch.cuda.Event()
            ev.record()
            for _s in window[:n_real]:
                in_free.put((_s, ev))
        window = window[n_real:]
        pbar.update(n_real)
        now = time.perf_counter()
        if now - pbar_t > 0.25:
            pbar_t = now
            el = max(now - t_start, 1e-6)
            if use_bar:
                pbar.set_postfix_str(f"{done/el:6.1f} pair/s  "
                                     f"{done*m/el:6.0f} fps out")
            elif want_progress and now - last_log >= 5.0:
                last_log = now
                tot = f"/{total_pairs}" if total_pairs else ""
                print(f"{tag}{done}{tot} pairs  {done/el:6.1f} pair/s  "
                      f"{done*m/el:6.0f} fps out", flush=True)

    if window:  # last frame
        if copy_stream is not None:
            slot = pool.get()
            buf = bufs[slot]
            last = (in_bufs[window[0]] if in_bufs is not None
                    else torch.from_numpy(window[0]))   # already NV12
            if use_nv12:
                buf[0].copy_(last)
            else:
                from src import kernels
                tmp = torch.zeros(1, 3, ph, pw, device=DEV, dtype=net_dtype(net))
                kernels.nv12_to_rgb(tmp, last.to(DEV)[None], h, w)
                buf[0].copy_(tmp[0, :, :h, :w].mul_(255.0).to(torch.uint8)
                             .permute(1, 2, 0).cpu())
            first = 1 if skip_pending else 0
            skip_pending = False
            if first < 1:
                wq.put((buf.numpy()[:1], None, slot, pool))
            else:                              # first source frame already emitted
                pool.put(slot)
        else:
            wq.put(window[:1 if not skip_pending else 0])
    pbar.close()
    wq.put(None)
    writer.join()
    reader.join()
    el = time.perf_counter() - t_start
    # A truncated/seek-past-EOF input would otherwise produce a short part and
    # still look successful to the multi-GPU scheduler.  For an explicit frame
    # range the expected pair count is exact, so fail the task and let the
    # scheduler retry/report it.
    expected_pairs = total_frames - 1 if total_frames else None
    if seg and expected_pairs is not None and done != expected_pairs:
        raise RuntimeError(
            f"{input_path}: segment {start_frame}..{end_frame - 1} produced "
            f"{done} pairs, expected {expected_pairs} (truncated/corrupt input?)")
    print(f"\n{tag}done: {done} pairs in {el:.1f}s ({done/el:.1f} pairs/s, "
          f"{done*m/el:.1f} out-fps)")
    if want_stats:
        nch = max(1, round(done / K))
        print(f"writer: wait {_WST['wait']/nch*1e3:6.2f} ms/chunk  "
              f"pipe {_WST['write']/nch*1e3:6.2f} ms/chunk")
        print("stage breakdown (ms/chunk):")
        for k, v in stats.items():
            print(f"  {k:8s} {v/nch*1e3:7.2f}  ({100*v/el:5.1f}% of wall)")
        print(f"  {'wall':8s} {el/nch*1e3:7.2f}")

    if args.encoder != "null" and os.path.exists(out_path):
        sz = os.path.getsize(out_path)
        # each pair contributes m output frames; the first source frame is the
        # +1, unless a multi-GPU segment already emitted it
        n_frames = done * m + (0 if skip_first else 1)
        dur = max(n_frames / max(out_fps, 1e-6), 1e-6)
        out_br = sz * 8 / dur
        print(f"{tag}output : {out_path}  {sz/1e6:7.1f} MB  {out_br/1e6:5.2f} Mbps  "
              f"({out_br/(w*h*out_fps):.3f} bit/pixel)")

    if args.encoder != "null" and args.audio:
        mux_audio(input_path, out_path)


def run_queue(args):
    """Process every input file in order, reusing the model/engine between them."""
    jobs = plan_jobs(args)
    runtime = Runtime(args)
    total = len(jobs)
    failed = []
    t0 = time.perf_counter()
    for i, (src, dst) in enumerate(jobs, 1):
        print(f"\n=== [{i}/{total}] {src} -> {dst}")
        try:
            run_video(args, src, dst, runtime)
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the queue
            failed.append((src, e))
            print(f"!! {src}: {e}", file=sys.stderr)

    elapsed = time.perf_counter() - t0
    print(f"\nqueue: {total - len(failed)}/{total} files done in {elapsed:.1f}s")
    if failed:
        print("failed: " + ", ".join(os.path.basename(p) for p, _ in failed),
              file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # -- 基本：日常只需要这几个 ----------------------------------------
    g = p.add_argument_group("基本参数")
    g.add_argument("-i", "--input", "--video", dest="input", nargs="+",
                   action="extend", metavar="VIDEO",
                   help="输入视频；可给多个，按给出顺序排队处理")
    g.add_argument("-o", "--output", "--out-dir", dest="output", default=".",
                   metavar="DIR",
                   help="输出文件夹（默认当前目录）；文件名自动加 _2X 等标志")
    g.add_argument("-m", "--multi", type=int, default=2,
                   help="插帧倍率：2 / 4 / 8（默认 2）")
    g.add_argument("--exp", type=int, default=None, help="倍率 = 2**exp（等价 -m）")
    g.add_argument("--codec", default="auto", choices=["auto", "h264", "h265", "x264", "x265"],
                   help="auto（默认）：≤1080p 用 h264，>1080p 用 h265（实测）")
    g.add_argument("--quality", "--cq", dest="cq", type=int, default=22,
                   help="画质档 18~30：越小越清晰、文件越大。默认 22（人眼不可闻）；"
                        "20=视觉无损，26=平衡，30=小体积")
    g.add_argument("--no-audio", dest="audio", action="store_false", help="不合并音轨")

    # -- 高级：一般不用动 ----------------------------------------------
    a = p.add_argument_group("高级参数（一般不用改）")
    a.add_argument("--trt-engine", default="auto", metavar="auto|BASE",
                   help="默认 auto：按 分辨率/K/倍率/权重 自动找或建引擎"
                        "（首次每个组合约 2~4 分钟）；也可给一个基准路径")
    a.add_argument("--no-trt", dest="trt_engine", action="store_const", const=None,
                   help="禁用 TensorRT，走 PyTorch（慢 15~25%%）")
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
    a.add_argument("--gpus", nargs="+", default=None, metavar="GPU",
                   help="多卡：0,1 或 all（每卡一个子进程，共享任务队列，"
                        "完成即取下一个任务）")
    a.add_argument("--gpu-strategy", default="auto",
                   choices=["auto", "files", "segments"],
                   help="auto=按视频数/时长自动选（默认）；files=整文件任务；"
                        "segments=视频都按帧分段后任务队列分发")
    a.add_argument("--engine-per-gpu", action="store_true",
                   help="每张卡单独构建 TRT 引擎（首次多花约一倍构建时间，"
                        "避免跨卡 plan 的 TRT 警告与潜在性能损失）")
    a.add_argument("--bench", action="store_true", help="只测模型吞吐，不读写视频")
    a.add_argument("--iters", type=int, default=20)
    a.add_argument("--width", type=int, default=1280, help="--bench only")
    a.add_argument("--height", type=int, default=720, help="--bench only")
    args = p.parse_args()

    if args.exp is not None:
        args.multi = 2 ** args.exp
    if args.multi < 2:
        p.error("--multi must be >= 2")
    if args.bench and args.gpus:
        print("note: --bench measures device 0 only; ignoring --gpus")
        args.gpus = None
    if args.engine_per_gpu:
        # read by engine.gpu_tag() when the engine file name is built; workers
        # inherit it through the environment
        os.environ["RIFE_ENGINE_PER_GPU"] = "1"

    # Every heavy step runs on the GPU; the only CPU work is small memcpys and
    # tensor bookkeeping.  PyTorch's intra-op pool would otherwise wake ~30
    # threads for those copies and keep them busy for the whole run
    # (measured: 9.5 cores vs 0.8 cores, identical throughput).
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass                        # already started; harmless

    if DEV.type != "cuda":
        args.fp16 = False          # half precision on CPU is slower, not faster
        args.cudagraph = False
    torch.set_grad_enabled(False)
    if DEV.type == "cuda" and not args.gpus:
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    elif DEV.type == "cuda":
        # parent stays off CUDA: each worker gets its own context/engine/graph
        print(f"multi-GPU: {args.gpus}  torch {torch.__version__}")

    if args.compile != "off" and args.cudagraph:
        print("note: torch.compile(reduce-overhead) already uses CUDA graphs; "
              "disabling --cudagraph")
        args.cudagraph = False
    if args.bench:
        bench(args)
        return
    if not args.input:
        p.error("--input is required (or use --bench)")
    try:
        ffmpeg_util.check()
    except RuntimeError as e:
        sys.exit(f"error: {e}")
    if args.gpus:
        from src.multigpu import run_multi_gpu
        if not run_multi_gpu(args):
            sys.exit(1)
        return
    run_queue(args)


if __name__ == "__main__":
    main()
