"""Multi-GPU orchestration for the RIFE pipeline.

The parent never touches CUDA.  Each worker is a fresh
``python -m src._worker`` with ``CUDA_VISIBLE_DEVICES`` pinned to one card, so
TensorRT engines, CUDA graphs and the Triton JIT cache stay per device.

Scheduling is a **shared file-backed work queue** (``src/taskqueue.py``), not a
static assignment: every worker atomically claims the next task, so when one
card finishes early it immediately takes another file/segment.  Two task kinds:

``files``     one task per input video (whole-file mode for batches).  A worker
              writes the final output directly (including audio), so there is
              no temporary file/concat overhead.
``segments``  each video is split into contiguous pair ranges; one task per
              range.  Workers claim ranges independently, then the parent
              concatenates the encoded parts losslessly and muxes audio.

``auto`` uses ``files`` when there are at least as many videos as GPUs, and
``segments`` otherwise.  In segment mode the target is ``2 * #GPUs`` tasks
(capped by frames), which gives the queue spare tasks for stealing and keeps
both cards busy even when one is slower.

Segment boundaries are pair ranges.  Segment *k* decodes frames ``[p0, p1]``
(one extra frame at the end so the last pair is real), produces
``(p1-p0) * multi + 1`` output frames, and every segment after the first drops
its leading source frame because the previous segment already emitted it.
Concatenation therefore reproduces exactly ``(N-1)*multi + 1`` frames, the same
sequence a single-process run produces.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import torch

from src import ROOT, ffmpeg_util, taskqueue

# ---------------------------------------------------------------------------
# GPU / worker plumbing
# ---------------------------------------------------------------------------


def _visible_tokens() -> list[str]:
    """CUDA_VISIBLE_DEVICES entries as tokens (indices or UUIDs)."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env and env.strip() not in ("", "-1"):
        toks = [t.strip() for t in env.split(",") if t.strip()]
        if toks:
            return toks
    return [str(i) for i in range(torch.cuda.device_count())]


def parse_gpus(spec) -> list[int] | None:
    """``None`` (not requested), ``all``, ``0,1``, ``0 1`` -> logical indices."""
    if not spec:
        return None
    toks = []
    for item in spec:
        toks += [t for t in str(item).replace(" ", ",").split(",") if t]
    if len(toks) == 1 and toks[0].lower() == "all":
        return list(range(torch.cuda.device_count()))
    out = []
    for t in toks:
        if not t.isdigit():
            raise SystemExit(f"--gpus: '{t}' is not a device index (use 0,1 or all)")
        i = int(t)
        if i not in out:
            out.append(i)
    return out


def resolve_gpus(spec) -> list[int]:
    idx = parse_gpus(spec)
    if idx is None:
        idx = [0]
    if not idx:
        raise SystemExit("--gpus needs at least one device index")
    n = torch.cuda.device_count()
    if not n:
        raise SystemExit("--gpus needs CUDA, but no GPU is visible")
    bad = [i for i in idx if i < 0 or i >= n]
    if bad:
        raise SystemExit(f"--gpus {bad}: only {n} GPU(s) visible "
                         f"({os.environ.get('CUDA_VISIBLE_DEVICES', 'all')})")
    return idx


def _launch(payload: dict, token: str, gpu_idx: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = token
    env["RIFE_WORKER"] = "1"
    env["RIFE_GPU_TAG"] = str(gpu_idx)
    env["RIFE_WORKER_JSON"] = json.dumps(payload)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, "-m", "src._worker"]
    print(f"[multi] gpu {gpu_idx} (CUDA_VISIBLE_DEVICES={token})", flush=True)
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env)


def _wait(procs: list[subprocess.Popen]) -> bool:
    try:
        rcs = [p.wait() for p in procs]
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        raise
    return all(rc == 0 for rc in rcs)


def _args_for(args, **overrides) -> dict:
    d = vars(args).copy()
    d.update(overrides)
    # workers get explicit input/output paths in their task list; the full queue
    # would only bloat RIFE_WORKER_JSON (and can be thousands of paths)
    d.pop("input", None)
    d.pop("output", None)
    return d


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


def segment_bounds(n_frames: int, parts: int) -> list[tuple[int, int]]:
    """Pair ranges ``[(p0, p1), ...]`` covering ``n_frames - 1`` pairs."""
    pairs = n_frames - 1
    parts = max(1, min(parts, pairs))
    cuts = [round(i * pairs / parts) for i in range(parts + 1)]
    cuts[0], cuts[-1] = 0, pairs
    return [(cuts[i], cuts[i + 1]) for i in range(parts)]


def _video_info(src: str) -> dict:
    """Probe once per video; a corrupt file is excluded by the caller."""
    from src.cli import ffprobe, ffprobe_frames
    w, h, fps = ffprobe(src)
    n = ffprobe_frames(src, fps)
    dur = n / fps if (n and fps > 0) else float(os.path.getsize(src))
    return {"input": src, "w": w, "h": h, "fps": fps, "n": n, "dur": dur}


def _allocate_parts(infos: list[dict], target: int) -> list[int]:
    """Split ``target`` tasks over videos, proportional to duration.

    Every video gets one task; extra tasks go greedily to the video with the
    largest remaining duration-per-task.  A video can never get more parts than
    it has frame pairs.
    """
    n = len(infos)
    parts = [1] * n
    max_parts = []
    for info in infos:
        frames = info["n"]
        max_parts.append(max(1, frames - 1) if (frames and frames > 2) else 1)
    target = min(target, sum(max_parts))
    while sum(parts) < target:
        cand = [i for i in range(n) if parts[i] < max_parts[i]]
        if not cand:
            break
        i = max(cand, key=lambda k: infos[k]["dur"] / parts[k])
        parts[i] += 1
    return parts


def _balanced(infos: list[dict], tol: float = 1.25) -> bool:
    durs = [max(i["dur"], 1e-3) for i in infos]
    return max(durs) <= min(durs) * tol


def _auto_mode(infos: list[dict], gpu_ids: list[int]) -> str:
    """Pick files vs segments from the *estimated* makespan, not file count.

    ``files`` has no temporary parts/concat overhead but, with one task per
    card, a 2-hour + 1-minute pair leaves a card idle for 2 hours.  When the
    longest video dominates the per-GPU share, splitting it (segments) wins
    even though there are enough files to fill the cards.
    """
    n, G = len(infos), len(gpu_ids)
    if G <= 1:
        return "files"
    if n == 1:
        return "segments"
    if any(i["n"] is None for i in infos):
        # cannot estimate seconds reliably; don't split blind
        return "segments" if n < G else "files"
    durs = [max(i["dur"], 1e-3) for i in infos]
    total, longest = sum(durs), max(durs)
    per_gpu = total / G
    if n < G:
        return "segments"
    # N not a multiple of G: files mode necessarily gives one card ceil(N/G)
    # videos and another fewer (e.g. 3 videos on 2 cards -> 2 vs 1).  Splitting
    # into a multiple of G tasks is the only way to balance, and long+short
    # mixes benefit too.
    if n > G and n % G and total >= 20:
        return "segments"
    if (total >= 10 and longest >= 20 and longest > per_gpu * 1.35
            and longest - per_gpu > 3):
        return "segments"       # split the long tail; _allocate_parts keeps
    return "files"               # short videos whole, no parts overhead


def _build_plan(args, jobs, infos, gpu_ids, mode: str, workdir: str) -> dict:
    """Create the task/video manifest (no files are touched except part names)."""
    G = len(gpu_ids)
    m = args.multi
    parts_dir = os.path.join(workdir, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    if mode == "files":
        n_parts = [1] * len(jobs)
    else:
        # more tasks than workers gives the queue spare work to steal; the
        # extra split is chosen so task durations stay similar.
        durs = [max(i["dur"], 1e-3) for i in infos]
        total, longest = sum(durs), max(durs)
        per_gpu = total / G
        dominant = longest >= 20 and longest > per_gpu * 1.35
        if len(jobs) == 1:
            target = G if infos[0]["dur"] < 120 else 2 * G
        elif len(jobs) > G and len(jobs) % G and total >= 20:
            if _balanced(infos) and len(jobs) <= 2 * G:
                target = len(jobs) * ((len(jobs) + G - 1) // G)
            elif dominant:
                target = len(jobs) + 1
            else:
                target = max(2 * G, len(jobs) + G)
        elif len(jobs) < G:
            target = 2 * G
        elif dominant:
            target = len(jobs) + 1
        else:
            target = max(2 * G, len(jobs) + G)
        n_parts = _allocate_parts(infos, target)

    tasks, videos = [], []
    for vi, (info, (src, dst)) in enumerate(zip(infos, jobs)):
        n = info["n"]
        k = n_parts[vi]
        if n is not None and n >= 3:
            k = min(k, n - 1)
        else:
            k = 1
        merge = (mode == "segments" and args.encoder != "null")
        video = {
            "input": src,
            "output": dst,
            "n": n,
            "fps": info["fps"],
            "merge": merge,
            "expected": ((n - 1) * m + 1) if n else None,
            "parts": [],
            "task_ids": [],
        }
        if k <= 1:
            out = dst if mode == "files" else os.path.join(
                parts_dir, f"v{vi:02d}_p000.mp4")
            task = {"input": src, "output": out,
                    "start_frame": None, "end_frame": None,
                    "skip_first": False, "video": vi, "part": 0}
            tasks.append(task)
            video["task_ids"].append(len(tasks) - 1)
            video["parts"].append(out)
        else:
            for pi, (p0, p1) in enumerate(segment_bounds(n, k)):
                out = os.path.join(parts_dir, f"v{vi:02d}_p{pi:03d}.mp4")
                task = {"input": src, "output": out,
                        "start_frame": p0, "end_frame": p1 + 1,
                        "skip_first": p0 > 0, "video": vi, "part": pi}
                tasks.append(task)
                video["task_ids"].append(len(tasks) - 1)
                video["parts"].append(out)
        videos.append(video)
    return {"tasks": tasks, "videos": videos, "mode": mode}


def _write_manifest(workdir: str, manifest: dict) -> str:
    path = os.path.join(workdir, "manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f)
    os.replace(tmp, path)
    taskqueue.init_counter(workdir)
    return path


def _concat(parts: list[str], out_path: str) -> None:
    """Losslessly join encoded parts (same codec/extradata)."""
    if os.path.exists(out_path):
        os.remove(out_path)
    fd, list_path = tempfile.mkstemp(suffix=".txt", prefix=".rife_concat_",
                                     dir=os.path.dirname(out_path) or ".")
    try:
        with os.fdopen(fd, "w") as f:
            for p in parts:
                q = p.replace("'", "'\\''")
                f.write(f"file '{q}'\n")
        r = subprocess.run(
            [ffmpeg_util.ffmpeg(), "-v", "error", "-y", "-f", "concat",
             "-safe", "0", "-i", list_path, "-c", "copy", "-an", out_path],
            capture_output=True, text=True)
        if r.returncode != 0:
            msg = (r.stderr or "").strip().splitlines()
            raise RuntimeError(f"concat failed: {msg[-1] if msg else 'ffmpeg error'}")
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)


def _verify_frames(path: str, expected: int | None, strict: bool = False) -> None:
    if not expected:
        return
    from src.cli import ffprobe, ffprobe_frames
    try:
        _w, _h, fps = ffprobe(path)
        n = ffprobe_frames(path, fps)
    except Exception:                                      # noqa: BLE001
        return
    if n is not None and n != expected:
        msg = (f"{os.path.basename(path)} has {n} frames, expected {expected}")
        if strict:                       # segment parts have an exact count
            raise RuntimeError(msg)
        print(f"[multi] warning: {msg} (container metadata may be approximate)",
              flush=True)


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


def _run_workers(args, workdir: str, manifest_path: str,
                 gpu_ids: list[int]):
    """Launch one persistent worker per GPU; all share the same queue."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    tasks = manifest["tasks"]
    status_dir = os.path.join(workdir, "status")
    os.makedirs(status_dir, exist_ok=True)
    audio = args.audio if manifest.get("mode") == "files" else False
    payload = {
        "args": _args_for(args, audio=audio),
        "mode": "queue",
        "workdir": workdir,
        "manifest": manifest_path,
        "status_dir": status_dir,
    }
    toks = _visible_tokens()
    # don't pay a process/engine startup for a card that cannot get a task
    n_workers = min(len(gpu_ids), len(tasks))
    procs = [_launch(payload, toks[g], g) for g in gpu_ids[:n_workers]]
    ok = _wait(procs)
    statuses = [taskqueue.read_status(status_dir, i) for i in range(len(tasks))]
    return statuses, ok


def _run_with_retry(args, workdir: str, manifest_path: str,
                    gpu_ids: list[int]) -> list[dict | None]:
    with open(manifest_path) as f:
        manifest = json.load(f)
    tasks = manifest["tasks"]
    statuses, _ = _run_workers(args, workdir, manifest_path, gpu_ids)
    failed = [i for i, s in enumerate(statuses) if not s or not s.get("ok")]
    if not failed:
        return statuses

    print(f"[multi] {len(failed)} task(s) failed/missing; retrying once", flush=True)
    retry_dir = os.path.join(workdir, "retry")
    os.makedirs(retry_dir, exist_ok=True)
    retry_manifest = {
        "tasks": [tasks[i] for i in failed],
        "videos": [],
        "mode": manifest.get("mode"),
    }
    retry_path = _write_manifest(retry_dir, retry_manifest)
    retry_statuses, _ = _run_workers(args, retry_dir, retry_path, gpu_ids)
    for local, orig in enumerate(failed):
        statuses[orig] = retry_statuses[local]
    still = [i for i, s in enumerate(statuses) if not s or not s.get("ok")]
    if still:
        print(f"[multi] {len(still)} task(s) still failed after retry",
              file=sys.stderr, flush=True)
    return statuses


def _merge(args, manifest: dict, statuses: list) -> bool:
    """Concatenate segment parts / verify whole-file outputs; mux audio."""
    from src.cli import has_audio, mux_audio

    if args.encoder == "null":
        return all(s and s.get("ok") for s in statuses)

    all_ok = True
    for vi, video in enumerate(manifest["videos"]):
        ids = video["task_ids"]
        if not all(statuses[i] and statuses[i].get("ok") for i in ids):
            print(f"[multi] video {vi} ({os.path.basename(video['input'])}) "
                  f"has failed tasks; not merging", file=sys.stderr, flush=True)
            all_ok = False
            continue
        if not video["merge"]:
            # whole-file worker: it already wrote+muxed the final output
            out = video["output"]
            if not os.path.exists(out) or os.path.getsize(out) == 0:
                print(f"[multi] missing output: {out}", file=sys.stderr, flush=True)
                all_ok = False
                continue
            _verify_frames(out, video["expected"])
            continue

        parts = video["parts"]
        missing = [p for p in parts
                   if not os.path.exists(p) or os.path.getsize(p) == 0]
        if missing:
            print(f"[multi] video {vi} missing {len(missing)} part(s); "
                  f"not merging", file=sys.stderr, flush=True)
            all_ok = False
            continue
        try:
            if len(parts) == 1:
                if os.path.exists(video["output"]):
                    os.remove(video["output"])
                shutil.move(parts[0], video["output"])
            else:
                _concat(parts, video["output"])
            if args.audio and has_audio(video["input"]):
                mux_audio(video["input"], video["output"])
            _verify_frames(video["output"], video["expected"], strict=True)
        except Exception as e:                              # noqa: BLE001
            print(f"[multi] video {vi} merge failed: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            all_ok = False
    return all_ok


def run_multi_gpu(args) -> bool:
    """Entry point called by ``cli.main`` when ``--gpus`` is given."""
    from src.cli import plan_jobs

    gpu_ids = resolve_gpus(args.gpus)
    jobs = plan_jobs(args)

    # Probe every input once.  A corrupt input must not kill the rest of the
    # batch: it is reported and excluded, and the exit status becomes non-zero.
    good_jobs, infos, bad = [], [], []
    for src, dst in jobs:
        try:
            infos.append(_video_info(src))
            good_jobs.append((src, dst))
        except Exception as e:                              # noqa: BLE001
            bad.append(src)
            print(f"[multi] !! cannot read {src}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
    if not good_jobs:
        return False

    # Longest first: a worker that finishes early steals the next task, and
    # starting the big job immediately avoids "two shorts, then one long".
    order = sorted(range(len(good_jobs)), key=lambda i: -infos[i]["dur"])
    good_jobs = [good_jobs[i] for i in order]
    infos = [infos[i] for i in order]

    mode = args.gpu_strategy
    if mode == "auto":
        mode = _auto_mode(infos, gpu_ids)
    if mode == "files" and len(good_jobs) < len(gpu_ids):
        print(f"[multi] note: files mode with {len(good_jobs)} video(s) on "
              f"{len(gpu_ids)} GPUs will leave some GPUs idle", flush=True)
    print(f"[multi] {len(gpu_ids)} GPU(s): {gpu_ids}  mode={mode}", flush=True)

    out_dir = os.path.dirname(good_jobs[0][1]) or "."
    workdir = tempfile.mkdtemp(prefix=".rife_queue_", dir=out_dir)
    manifest = None
    statuses: list = []
    elapsed = 0.0
    merge_ok = False
    try:
        manifest = _build_plan(args, good_jobs, infos, gpu_ids, mode, workdir)
        manifest_path = _write_manifest(workdir, manifest)
        print(f"[multi] plan: {len(manifest['tasks'])} task(s), "
              f"{sum(v['merge'] for v in manifest['videos'])} video(s) to merge",
              flush=True)
        for vi, video in enumerate(manifest["videos"]):
            n = video["n"] if video["n"] is not None else "?"
            print(f"[multi]   video {vi}: {os.path.basename(video['input'])} "
                  f"({n} frames, {len(video['parts'])} part(s)) -> "
                  f"{os.path.basename(video['output'])}", flush=True)

        t0 = time.perf_counter()
        statuses = _run_with_retry(args, workdir, manifest_path, gpu_ids)
        elapsed = time.perf_counter() - t0
        merge_ok = _merge(args, manifest, statuses)
    except Exception as e:                                  # noqa: BLE001
        print(f"[multi] scheduler failed: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        print(f"[multi] work directory kept in {workdir}",
              file=sys.stderr, flush=True)
        return False

    failed = [i for i, s in enumerate(statuses) if not s or not s.get("ok")]
    pairs = sum((v["n"] - 1) for v in manifest["videos"] if v["n"])
    print(f"[multi] {len(manifest['videos'])} video(s), {pairs} pairs in "
          f"{elapsed:.1f}s ({pairs / max(elapsed, 1e-6):.1f} pairs/s wall, "
          f"{len(gpu_ids)} GPUs)", flush=True)
    if failed:
        print(f"[multi] {len(failed)} task(s) failed; parts kept in {workdir}",
              file=sys.stderr, flush=True)
    if bad:
        print(f"[multi] {len(bad)} unreadable input(s): "
              f"{', '.join(os.path.basename(p) for p in bad)}",
              file=sys.stderr, flush=True)
    all_ok = merge_ok and not failed and not bad
    if all_ok:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"[multi] artifacts kept in {workdir}", file=sys.stderr, flush=True)
    return all_ok
