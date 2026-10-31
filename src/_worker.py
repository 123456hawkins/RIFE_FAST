"""Internal entry point for one multi-GPU worker.

The parent (``src.multigpu``) starts ``python -m src._worker`` once per GPU with

  * ``CUDA_VISIBLE_DEVICES`` pinned to that card,
  * ``RIFE_WORKER_JSON`` holding ``{"args": <cli namespace>, "mode": "queue",
    "workdir": ..., "manifest": ..., "status_dir": ...}``.

The worker builds the model/runtime **once**, then atomically claims tasks from
the shared queue until it is empty.  An idle card therefore starts the next
file/segment immediately; it does not wait for a statically assigned list.
Running each card in its own process keeps CUDA contexts, TensorRT engines,
CUDA graphs and Triton's JIT cache independent.

This module is not part of the public CLI.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402


def _setup():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def _tag() -> str:
    return f"[gpu {os.environ.get('RIFE_GPU_TAG', '?')}]"


def _queue(spec: dict, args) -> int:
    """Claim and run tasks from the shared queue; returns an exit code."""
    from src import cli, taskqueue

    with open(spec["manifest"]) as f:
        manifest = json.load(f)
    tasks = manifest["tasks"]
    workdir, status_dir = spec["workdir"], spec["status_dir"]
    tag = _tag()
    # claim first: a worker that loses the race for the last task shouldn't pay
    # the model/engine load cost
    idx = taskqueue.claim(workdir, len(tasks))
    if idx is None:
        return 0
    runtime = cli.Runtime(args)
    failed = 0
    while idx is not None:
        task = tasks[idx]
        seg = ""
        if task.get("start_frame") is not None:
            seg = f" [{task['start_frame']}..{task['end_frame'] - 1}]"
        print(f"{tag} task {idx + 1}/{len(tasks)}: {task['input']}{seg} "
              f"-> {task['output']}", flush=True)
        try:
            cli.run_video(args, task["input"], task["output"], runtime,
                          start_frame=task.get("start_frame"),
                          end_frame=task.get("end_frame"),
                          skip_first=bool(task.get("skip_first")))
        except Exception as e:                              # noqa: BLE001
            traceback.print_exc()
            taskqueue.write_status(status_dir, idx, False,
                                   f"{type(e).__name__}: {e}")
            failed += 1
        else:
            taskqueue.write_status(status_dir, idx, True)
        idx = taskqueue.claim(workdir, len(tasks))
    if failed:
        print(f"{tag} failed {failed} task(s)", file=sys.stderr, flush=True)
    return 1 if failed else 0


def _jobs(spec: dict, args) -> int:
    """Legacy mode: a fixed list of jobs (kept for compatibility/tests)."""
    from src import cli

    jobs = spec.get("jobs", [])
    runtime = cli.Runtime(args)
    tag = _tag()
    failed = 0
    for job in jobs:
        src, dst = job["input"], job["output"]
        seg = ""
        if job.get("start_frame") is not None:
            seg = f" [{job['start_frame']}..{job['end_frame'] - 1}]"
        print(f"{tag} {src}{seg} -> {dst}", flush=True)
        try:
            cli.run_video(args, src, dst, runtime,
                          start_frame=job.get("start_frame"),
                          end_frame=job.get("end_frame"),
                          skip_first=bool(job.get("skip_first")))
        except Exception:                                  # noqa: BLE001
            traceback.print_exc()
            failed += 1
            if job.get("start_frame") is not None:
                break                    # a broken segment makes concat useless
    if failed:
        print(f"{tag} failed: {failed} job(s)", file=sys.stderr, flush=True)
        return 1
    return 0


def main() -> int:
    spec = json.loads(os.environ["RIFE_WORKER_JSON"])
    args = argparse.Namespace(**spec["args"])
    _setup()

    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"{_tag()} {dev}  torch {torch.__version__}", flush=True)
    if spec.get("mode") == "queue":
        return _queue(spec, args)
    return _jobs(spec, args)


if __name__ == "__main__":
    sys.exit(main())
