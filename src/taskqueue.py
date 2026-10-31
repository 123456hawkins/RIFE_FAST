"""File-backed task queue shared by GPU worker processes.

The parent writes ``workdir/manifest.json`` with a list of tasks and a
``workdir/next`` counter.  Every worker atomically claims the next task with
``flock``:

    with flock(next):
        i = int(next.read() or 0)
        if i >= len(tasks): no work
        next.write(str(i + 1))

This is what makes an idle GPU pick up the next file/segment instead of waiting
for a statically assigned queue to finish.  Status is written per task
(``status/task_00000.json``) so the parent can retry only the tasks whose worker
crashed or failed, without rerunning the whole batch.
"""
from __future__ import annotations

import fcntl
import json
import os


def init_counter(workdir: str) -> None:
    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, "next"), "w") as f:
        f.write("0")


def claim(workdir: str, total: int) -> int | None:
    """Claim the next task index, or ``None`` when the queue is empty."""
    path = os.path.join(workdir, "next")
    with open(path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            raw = f.read().strip()
            idx = int(raw) if raw else 0
            if idx >= total:
                return None
            f.seek(0)
            f.write(str(idx + 1))
            f.truncate()
            f.flush()
            os.fsync(f.fileno())       # visible to the next process immediately
            return idx
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def status_path(status_dir: str, idx: int) -> str:
    return os.path.join(status_dir, f"task_{idx:05d}.json")


def write_status(status_dir: str, idx: int, ok: bool,
                 error: str | None = None) -> None:
    os.makedirs(status_dir, exist_ok=True)
    path = status_path(status_dir, idx)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump({"ok": bool(ok), "error": error}, f)
    os.replace(tmp, path)


def read_status(status_dir: str, idx: int) -> dict | None:
    try:
        with open(status_path(status_dir, idx)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
