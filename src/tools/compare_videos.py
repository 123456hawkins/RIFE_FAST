"""Per-frame PSNR between two videos.

    uv run python -m src.tools.compare_videos A.mp4 B.mp4 1280 720
"""
import subprocess
import sys

import numpy as np

from src.ffmpeg_util import ffmpeg


def frames(path, w, h):
    p = subprocess.Popen([ffmpeg(), "-v", "error", "-i", path, "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    n = w * h * 3
    try:
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                break
            yield np.frombuffer(b, np.uint8).reshape(h, w, 3).astype(np.float32)
    finally:
        p.stdout.close()
        p.wait()


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    a, b = sys.argv[1], sys.argv[2]
    w, h = int(sys.argv[3]), int(sys.argv[4])
    n, se, mx, worst = 0, 0.0, 0.0, []
    for i, (x, y) in enumerate(zip(frames(a, w, h), frames(b, w, h))):
        d = (x - y) ** 2
        se += d.mean()
        n += 1
        mx = max(mx, float(np.abs(x - y).max()))
        worst.append((float(10 * np.log10(255 ** 2 / max(d.mean(), 1e-9))), i))
    if not n:
        sys.exit("no frames decoded")
    mse = se / n
    db = 99.0 if mse <= 0 else 10 * np.log10(255 ** 2 / mse)
    print(f"frames {n}  PSNR {db:.2f} dB  max|d| {mx:.0f}")
    worst.sort()
    print("worst frames:", [(f"#{i}", round(p, 1)) for p, i in worst[:5]])


if __name__ == "__main__":
    main()
