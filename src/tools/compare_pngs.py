"""Per-frame PSNR between two PNG directories.

    uv run python -m src.tools.compare_pngs dir_a dir_b
"""
import os
import sys

import numpy as np

from src.imgio import read_png


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    a, b = sys.argv[1], sys.argv[2]
    fa, fb = sorted(os.listdir(a)), sorted(os.listdir(b))
    n = min(len(fa), len(fb))
    if not n:
        sys.exit("no pngs")
    se, mx, worst = 0.0, 0.0, []
    for i in range(n):
        x = read_png(os.path.join(a, fa[i])).astype(np.float32)
        y = read_png(os.path.join(b, fb[i])).astype(np.float32)
        d = (x - y) ** 2
        se += d.mean()
        mx = max(mx, float(np.abs(x - y).max()))
        worst.append((float(10 * np.log10(255 ** 2 / max(d.mean(), 1e-9))), i))
    worst.sort()
    print(f"{a} vs {b}: {len(fa)}/{len(fb)} frames, "
          f"PSNR {10*np.log10(255**2/(se/n)):.2f} dB, max|d| {mx:.0f}")
    print("  worst:", [(f"#{i}", round(p, 1)) for p, i in worst[:4]])


if __name__ == "__main__":
    main()
