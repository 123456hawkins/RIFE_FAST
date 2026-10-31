"""CPU-only unit tests for the scheduler and the PNG reader.

    uv run python -m unittest discover -s tests -v
"""
from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

from src import ROOT
from src.imgio import read_png
from src.multigpu import _auto_mode, _build_plan, segment_bounds


def _infos(durs):
    return [{"dur": d, "n": int(d * 24) + 1, "fps": 24} for d in durs]


class TestSegmentBounds(unittest.TestCase):
    def test_partition_invariants(self):
        for n in range(3, 200):
            for parts in range(2, min(6, n - 1) + 1):
                b = segment_bounds(n, parts)
                self.assertEqual(len(b), parts)
                self.assertEqual(b[0][0], 0)
                self.assertEqual(b[-1][1], n - 1)
                self.assertTrue(all(p1 > p0 for p0, p1 in b))
                self.assertTrue(all(b[i][1] == b[i + 1][0]
                                    for i in range(len(b) - 1)))


class TestAutoMode(unittest.TestCase):
    def test_single_video_segments(self):
        self.assertEqual(_auto_mode(_infos([100]), [0, 1]), "segments")

    def test_balanced_two_videos_files(self):
        self.assertEqual(_auto_mode(_infos([100, 100]), [0, 1]), "files")

    def test_three_videos_not_multiple_of_gpus(self):
        self.assertEqual(_auto_mode(_infos([100, 100, 100]), [0, 1]), "segments")

    def test_long_plus_short_segments(self):
        self.assertEqual(_auto_mode(_infos([100, 10]), [0, 1]), "segments")

    def test_long_plus_many_shorts_segments(self):
        self.assertEqual(_auto_mode(_infos([100, 10, 10]), [0, 1]), "segments")

    def test_one_gpu_always_files(self):
        self.assertEqual(_auto_mode(_infos([100]), [0]), "files")


class TestPlan(unittest.TestCase):
    args = SimpleNamespace(multi=2, encoder="x264")

    def plan(self, durs, ngpu, mode=None):
        infos = _infos(durs)
        gpus = list(range(ngpu))
        mode = mode or _auto_mode(infos, gpus)
        jobs = [(f"v{i}.mp4", f"o{i}.mp4") for i in range(len(durs))]
        with tempfile.TemporaryDirectory(prefix="rife_plan_test_") as wd:
            manifest = _build_plan(self.args, jobs, infos, gpus, mode, wd)
        return mode, [len(v["parts"]) for v in manifest["videos"]], \
            len(manifest["tasks"])

    def test_single_100s_2gpu(self):
        self.assertEqual(self.plan([100], 2), ("segments", [2], 2))

    def test_single_400s_2gpu(self):
        self.assertEqual(self.plan([400], 2), ("segments", [4], 4))

    def test_three_equal_videos_2gpu(self):
        self.assertEqual(self.plan([100] * 3, 2),
                         ("segments", [2, 2, 2], 6))

    def test_long_plus_short_2gpu(self):
        self.assertEqual(self.plan([100, 10], 2),
                         ("segments", [2, 1], 3))

    def test_long_plus_two_shorts_2gpu(self):
        self.assertEqual(self.plan([100, 10, 10], 2),
                         ("segments", [2, 1, 1], 4))

    def test_five_equal_videos_4gpu(self):
        self.assertEqual(self.plan([100] * 5, 4),
                         ("segments", [2] * 5, 10))

    def test_two_equal_videos_files(self):
        self.assertEqual(self.plan([100, 100], 2),
                         ("files", [1, 1], 2))


class TestPng(unittest.TestCase):
    def test_read_asset(self):
        img = read_png(str(ROOT / "assets" / "I0_0.png"))
        self.assertEqual(img.shape, (256, 448, 3))
        self.assertEqual(img.dtype.name, "uint8")


if __name__ == "__main__":
    unittest.main()
