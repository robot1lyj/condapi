# Standard-library unittest entrypoint is intentional.
# ruff: noqa: PT009, PLC0415
"""CPU tests for annotation windows; no models or training loops."""

from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("windows", Path(__file__).with_name("prepare_flora_windows.py"))
windows = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(windows)


class WindowTests(unittest.TestCase):
    def test_every_candidate_frame_owned_once_context_bounded(self):
        report = Path(__file__).resolve().parents[2] / "docs/reports/data-tshirt-fineart-design-20261103"
        for pool in ("train", "dev", "reserve"):
            for row in json.loads((report / f"{pool}-candidate-episodes.json").read_text()):
                length = row["length"]
                intervals = windows.intervals(length)
                cursor = 0
                for a, b, start, end in intervals:
                    self.assertEqual(a, cursor)
                    self.assertLessEqual(start, a)
                    self.assertLessEqual(b, end)
                    self.assertGreater(end, start)
                    self.assertLessEqual(end - start, 3600)
                    self.assertGreaterEqual(start, 0)
                    self.assertLessEqual(end, length)
                    cursor = b
                self.assertEqual(cursor, length)

    def test_non_keyframe_remux_preserves_pixels_and_frame_zero(self):
        try:
            import av
            import numpy as np
        except ImportError:
            self.skipTest("PyAV and NumPy needed for packet remux test")
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.mp4", Path(directory) / "window.mp4"
            try:
                container = av.open(str(source), "w")
                stream = container.add_stream("libx264", rate=30)
            except av.error.UnknownCodecError:
                self.skipTest("Synthetic H264 encoder unavailable")
            stream.width = stream.height = 32
            stream.pix_fmt = "yuv420p"
            stream.options = {"g": "30", "bf": "0", "sc_threshold": "0"}
            with container:
                for i in range(90):
                    frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 2, dtype=np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            with av.open(str(source)) as container:
                expected = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)][7:48]
            for index, start in enumerate(
                (Fraction(7, 30), Fraction(7, 30) + Fraction(1, 10**12), Fraction(7, 30) - Fraction(1, 10**12))
            ):
                with self.subTest(start=str(start)):
                    target = output.with_name(f"window-{index}.mp4")
                    windows.remux(source, target, start, 41)
                    with av.open(str(target)) as container:
                        actual = list(container.decode(video=0))
                    self.assertEqual(len(actual), 41)
                    for i, (image, frame) in enumerate(zip(expected, actual, strict=True)):
                        self.assertLess(abs(float(frame.pts * frame.time_base) - i / 30), 1e-5)
                        self.assertTrue(np.array_equal(image, frame.to_ndarray(format="rgb24")))


if __name__ == "__main__":
    unittest.main()
