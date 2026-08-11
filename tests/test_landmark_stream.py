import io
import unittest

from landmark_stream import LatestFrame, iter_mjpeg_frames


class MJPEGFrameParserTests(unittest.TestCase):
    def test_extracts_concatenated_jpeg_frames(self):
        first = b"\xff\xd8first-frame\xff\xd9"
        second = b"\xff\xd8second-frame\xff\xd9"
        stream = io.BytesIO(b"noise" + first + second)

        self.assertEqual(list(iter_mjpeg_frames(stream)), [first, second])


class LatestFrameTests(unittest.TestCase):
    def test_store_keeps_newest_frame(self):
        store = LatestFrame()
        store.publish(b"first")
        store.publish(b"second")

        frame, sequence, stopped = store.wait_for_next(0, timeout=0)

        self.assertEqual(frame, b"second")
        self.assertEqual(sequence, 2)
        self.assertFalse(stopped)


if __name__ == "__main__":
    unittest.main()
