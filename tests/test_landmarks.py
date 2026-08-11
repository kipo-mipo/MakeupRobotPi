import base64
import io
import unittest

import numpy as np
from PIL import Image

from landmarks import (
    CALIBRATION_LANDMARKS,
    encode_raw_rgb_as_jpeg_base64,
    landmark_confidence,
    upright_normalized_to_raw_pixel,
)


class OrientationMappingTests(unittest.TestCase):
    def test_upright_top_left_maps_to_raw_top_right(self):
        self.assertEqual(
            upright_normalized_to_raw_pixel(0.0, 0.0),
            (1279.0, 0.0),
        )

    def test_upright_bottom_right_maps_to_raw_bottom_left(self):
        self.assertEqual(
            upright_normalized_to_raw_pixel(1.0, 1.0),
            (0.0, 719.0),
        )

    def test_upright_center_maps_to_raw_center(self):
        raw_u, raw_v = upright_normalized_to_raw_pixel(0.5, 0.5)

        self.assertAlmostEqual(raw_u, 639.5)
        self.assertAlmostEqual(raw_v, 359.5)

    def test_out_of_range_detector_values_are_clamped(self):
        raw_u, raw_v = upright_normalized_to_raw_pixel(-0.1, 1.1)

        self.assertEqual(raw_u, 0.0)
        self.assertEqual(raw_v, 0.0)


class CalibrationLandmarkContractTests(unittest.TestCase):
    def test_semantic_landmark_mapping_is_stable(self):
        self.assertEqual(
            dict(CALIBRATION_LANDMARKS),
            {
                "nose_tip": 1,
                "left_inner_eye": 362,
                "right_inner_eye": 133,
                "left_mouth_corner": 291,
                "right_mouth_corner": 61,
                "chin": 152,
            },
        )

    def test_missing_per_landmark_confidence_uses_conservative_floor(self):
        class Landmark:
            presence = None
            visibility = None

        score, source = landmark_confidence(Landmark(), fallback=0.50)

        self.assertEqual(score, 0.50)
        self.assertEqual(source, "model_acceptance_floor")

    def test_presence_and_visibility_use_more_conservative_score(self):
        class Landmark:
            presence = 0.92
            visibility = 0.81

        score, source = landmark_confidence(Landmark())

        self.assertEqual(score, 0.81)
        self.assertEqual(source, "min_presence_visibility")

    def test_returned_jpeg_keeps_raw_frame_dimensions(self):
        raw = np.zeros((3, 5, 3), dtype=np.uint8)
        encoded = encode_raw_rgb_as_jpeg_base64(raw)
        decoded = Image.open(io.BytesIO(base64.b64decode(encoded)))

        self.assertEqual(decoded.size, (5, 3))


if __name__ == "__main__":
    unittest.main()
