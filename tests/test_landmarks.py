import unittest

from landmarks import upright_normalized_to_raw_pixel


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


if __name__ == "__main__":
    unittest.main()
