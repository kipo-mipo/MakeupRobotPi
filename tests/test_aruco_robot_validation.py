import unittest

import numpy as np

from scripts.validate_aruco_robot import (
    apply_transform,
    default_validation_points,
    residual_stats,
)


class ArucoHeldoutValidationTests(unittest.TestCase):
    def test_default_points_are_off_grid_and_span_workspace(self) -> None:
        points = default_validation_points()
        self.assertEqual(len(points), 8)

        training_x = {0.0, 55.0, 110.0, 165.0, 220.0}
        training_z = {0.0, 70.0 / 3.0, 140.0 / 3.0, 70.0}

        self.assertTrue(all(point.x_mm not in training_x for point in points))
        self.assertTrue(all(all(abs(point.z_mm - z) > 0.01 for z in training_z) for point in points))
        self.assertLessEqual(min(point.x_mm for point in points), 27.5)
        self.assertGreaterEqual(max(point.x_mm for point in points), 192.5)
        self.assertLessEqual(min(point.z_mm for point in points), 11.67)
        self.assertGreaterEqual(max(point.z_mm for point in points), 58.33)

    def test_apply_transform_uses_saved_rotation_and_translation(self) -> None:
        camera = np.asarray([1.0, 2.0, 3.0])
        rotation = [
            0.0, -1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
        ]
        translation = [10.0, 20.0, 30.0]
        result = apply_transform(camera, rotation, translation)
        np.testing.assert_allclose(result, [8.0, 21.0, 33.0])

    def test_residual_stats_returns_rms_and_max(self) -> None:
        rms, maximum = residual_stats([1.0, 2.0, 2.0])
        self.assertAlmostEqual(rms, np.sqrt(3.0))
        self.assertEqual(maximum, 2.0)


if __name__ == "__main__":
    unittest.main()
