import unittest

import cv2
import numpy as np

from calibration_depth import (
    _deproject_aligned_color_pixel,
    _distorted_rgb_to_aligned_pixel,
)


def geometry(distortion: dict[str, float]) -> dict:
    return {
        "rgb_intrinsic": {
            "fx": 1275.0,
            "fy": 1275.0,
            "cx": 962.0,
            "cy": 549.0,
            "width": 1920,
            "height": 1080,
        },
        "rgb_distortion": distortion,
    }


class DistortionAlignedDepthGeometryTests(unittest.TestCase):
    def test_zero_distortion_keeps_pixel_unchanged(self) -> None:
        g = geometry({})
        u, v = _distorted_rgb_to_aligned_pixel(
            raw_u_px=1234.5,
            raw_v_px=456.25,
            geometry=g,
            width=1920,
            height=1080,
        )
        self.assertAlmostEqual(u, 1234.5, places=9)
        self.assertAlmostEqual(v, 456.25, places=9)

    def test_distorted_rgb_pixel_maps_back_to_ideal_aligned_ray(self) -> None:
        distortion = {
            "k1": 0.1130407527,
            "k2": -0.3673901260,
            "p1": -0.0002084698,
            "p2": -0.0006828689,
            "k3": 0.2929854393,
            "k4": 0.0,
            "k5": 0.0,
            "k6": 0.0,
        }
        g = geometry(distortion)
        intrinsic = g["rgb_intrinsic"]
        camera_matrix = np.array(
            [
                [intrinsic["fx"], 0.0, intrinsic["cx"]],
                [0.0, intrinsic["fy"], intrinsic["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        coefficients = np.array(
            [
                distortion["k1"],
                distortion["k2"],
                distortion["p1"],
                distortion["p2"],
                distortion["k3"],
                distortion["k4"],
                distortion["k5"],
                distortion["k6"],
            ],
            dtype=np.float64,
        )

        # A known ideal RGB optical ray.
        normalized_x = 0.18
        normalized_y = -0.08
        object_point = np.array(
            [[[normalized_x, normalized_y, 1.0]]],
            dtype=np.float64,
        )
        distorted, _ = cv2.projectPoints(
            object_point,
            np.zeros(3),
            np.zeros(3),
            camera_matrix,
            coefficients,
        )
        distorted_u = float(distorted[0, 0, 0])
        distorted_v = float(distorted[0, 0, 1])

        aligned_u, aligned_v = _distorted_rgb_to_aligned_pixel(
            raw_u_px=distorted_u,
            raw_v_px=distorted_v,
            geometry=g,
            width=1920,
            height=1080,
        )

        expected_u = intrinsic["fx"] * normalized_x + intrinsic["cx"]
        expected_v = intrinsic["fy"] * normalized_y + intrinsic["cy"]
        self.assertAlmostEqual(aligned_u, expected_u, places=5)
        self.assertAlmostEqual(aligned_v, expected_v, places=5)

        depth_mm = 300.0
        x_mm, y_mm, z_mm = _deproject_aligned_color_pixel(
            aligned_u_px=aligned_u,
            aligned_v_px=aligned_v,
            depth_mm=depth_mm,
            geometry=g,
            width=1920,
            height=1080,
        )
        self.assertAlmostEqual(x_mm, normalized_x * depth_mm, places=5)
        self.assertAlmostEqual(y_mm, normalized_y * depth_mm, places=5)
        self.assertAlmostEqual(z_mm, depth_mm, places=9)


if __name__ == "__main__":
    unittest.main()
