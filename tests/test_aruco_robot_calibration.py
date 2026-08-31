import unittest

import numpy as np

from aruco_robot_calibration import (
    ArucoCalibrationError,
    build_grid,
    build_xz_move_commands,
    fit_rigid_transform,
    leave_one_out_errors,
    marker_robot_coordinate,
)


class ArucoRobotCalibrationTests(unittest.TestCase):
    def test_default_grid_spans_full_requested_x_and_z(self) -> None:
        grid = build_grid(
            x_min_mm=0.0,
            x_max_mm=220.0,
            z_min_mm=0.0,
            z_max_mm=70.0,
            x_count=5,
            z_count=4,
        )
        self.assertEqual(len(grid), 20)
        self.assertEqual((grid[0].x_mm, grid[0].z_mm), (0.0, 0.0))
        self.assertEqual((grid[4].x_mm, grid[4].z_mm), (220.0, 0.0))
        self.assertEqual((grid[5].x_mm, grid[5].z_mm), (220.0, 70.0 / 3.0))
        self.assertEqual((grid[-1].x_mm, grid[-1].z_mm), (0.0, 70.0))

    def test_motion_commands_never_command_y(self) -> None:
        grid = build_grid(
            x_min_mm=0.0,
            x_max_mm=220.0,
            z_min_mm=0.0,
            z_max_mm=70.0,
            x_count=5,
            z_count=4,
        )
        for point in grid:
            commands = build_xz_move_commands(point, 35.0)
            motion = [command for command in commands if command.startswith(("G0 ", "G1 "))]
            self.assertTrue(motion)
            self.assertTrue(all(" Y" not in command for command in motion))

    def test_marker_offset_is_added_to_robot_command(self) -> None:
        result = marker_robot_coordinate(
            np.asarray([100.0, 0.0, 40.0]),
            np.asarray([2.0, -3.0, 27.5]),
        )
        np.testing.assert_allclose(result, [102.0, -3.0, 67.5])

    def test_rigid_fit_recovers_known_transform_from_coplanar_points(self) -> None:
        camera = np.asarray(
            [
                [-50.0, 0.0, 280.0],
                [0.0, 0.0, 300.0],
                [50.0, 0.0, 320.0],
                [-45.0, 0.0, 285.0],
                [5.0, 0.0, 305.0],
                [55.0, 0.0, 325.0],
            ],
            dtype=np.float64,
        )
        camera[:, 1] = 0.2 * camera[:, 0] + 0.1 * camera[:, 2] - 20.0

        angle = np.deg2rad(23.0)
        rotation = np.asarray(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ],
            dtype=np.float64,
        )
        translation = np.asarray([110.0, -18.0, 42.0])
        robot = (rotation @ camera.T).T + translation

        fit = fit_rigid_transform(camera, robot)
        np.testing.assert_allclose(fit.rotation, rotation, atol=1e-10)
        np.testing.assert_allclose(fit.translation, translation, atol=1e-10)
        self.assertLess(fit.rms_mm, 1e-9)
        self.assertLess(float(np.max(leave_one_out_errors(camera, robot))), 1e-8)

    def test_rigid_fit_rejects_collinear_points(self) -> None:
        camera = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
        )
        robot = camera + np.asarray([1.0, 2.0, 3.0])
        with self.assertRaises(ArucoCalibrationError):
            fit_rigid_transform(camera, robot)


if __name__ == "__main__":
    unittest.main()
