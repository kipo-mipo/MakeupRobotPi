import unittest

from robot_motion import RobotMotionError, build_test_move_plan


class CalibrationPointMotionPlanTests(unittest.TestCase):
    def test_plan_retracts_then_moves_xz_then_approaches_y(self) -> None:
        plan = build_test_move_plan(
            x_mm=120.0,
            y_mm=135.0,
            z_mm=80.0,
            axis_minimum=[0.0, 0.0, 0.0],
            axis_maximum=[220.0, 220.0, 250.0],
            safe_y=0.0,
            retract_feed_mm_min=900.0,
            travel_feed_mm_min=1800.0,
            approach_feed_mm_min=600.0,
        )

        self.assertEqual(
            plan["commands"],
            [
                "G90",
                "G0 Y0.000 F900",
                "G0 X120.000 Z80.000 F1800",
                "G0 Y135.000 F600",
            ],
        )
        self.assertEqual(plan["target"]["x_mm"], 120.0)
        self.assertEqual(plan["target"]["y_mm"], 135.0)
        self.assertEqual(plan["target"]["z_mm"], 80.0)

    def test_target_outside_axis_limit_is_rejected(self) -> None:
        with self.assertRaises(RobotMotionError):
            build_test_move_plan(
                x_mm=221.0,
                y_mm=100.0,
                z_mm=80.0,
                axis_minimum=[0.0, 0.0, 0.0],
                axis_maximum=[220.0, 220.0, 250.0],
                safe_y=0.0,
            )

    def test_target_cannot_move_behind_safe_y_plane(self) -> None:
        with self.assertRaises(RobotMotionError):
            build_test_move_plan(
                x_mm=100.0,
                y_mm=-1.0,
                z_mm=80.0,
                axis_minimum=[0.0, -10.0, 0.0],
                axis_maximum=[220.0, 220.0, 250.0],
                safe_y=0.0,
            )

    def test_safe_y_must_be_inside_machine_limits(self) -> None:
        with self.assertRaises(RobotMotionError):
            build_test_move_plan(
                x_mm=100.0,
                y_mm=100.0,
                z_mm=80.0,
                axis_minimum=[0.0, 10.0, 0.0],
                axis_maximum=[220.0, 220.0, 250.0],
                safe_y=0.0,
            )


if __name__ == "__main__":
    unittest.main()
