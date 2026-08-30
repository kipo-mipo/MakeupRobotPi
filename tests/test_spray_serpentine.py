import unittest

from scripts.spray_serpentine_test import (
    SprayTestError,
    build_preview,
    build_serpentine_passes,
    evenly_spaced_positions,
)


class SerpentineSprayTestTests(unittest.TestCase):
    def test_step_over_is_evenly_distributed_without_tiny_last_strip(self) -> None:
        positions = evenly_spaced_positions(
            10.0,
            70.0,
            9.0,
        )

        self.assertEqual(len(positions), 8)
        steps = [
            positions[index + 1] - positions[index]
            for index in range(len(positions) - 1)
        ]
        self.assertTrue(all(step <= 9.0 for step in steps))
        self.assertAlmostEqual(positions[0], 10.0)
        self.assertAlmostEqual(positions[-1], 70.0)
        self.assertAlmostEqual(steps[0], 60.0 / 7.0)

    def test_serpentine_alternates_vertical_direction(self) -> None:
        passes = build_serpentine_passes(
            x_start_mm=20.0,
            x_end_mm=38.0,
            z_min_mm=50.0,
            z_max_mm=150.0,
            step_over_mm=9.0,
        )

        self.assertEqual(len(passes), 3)
        self.assertEqual(
            (passes[0].z_start_mm, passes[0].z_end_mm),
            (50.0, 150.0),
        )
        self.assertEqual(
            (passes[1].z_start_mm, passes[1].z_end_mm),
            (150.0, 50.0),
        )
        self.assertEqual(
            (passes[2].z_start_mm, passes[2].z_end_mm),
            (50.0, 150.0),
        )

    def test_preview_contains_no_y_axis_motion(self) -> None:
        passes = build_serpentine_passes(
            x_start_mm=20.0,
            x_end_mm=38.0,
            z_min_mm=50.0,
            z_max_mm=150.0,
            step_over_mm=9.0,
        )
        commands = build_preview(
            passes=passes,
            spray_speed_mm_s=15.0,
            travel_speed_mm_s=40.0,
            servo_gpio_pin=18,
            spray_pulse_us=1800,
            release_pulse_us=1100,
            spray_settle_ms=150,
            release_settle_ms=100,
        )

        movement_commands = [
            command
            for command in commands
            if command.startswith(("G0 ", "G1 "))
        ]

        self.assertTrue(movement_commands)
        self.assertTrue(
            all(" Y" not in command for command in movement_commands)
        )

    def test_preview_releases_gpio_servo_before_each_x_reposition(self) -> None:
        passes = build_serpentine_passes(
            x_start_mm=20.0,
            x_end_mm=38.0,
            z_min_mm=50.0,
            z_max_mm=150.0,
            step_over_mm=9.0,
        )
        commands = build_preview(
            passes=passes,
            spray_speed_mm_s=15.0,
            travel_speed_mm_s=40.0,
            servo_gpio_pin=18,
            spray_pulse_us=1800,
            release_pulse_us=1100,
            spray_settle_ms=150,
            release_settle_ms=100,
        )

        for index, command in enumerate(commands):
            if command.startswith("G0 X") and " Z" not in command:
                self.assertTrue(
                    any(
                        "GPIO18 release pulse 1100 us" in earlier
                        for earlier in commands[:index]
                    )
                )

    def test_invalid_z_range_is_rejected(self) -> None:
        with self.assertRaises(SprayTestError):
            build_serpentine_passes(
                x_start_mm=20.0,
                x_end_mm=40.0,
                z_min_mm=100.0,
                z_max_mm=100.0,
                step_over_mm=9.0,
            )


if __name__ == "__main__":
    unittest.main()
