import unittest

from repeatability import summarize_landmark_samples


class RepeatabilitySummaryTests(unittest.TestCase):
    def test_summary_reports_mean_sample_stddev_and_spread(self):
        captures = [
            {
                "landmarks": [
                    {"id": "nose_tip", "u_px": 10.0, "v_px": 20.0},
                ]
            },
            {
                "landmarks": [
                    {"id": "nose_tip", "u_px": 12.0, "v_px": 24.0},
                ]
            },
            {
                "landmarks": [
                    {"id": "nose_tip", "u_px": 14.0, "v_px": 22.0},
                ]
            },
        ]

        summary = summarize_landmark_samples(captures)["nose_tip"]

        self.assertEqual(summary["samples"], 3)
        self.assertAlmostEqual(summary["mean_u_px"], 12.0)
        self.assertAlmostEqual(summary["mean_v_px"], 22.0)
        self.assertAlmostEqual(summary["stddev_u_px"], 2.0)
        self.assertAlmostEqual(summary["stddev_v_px"], 2.0)
        self.assertAlmostEqual(summary["spread_u_px"], 4.0)
        self.assertAlmostEqual(summary["spread_v_px"], 4.0)

    def test_missing_landmark_reports_zero_samples(self):
        summary = summarize_landmark_samples([])
        self.assertEqual(summary["chin"]["samples"], 0)


if __name__ == "__main__":
    unittest.main()
