import json
import tempfile
import unittest
from pathlib import Path

from automatic_calibration import (
    AutomaticCalibrationError,
    read_automatic_calibration_bundle,
)


class AutomaticCalibrationBundleTests(unittest.TestCase):
    def _write(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_ready_bundle_requires_matching_validated_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = root / "cal.json"
            validation_path = root / "val.json"
            self._write(
                calibration_path,
                {
                    "created_at": "2026-08-31T01:00:00+00:00",
                    "camera": {"serial_number": "ABC"},
                    "transform": {
                        "rotation_row_major": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        "translation_mm": [1, 2, 3],
                    },
                    "metrics": {"quality_pass": True},
                },
            )
            self._write(
                validation_path,
                {
                    "source_calibration_created_at": "2026-08-31T01:00:00+00:00",
                    "camera_serial_number": "ABC",
                    "metrics": {"quality_pass": True},
                },
            )

            result = read_automatic_calibration_bundle(
                calibration_path=calibration_path,
                validation_path=validation_path,
            )

            self.assertTrue(result["ready_for_targeting"])
            self.assertEqual(result["not_ready_reasons"], [])

    def test_stale_validation_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = root / "cal.json"
            validation_path = root / "val.json"
            self._write(
                calibration_path,
                {
                    "created_at": "new",
                    "camera": {"serial_number": "ABC"},
                    "transform": {
                        "rotation_row_major": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        "translation_mm": [1, 2, 3],
                    },
                    "metrics": {"quality_pass": True},
                },
            )
            self._write(
                validation_path,
                {
                    "source_calibration_created_at": "old",
                    "camera_serial_number": "ABC",
                    "metrics": {"quality_pass": True},
                },
            )

            result = read_automatic_calibration_bundle(
                calibration_path=calibration_path,
                validation_path=validation_path,
            )

            self.assertFalse(result["ready_for_targeting"])
            self.assertIn(
                "validation does not correspond to the current calibration",
                result["not_ready_reasons"],
            )

    def test_missing_file_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(AutomaticCalibrationError):
                read_automatic_calibration_bundle(
                    calibration_path=root / "missing_cal.json",
                    validation_path=root / "missing_val.json",
                )


if __name__ == "__main__":
    unittest.main()
