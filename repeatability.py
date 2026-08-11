from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from typing import Any
from uuid import uuid4

from landmarks import CALIBRATION_LANDMARKS, FaceLandmarkCapture


def summarize_landmark_samples(
    captures: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Summarize repeated raw U/V measurements for each semantic landmark."""

    by_landmark: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"u": [], "v": []}
    )

    for capture in captures:
        for landmark in capture.get("landmarks", []):
            landmark_id = str(landmark["id"])
            by_landmark[landmark_id]["u"].append(float(landmark["u_px"]))
            by_landmark[landmark_id]["v"].append(float(landmark["v_px"]))

    summary: dict[str, dict[str, float | int]] = {}

    for landmark_id, _ in CALIBRATION_LANDMARKS:
        values = by_landmark.get(landmark_id)
        if not values or not values["u"] or not values["v"]:
            summary[landmark_id] = {"samples": 0}
            continue

        u_values = values["u"]
        v_values = values["v"]

        summary[landmark_id] = {
            "samples": len(u_values),
            "mean_u_px": statistics.fmean(u_values),
            "mean_v_px": statistics.fmean(v_values),
            "stddev_u_px": statistics.stdev(u_values) if len(u_values) > 1 else 0.0,
            "stddev_v_px": statistics.stdev(v_values) if len(v_values) > 1 else 0.0,
            "min_u_px": min(u_values),
            "max_u_px": max(u_values),
            "spread_u_px": max(u_values) - min(u_values),
            "min_v_px": min(v_values),
            "max_v_px": max(v_values),
            "spread_v_px": max(v_values) - min(v_values),
        }

    return summary


def print_summary(summary: dict[str, dict[str, float | int]]) -> None:
    print()
    print("Raw U/V repeatability summary")
    print(
        f"{'landmark':<20} {'n':>3} "
        f"{'mean U':>9} {'sd U':>8} {'spread U':>9} "
        f"{'mean V':>9} {'sd V':>8} {'spread V':>9}"
    )
    print("-" * 88)

    for landmark_id, _ in CALIBRATION_LANDMARKS:
        row = summary[landmark_id]
        samples = int(row.get("samples", 0))
        if samples == 0:
            print(f"{landmark_id:<20} {samples:>3}  no samples")
            continue

        print(
            f"{landmark_id:<20} {samples:>3} "
            f"{float(row['mean_u_px']):>9.2f} "
            f"{float(row['stddev_u_px']):>8.2f} "
            f"{float(row['spread_u_px']):>9.2f} "
            f"{float(row['mean_v_px']):>9.2f} "
            f"{float(row['stddev_v_px']):>8.2f} "
            f"{float(row['spread_v_px']):>9.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a stationary mannequin repeatedly and quantify raw U/V "
            "landmark jitter without involving the iOS app."
        )
    )
    parser.add_argument("--captures", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--minimum-confidence", type=float, default=0.50)
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to save the complete capture + summary JSON.",
    )
    args = parser.parse_args()

    if args.captures < 2:
        parser.error("--captures must be at least 2")
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    if not 0 <= args.minimum_confidence <= 1:
        parser.error("--minimum-confidence must be between 0 and 1")

    detector = FaceLandmarkCapture()
    captures: list[dict[str, Any]] = []

    try:
        for index in range(args.captures):
            print(f"Capture {index + 1}/{args.captures}...", flush=True)
            capture = detector.capture_for_calibration(
                request_id=str(uuid4()),
                return_image=False,
                minimum_confidence=args.minimum_confidence,
            )
            captures.append(capture)

            if index + 1 < args.captures and args.delay > 0:
                time.sleep(args.delay)
    finally:
        detector.close()

    summary = summarize_landmark_samples(captures)
    print_summary(summary)

    if args.json_output:
        payload = {
            "capture_count": len(captures),
            "summary": summary,
            "captures": captures,
        }
        with open(args.json_output, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2)
            output_file.write("\n")
        print(f"\nSaved complete results to {args.json_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
