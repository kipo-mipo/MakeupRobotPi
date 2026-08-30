#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from spray_serpentine_test import (
    DEFAULT_SERVO_GPIO_PIN,
    DirectGPIOServo,
    SprayTestError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test only the Raspberry Pi airbrush servo. "
            "No Klipper motion and no solenoid commands are sent."
        )
    )
    parser.add_argument(
        "--servo-gpio",
        type=int,
        default=DEFAULT_SERVO_GPIO_PIN,
        help="BCM GPIO pin. Default: 18.",
    )
    parser.add_argument(
        "--spray-pulse-us",
        type=int,
        required=True,
        help="Known-safe pulse width that presses the airbrush trigger.",
    )
    parser.add_argument(
        "--release-pulse-us",
        type=int,
        required=True,
        help="Known-safe pulse width that releases the airbrush trigger.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.0,
        help="How long to hold the spray position. Default: 1 second.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if args.hold_seconds <= 0:
            raise SprayTestError("hold-seconds must be greater than zero.")

        print(
            f"Testing direct GPIO servo on BCM GPIO{args.servo_gpio}: "
            f"release={args.release_pulse_us} us, "
            f"spray={args.spray_pulse_us} us"
        )
        print("No robot motion will be commanded.")

        with DirectGPIOServo(
            gpio_pin=args.servo_gpio,
            spray_pulse_us=args.spray_pulse_us,
            release_pulse_us=args.release_pulse_us,
        ) as servo:
            print("Release position...")
            servo.release()
            time.sleep(1.0)

            print("Spray position...")
            servo.spray()
            time.sleep(args.hold_seconds)

            print("Release position...")
            servo.release()
            time.sleep(1.0)

        print("Servo-only test complete.")
        return 0

    except SprayTestError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
