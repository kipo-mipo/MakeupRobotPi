#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_MOONRAKER_URL = "http://127.0.0.1:7125"
DEFAULT_STEP_OVER_MM = 9.0
DEFAULT_SPRAY_SPEED_MM_S = 15.0
DEFAULT_TRAVEL_SPEED_MM_S = 40.0
DEFAULT_SPRAY_SETTLE_MS = 150
DEFAULT_RELEASE_SETTLE_MS = 100
DEFAULT_SERVO_GPIO_PIN = 18
DEFAULT_SERVO_FREQUENCY_HZ = 50
MIN_SAFE_SERVO_PULSE_US = 400
MAX_SAFE_SERVO_PULSE_US = 2600


class SprayTestError(RuntimeError):
    pass


class DirectGPIOServo:
    """Direct Raspberry Pi GPIO servo output using lgpio.

    Pulse widths are explicit so the test never guesses trigger geometry.
    The GPIO PWM signal is independent of Klipper/Moonraker.
    """

    def __init__(
        self,
        *,
        gpio_pin: int,
        spray_pulse_us: int,
        release_pulse_us: int,
        frequency_hz: int = DEFAULT_SERVO_FREQUENCY_HZ,
    ) -> None:
        self.gpio_pin = int(gpio_pin)
        self.spray_pulse_us = self._validated_pulse(
            spray_pulse_us,
            "spray pulse",
        )
        self.release_pulse_us = self._validated_pulse(
            release_pulse_us,
            "release pulse",
        )
        self.frequency_hz = int(frequency_hz)
        if self.gpio_pin < 0:
            raise SprayTestError("GPIO pin must be non-negative.")
        if self.frequency_hz <= 0:
            raise SprayTestError("Servo frequency must be positive.")

        try:
            import lgpio  # type: ignore
        except ImportError as exc:
            raise SprayTestError(
                "Direct GPIO servo control requires the Python lgpio package. "
                "On Raspberry Pi OS install it in the active environment "
                "(for example: pip install lgpio) and rerun the test."
            ) from exc

        self._lgpio = lgpio
        self._chip = None

    @staticmethod
    def _validated_pulse(value: int, name: str) -> int:
        pulse = int(value)
        if not MIN_SAFE_SERVO_PULSE_US <= pulse <= MAX_SAFE_SERVO_PULSE_US:
            raise SprayTestError(
                f"{name} must be between "
                f"{MIN_SAFE_SERVO_PULSE_US} and "
                f"{MAX_SAFE_SERVO_PULSE_US} microseconds."
            )
        return pulse

    def open(self) -> None:
        if self._chip is not None:
            return

        chip = self._lgpio.gpiochip_open(0)
        try:
            self._lgpio.gpio_claim_output(
                chip,
                self.gpio_pin,
                0,
            )
            self._chip = chip
            self.release()
        except Exception:
            self._lgpio.gpiochip_close(chip)
            raise

    def _set_pulse(self, pulse_us: int) -> None:
        if self._chip is None:
            raise SprayTestError("GPIO servo is not open.")

        result_code = self._lgpio.tx_servo(
            self._chip,
            self.gpio_pin,
            int(pulse_us),
            self.frequency_hz,
        )
        if result_code < 0:
            raise SprayTestError(
                "lgpio rejected the servo pulse "
                f"{pulse_us} us on GPIO{self.gpio_pin} "
                f"(code {result_code})."
            )

    def spray(self) -> None:
        self._set_pulse(self.spray_pulse_us)

    def release(self) -> None:
        self._set_pulse(self.release_pulse_us)

    def close(self) -> None:
        if self._chip is None:
            return

        chip = self._chip
        self._chip = None

        try:
            # Leave enough time for the airbrush trigger to mechanically
            # return before removing the PWM signal.
            self._lgpio.tx_servo(
                chip,
                self.gpio_pin,
                self.release_pulse_us,
                self.frequency_hz,
            )
            time.sleep(0.25)
            self._lgpio.tx_servo(
                chip,
                self.gpio_pin,
                0,
                self.frequency_hz,
            )
            self._lgpio.gpio_free(chip, self.gpio_pin)
        finally:
            self._lgpio.gpiochip_close(chip)

    def __enter__(self) -> "DirectGPIOServo":
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


@dataclass(frozen=True)
class SprayPass:
    index: int
    x_mm: float
    z_start_mm: float
    z_end_mm: float


def finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise SprayTestError(f"{name} must be finite.")
    return value


def evenly_spaced_positions(
    start_mm: float,
    end_mm: float,
    maximum_step_mm: float,
) -> list[float]:
    start = finite(start_mm, "X start")
    end = finite(end_mm, "X end")
    maximum_step = finite(maximum_step_mm, "step-over")

    if maximum_step <= 0:
        raise SprayTestError("Step-over must be greater than zero.")

    span = abs(end - start)
    if span <= 1e-9:
        return [start]

    interval_count = max(1, int(math.ceil(span / maximum_step)))
    step = (end - start) / interval_count
    return [start + step * index for index in range(interval_count + 1)]


def build_serpentine_passes(
    *,
    x_start_mm: float,
    x_end_mm: float,
    z_min_mm: float,
    z_max_mm: float,
    step_over_mm: float,
) -> list[SprayPass]:
    z_min = finite(z_min_mm, "Z minimum")
    z_max = finite(z_max_mm, "Z maximum")
    if z_max <= z_min:
        raise SprayTestError("Z maximum must be greater than Z minimum.")

    x_positions = evenly_spaced_positions(
        x_start_mm,
        x_end_mm,
        step_over_mm,
    )

    passes: list[SprayPass] = []
    for index, x_mm in enumerate(x_positions):
        if index % 2 == 0:
            z_start, z_end = z_min, z_max
        else:
            z_start, z_end = z_max, z_min

        passes.append(
            SprayPass(
                index=index + 1,
                x_mm=x_mm,
                z_start_mm=z_start,
                z_end_mm=z_end,
            )
        )

    return passes


def moonraker_base_url() -> str:
    return os.getenv("MOONRAKER_URL", DEFAULT_MOONRAKER_URL).rstrip("/")


def request_json(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    url = moonraker_base_url() + path
    data = None
    headers: dict[str, str] = {}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SprayTestError(
            f"Moonraker returned HTTP {exc.code}: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SprayTestError(
            f"Could not reach Moonraker at {moonraker_base_url()}: {exc}"
        ) from exc

    try:
        decoded = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise SprayTestError("Moonraker returned invalid JSON.") from exc

    if not isinstance(decoded, dict):
        raise SprayTestError("Moonraker returned an unexpected response.")

    if "error" in decoded:
        raise SprayTestError(f"Moonraker error: {decoded['error']}")

    return decoded


def result(payload: dict[str, Any]) -> Any:
    if "result" not in payload:
        raise SprayTestError("Moonraker response is missing result.")
    return payload["result"]


def printer_status() -> dict[str, Any]:
    printer_info = result(request_json("GET", "/printer/info"))
    query = result(
        request_json(
            "POST",
            "/printer/objects/query",
            {
                "objects": {
                    "toolhead": [
                        "homed_axes",
                        "position",
                        "axis_minimum",
                        "axis_maximum",
                    ],
                    "print_stats": ["state"],
                }
            },
        )
    )

    status = query.get("status") or {}
    toolhead = status.get("toolhead") or {}
    print_stats = status.get("print_stats") or {}

    return {
        "printer_state": str(printer_info.get("state") or ""),
        "homed_axes": str(toolhead.get("homed_axes") or ""),
        "position": toolhead.get("position"),
        "axis_minimum": toolhead.get("axis_minimum"),
        "axis_maximum": toolhead.get("axis_maximum"),
        "print_state": str(print_stats.get("state") or ""),
    }


def validate_status_and_bounds(
    status: dict[str, Any],
    passes: list[SprayPass],
) -> None:
    if status["printer_state"].lower() != "ready":
        raise SprayTestError(
            "Klipper is not ready "
            f"(state={status['printer_state'] or 'unknown'})."
        )

    homed = status["homed_axes"].lower()
    missing = [
        axis.upper()
        for axis in ("x", "z")
        if axis not in homed
    ]
    if missing:
        raise SprayTestError(
            "Home X and Z before this spray test; missing "
            + ", ".join(missing)
            + "."
        )

    print_state = status["print_state"].lower()
    if print_state in {"printing", "paused"}:
        raise SprayTestError(
            f"Spray test is blocked while print state is {print_state}."
        )

    axis_minimum = status.get("axis_minimum")
    axis_maximum = status.get("axis_maximum")
    if (
        not isinstance(axis_minimum, list)
        or not isinstance(axis_maximum, list)
        or len(axis_minimum) < 3
        or len(axis_maximum) < 3
    ):
        raise SprayTestError("Klipper did not report XYZ axis limits.")

    x_min = float(axis_minimum[0])
    x_max = float(axis_maximum[0])
    z_min = float(axis_minimum[2])
    z_max = float(axis_maximum[2])

    for spray_pass in passes:
        if not x_min <= spray_pass.x_mm <= x_max:
            raise SprayTestError(
                f"X={spray_pass.x_mm:.3f} mm is outside "
                f"Klipper limits [{x_min:.3f}, {x_max:.3f}]."
            )
        for z_value in (
            spray_pass.z_start_mm,
            spray_pass.z_end_mm,
        ):
            if not z_min <= z_value <= z_max:
                raise SprayTestError(
                    f"Z={z_value:.3f} mm is outside "
                    f"Klipper limits [{z_min:.3f}, {z_max:.3f}]."
                )


def send_gcode(
    commands: list[str],
    *,
    timeout_seconds: float = 60.0,
) -> Any:
    script = "\n".join(commands)
    return result(
        request_json(
            "POST",
            "/printer/gcode/script",
            {"script": script},
            timeout_seconds=timeout_seconds,
        )
    )


def movement_feed(speed_mm_s: float, name: str) -> float:
    speed = finite(speed_mm_s, name)
    if speed <= 0:
        raise SprayTestError(f"{name} must be greater than zero.")
    return speed * 60.0


def build_preview(
    *,
    passes: list[SprayPass],
    spray_speed_mm_s: float,
    travel_speed_mm_s: float,
    servo_gpio_pin: int,
    spray_pulse_us: int,
    release_pulse_us: int,
    spray_settle_ms: int,
    release_settle_ms: int,
) -> list[str]:
    spray_feed = movement_feed(
        spray_speed_mm_s,
        "spray speed",
    )
    travel_feed = movement_feed(
        travel_speed_mm_s,
        "travel speed",
    )

    if spray_settle_ms < 0 or release_settle_ms < 0:
        raise SprayTestError("Servo settle times cannot be negative.")

    commands = [
        "M400",
        "G90",
        f"; GPIO{servo_gpio_pin} release pulse {release_pulse_us} us",
        f"G0 X{passes[0].x_mm:.3f} "
        f"Z{passes[0].z_start_mm:.3f} "
        f"F{travel_feed:.0f}",
        "M400",
    ]

    for index, spray_pass in enumerate(passes):
        commands.extend(
            [
                f"; GPIO{servo_gpio_pin} spray pulse {spray_pulse_us} us",
                f"; wait {spray_settle_ms} ms for servo",
                f"G1 Z{spray_pass.z_end_mm:.3f} "
                f"F{spray_feed:.0f}",
                "M400",
                f"; GPIO{servo_gpio_pin} release pulse {release_pulse_us} us",
                f"; wait {release_settle_ms} ms before X travel",
            ]
        )

        if index + 1 < len(passes):
            next_pass = passes[index + 1]
            commands.extend(
                [
                    f"G0 X{next_pass.x_mm:.3f} "
                    f"F{travel_feed:.0f}",
                    "M400",
                ]
            )

    commands.extend(
        [
            f"; GPIO{servo_gpio_pin} release pulse {release_pulse_us} us",
            "M400",
        ]
    )
    return commands


def execute_pattern(
    *,
    passes: list[SprayPass],
    spray_speed_mm_s: float,
    travel_speed_mm_s: float,
    servo: DirectGPIOServo,
    spray_settle_ms: int,
    release_settle_ms: int,
) -> None:
    spray_feed = movement_feed(
        spray_speed_mm_s,
        "spray speed",
    )
    travel_feed = movement_feed(
        travel_speed_mm_s,
        "travel speed",
    )

    try:
        send_gcode(
            [
                "M400",
                "G90",
                f"G0 X{passes[0].x_mm:.3f} "
                f"Z{passes[0].z_start_mm:.3f} "
                f"F{travel_feed:.0f}",
                "M400",
            ]
        )

        for index, spray_pass in enumerate(passes):
            print(
                f"Pass {spray_pass.index}/{len(passes)}: "
                f"X={spray_pass.x_mm:.2f} mm, "
                f"Z {spray_pass.z_start_mm:.2f} "
                f"→ {spray_pass.z_end_mm:.2f} mm"
            )

            servo.spray()
            time.sleep(spray_settle_ms / 1000.0)

            send_gcode(
                [
                    f"G1 Z{spray_pass.z_end_mm:.3f} "
                    f"F{spray_feed:.0f}",
                    "M400",
                ],
                timeout_seconds=120.0,
            )

            servo.release()
            time.sleep(release_settle_ms / 1000.0)

            if index + 1 < len(passes):
                next_pass = passes[index + 1]
                send_gcode(
                    [
                        f"G0 X{next_pass.x_mm:.3f} "
                        f"F{travel_feed:.0f}",
                        "M400",
                    ]
                )
    finally:
        try:
            servo.release()
            time.sleep(max(release_settle_ms, 100) / 1000.0)
        except Exception as exc:
            print(
                "WARNING: failed to send final GPIO servo release: "
                f"{exc}",
                file=sys.stderr,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a mannequin-only serpentine X/Z airbrush test. "
            "The script never commands Y and never controls a solenoid."
        )
    )

    parser.add_argument("--x-start", type=float, required=True)
    parser.add_argument("--x-end", type=float, required=True)
    parser.add_argument("--z-min", type=float, required=True)
    parser.add_argument("--z-max", type=float, required=True)

    parser.add_argument(
        "--step-over",
        type=float,
        default=DEFAULT_STEP_OVER_MM,
        help="Maximum X spacing between vertical spray passes in mm.",
    )
    parser.add_argument(
        "--spray-speed",
        type=float,
        default=DEFAULT_SPRAY_SPEED_MM_S,
        help="Vertical spray velocity in mm/s. Default: 15.",
    )
    parser.add_argument(
        "--travel-speed",
        type=float,
        default=DEFAULT_TRAVEL_SPEED_MM_S,
        help="Spray-off X reposition velocity in mm/s. Default: 40.",
    )
    parser.add_argument(
        "--spray-settle-ms",
        type=int,
        default=DEFAULT_SPRAY_SETTLE_MS,
        help="Pause after servo spray-on before vertical motion.",
    )
    parser.add_argument(
        "--release-settle-ms",
        type=int,
        default=DEFAULT_RELEASE_SETTLE_MS,
        help="Pause after servo spray-off before X reposition.",
    )

    parser.add_argument(
        "--servo-gpio",
        type=int,
        default=DEFAULT_SERVO_GPIO_PIN,
        help="BCM GPIO pin driving the airbrush servo. Default: 18.",
    )
    parser.add_argument(
        "--spray-pulse-us",
        type=int,
        required=True,
        help=(
            "Known-safe servo pulse width in microseconds that "
            "presses the airbrush trigger."
        ),
    )
    parser.add_argument(
        "--release-pulse-us",
        type=int,
        required=True,
        help=(
            "Known-safe servo pulse width in microseconds that "
            "releases the airbrush trigger."
        ),
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually move and actuate the spray servo. "
            "Without this flag the script only prints a preview."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        DirectGPIOServo._validated_pulse(
            args.spray_pulse_us,
            "spray pulse",
        )
        DirectGPIOServo._validated_pulse(
            args.release_pulse_us,
            "release pulse",
        )

        passes = build_serpentine_passes(
            x_start_mm=args.x_start,
            x_end_mm=args.x_end,
            z_min_mm=args.z_min,
            z_max_mm=args.z_max,
            step_over_mm=args.step_over,
        )

        actual_step = (
            abs(passes[1].x_mm - passes[0].x_mm)
            if len(passes) > 1
            else 0.0
        )

        preview = build_preview(
            passes=passes,
            spray_speed_mm_s=args.spray_speed,
            travel_speed_mm_s=args.travel_speed,
            servo_gpio_pin=args.servo_gpio,
            spray_pulse_us=args.spray_pulse_us,
            release_pulse_us=args.release_pulse_us,
            spray_settle_ms=args.spray_settle_ms,
            release_settle_ms=args.release_settle_ms,
        )

        print("\nSerpentine mannequin spray test")
        print("--------------------------------")
        print(f"Vertical passes: {len(passes)}")
        print(f"Actual X step-over: {actual_step:.2f} mm")
        print(f"Vertical spray speed: {args.spray_speed:.2f} mm/s")
        print(f"Servo GPIO (BCM): {args.servo_gpio}")
        print(f"Spray pulse: {args.spray_pulse_us} us")
        print(f"Release pulse: {args.release_pulse_us} us")
        print("Servo control: direct Raspberry Pi GPIO via lgpio")
        print("Y motion: NONE")
        print("Solenoid control: NONE")

        print("\nPlanned passes:")
        for spray_pass in passes:
            print(
                f"  {spray_pass.index:02d}: "
                f"X={spray_pass.x_mm:.2f}, "
                f"Z {spray_pass.z_start_mm:.2f}"
                f" -> {spray_pass.z_end_mm:.2f}"
            )

        if not args.execute:
            print("\nPREVIEW ONLY — no motion or spray command was sent.")
            print("\nGenerated G-code:")
            print("\n".join(preview))
            print("\nRe-run with --execute when the bounds are verified.")
            return 0

        status = printer_status()
        validate_status_and_bounds(status, passes)

        print("\nEXECUTING on mannequin. Ctrl-C requests GPIO18 servo release.")
        with DirectGPIOServo(
            gpio_pin=args.servo_gpio,
            spray_pulse_us=args.spray_pulse_us,
            release_pulse_us=args.release_pulse_us,
        ) as servo:
            execute_pattern(
                passes=passes,
                spray_speed_mm_s=args.spray_speed,
                travel_speed_mm_s=args.travel_speed,
                servo=servo,
                spray_settle_ms=args.spray_settle_ms,
                release_settle_ms=args.release_settle_ms,
            )
        print("Pattern complete; GPIO spray servo released.")
        return 0

    except (SprayTestError, KeyboardInterrupt) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
