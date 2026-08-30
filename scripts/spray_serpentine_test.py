#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
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


class SprayTestError(RuntimeError):
    pass


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


def resolve_spray_commands(args: argparse.Namespace) -> tuple[str, str]:
    on_command = (
        args.spray_on_gcode
        or os.getenv("MAKEUP_SPRAY_ON_GCODE")
    )
    off_command = (
        args.spray_off_gcode
        or os.getenv("MAKEUP_SPRAY_OFF_GCODE")
    )

    if on_command and off_command:
        return on_command.strip(), off_command.strip()

    if (
        args.servo_name
        and args.spray_angle is not None
        and args.release_angle is not None
    ):
        return (
            "SET_SERVO "
            f"SERVO={args.servo_name} "
            f"ANGLE={args.spray_angle:g}",
            "SET_SERVO "
            f"SERVO={args.servo_name} "
            f"ANGLE={args.release_angle:g}",
        )

    raise SprayTestError(
        "Provide spray servo commands using either "
        "--spray-on-gcode/--spray-off-gcode, "
        "MAKEUP_SPRAY_ON_GCODE/MAKEUP_SPRAY_OFF_GCODE, "
        "or --servo-name plus --spray-angle and --release-angle."
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
    spray_on_gcode: str,
    spray_off_gcode: str,
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
        spray_off_gcode,
        f"G0 X{passes[0].x_mm:.3f} "
        f"Z{passes[0].z_start_mm:.3f} "
        f"F{travel_feed:.0f}",
        "M400",
    ]

    for index, spray_pass in enumerate(passes):
        commands.extend(
            [
                spray_on_gcode,
                f"G4 P{spray_settle_ms}",
                f"G1 Z{spray_pass.z_end_mm:.3f} "
                f"F{spray_feed:.0f}",
                "M400",
                spray_off_gcode,
                f"G4 P{release_settle_ms}",
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

    commands.extend([spray_off_gcode, "M400"])
    return commands


def execute_pattern(
    *,
    passes: list[SprayPass],
    spray_speed_mm_s: float,
    travel_speed_mm_s: float,
    spray_on_gcode: str,
    spray_off_gcode: str,
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
                spray_off_gcode,
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

            send_gcode(
                [
                    spray_on_gcode,
                    f"G4 P{spray_settle_ms}",
                    f"G1 Z{spray_pass.z_end_mm:.3f} "
                    f"F{spray_feed:.0f}",
                    "M400",
                    spray_off_gcode,
                    f"G4 P{release_settle_ms}",
                ],
                timeout_seconds=120.0,
            )

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
            send_gcode(
                [spray_off_gcode, "M400"],
                timeout_seconds=10.0,
            )
        except Exception as exc:
            print(
                "WARNING: failed to send final spray-off command: "
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
        "--spray-on-gcode",
        help=(
            "Exact Klipper G-code that presses/opens the airbrush "
            "servo. Can also use MAKEUP_SPRAY_ON_GCODE."
        ),
    )
    parser.add_argument(
        "--spray-off-gcode",
        help=(
            "Exact Klipper G-code that releases/closes the airbrush "
            "servo. Can also use MAKEUP_SPRAY_OFF_GCODE."
        ),
    )

    parser.add_argument(
        "--servo-name",
        help="Klipper [servo] name for SET_SERVO convenience mode.",
    )
    parser.add_argument(
        "--spray-angle",
        type=float,
        help="Servo angle that presses the spray trigger.",
    )
    parser.add_argument(
        "--release-angle",
        type=float,
        help="Servo angle that releases the spray trigger.",
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
        spray_on, spray_off = resolve_spray_commands(args)
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
            spray_on_gcode=spray_on,
            spray_off_gcode=spray_off,
            spray_settle_ms=args.spray_settle_ms,
            release_settle_ms=args.release_settle_ms,
        )

        print("\nSerpentine mannequin spray test")
        print("--------------------------------")
        print(f"Vertical passes: {len(passes)}")
        print(f"Actual X step-over: {actual_step:.2f} mm")
        print(f"Vertical spray speed: {args.spray_speed:.2f} mm/s")
        print(f"Spray-on command: {spray_on}")
        print(f"Spray-off command: {spray_off}")
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

        print("\nEXECUTING on mannequin. Ctrl-C requests spray-off in cleanup.")
        execute_pattern(
            passes=passes,
            spray_speed_mm_s=args.spray_speed,
            travel_speed_mm_s=args.travel_speed,
            spray_on_gcode=spray_on,
            spray_off_gcode=spray_off,
            spray_settle_ms=args.spray_settle_ms,
            release_settle_ms=args.release_settle_ms,
        )
        print("Pattern complete; spray servo released.")
        return 0

    except (SprayTestError, KeyboardInterrupt) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
