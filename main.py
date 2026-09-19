#!/usr/bin/env python3
"""RespiRadar - run me.

    python main.py                      # find the sensor, or fall back to the simulator
    python main.py --port /dev/cu.xxx   # a specific port
    python main.py --simulate           # no hardware
    python main.py --replay FILE.h5     # a recorded session

On macOS the SparkFun XM125 needs WCH's CH34x driver installed and a reboot before any
port shows up. See the README.
"""

from __future__ import annotations

import argparse
import sys
from functools import partial

from respiradar import gui
from respiradar.sources import (
    RadarConfig,
    find_serial_port,
    radar_frames,
    recorded_config,
    replay_frames,
    simulated_frames,
)

# The rate this board has proven stable at. Drop to 115200 if the link misbehaves.
DEFAULT_BAUDRATE = 230400


def list_ports() -> int:
    from serial.tools import list_ports as lp

    ports = list(lp.comports())
    if not ports:
        print("No serial ports found.")
        return 1
    for port in ports:
        print(f"  {port.device:<28} {port.description}")
    return 0


def fit_config(config: RadarConfig, baudrate: int) -> RadarConfig:
    """Back the frame rate off until the IQ stream fits in the available bandwidth."""
    if config.fits_in(baudrate):
        return config

    original = config
    for frame_rate in (20.0, 15.0, 10.0, 5.0):
        for sweeps in (16, 8, 4):
            candidate = RadarConfig(
                start_m=config.start_m,
                end_m=config.end_m,
                step_length=config.step_length,
                frame_rate=frame_rate,
                sweeps_per_frame=sweeps,
                hwaas=config.hwaas,
                profile=config.profile,
            )
            if candidate.fits_in(baudrate):
                print(
                    f"Config needs {original.bits_per_second/1000:.0f} kbit/s but the link is "
                    f"{baudrate/1000:.0f} kbaud.\n"
                    f"Backing off to {frame_rate:.0f} Hz / {sweeps} sweeps per frame "
                    f"({candidate.bits_per_second/1000:.0f} kbit/s)."
                )
                return candidate
    print(
        f"Warning: even the lightest config will not fit in {baudrate} baud. "
        "Expect delayed frames.",
        file=sys.stderr,
    )
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", help="serial port, e.g. /dev/cu.wchusbserial1420")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--no-flow-control", action="store_true", help="try this if connecting hangs")
    parser.add_argument("--simulate", action="store_true", help="synthetic data, no hardware")
    parser.add_argument("--bpm", type=float, default=14.0, help="simulated breathing rate")
    parser.add_argument("--replay", help="replay a recorded .h5 session")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--frame-rate", type=float)
    parser.add_argument("--sweeps", type=int)
    args = parser.parse_args()

    if args.list_ports:
        return list_ports()

    if args.replay:
        config = recorded_config(args.replay)
        frames = partial(replay_frames, args.replay, realtime=True)
        return gui.run(frames, config, f"replay: {args.replay}")

    overrides = {}
    if args.frame_rate:
        overrides["frame_rate"] = args.frame_rate
    if args.sweeps:
        overrides["sweeps_per_frame"] = args.sweeps
    config = RadarConfig(**overrides)

    port = None if args.simulate else (args.port or find_serial_port())

    if port is None:
        if not args.simulate:
            print("No radar found - running the simulator instead.")
            print("Plug in the XM125, or run `python main.py --list-ports` to look for it.\n")
        frames = partial(simulated_frames, config, breaths_per_min=args.bpm, realtime=True)
        return gui.run(frames, config, f"simulator @ {args.bpm:.0f} bpm")

    config = fit_config(config, args.baudrate)
    print(f"Connecting to {port} at {args.baudrate} baud...")
    frames = partial(
        radar_frames,
        port,
        config=config,
        baudrate=args.baudrate,
        flow_control=not args.no_flow_control,
    )
    return gui.run(frames, config, port)


if __name__ == "__main__":
    sys.exit(main())
