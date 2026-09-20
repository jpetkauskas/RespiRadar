"""RespiRadar on the UNO Q: the Linux half.

Arduino App Lab starts this, `App.run` keeps the process alive and stops it cleanly, and
everything underneath is the ordinary `respiradar` pipeline - the same code the bake-off
scores. This file only picks a frame source and hands the loop over.

Configuration is by environment variable, because App Lab has no argv to pass:

    RESPIRADAR_PORT       serial port of the XM125 (default: autodetect, /dev/ttyUSB0)
    RESPIRADAR_BAUDRATE   default 230400
    RESPIRADAR_SIMULATE   set to 1 to run without a sensor, for testing the matrix
    RESPIRADAR_SESSION    replay a recording by name instead of reading the sensor
"""

import os
import sys
from pathlib import Path

# The app lives inside the repo, so the package is two directories up. Prepending rather than
# appending: if a stale `respiradar` is ever installed system-wide, the checkout should win.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arduino.app_utils import App  # noqa: E402

from respiradar.sources import RadarConfig, find_serial_port  # noqa: E402
from respiradar.unoq import BridgeSink, run  # noqa: E402


def frames_and_config():
    session = os.environ.get("RESPIRADAR_SESSION")
    if session:
        from respiradar.dataset import session_by_name
        from respiradar.sources import recorded_config, replay_frames

        recording = session_by_name(session)
        print(f"replaying {recording.name}")
        return replay_frames(recording.path, realtime=True), recorded_config(recording.path)

    if os.environ.get("RESPIRADAR_SIMULATE") == "1":
        from respiradar.sources import simulated_frames

        config = RadarConfig()
        print("simulated sensor: no hardware, a synthetic chest at 0.8 m")
        return simulated_frames(config), config

    from main import fit_config
    from respiradar.sources import radar_frames

    baudrate = int(os.environ.get("RESPIRADAR_BAUDRATE", "230400"))
    port = os.environ.get("RESPIRADAR_PORT") or find_serial_port()
    if not port:
        raise SystemExit(
            "No XM125 found. On Debian the CH340 appears as /dev/ttyUSB0 once the driver is "
            "loaded - see build_ch341.sh - and you must be in the dialout group. "
            "Set RESPIRADAR_SIMULATE=1 to run the display without a sensor."
        )
    config = fit_config(RadarConfig(), baudrate)
    print(f"XM125 on {port}: {config.frame_rate:.0f} Hz, "
          f"{config.sweeps_per_frame} sweeps x {config.num_points} bins")
    return radar_frames(port, config=config, baudrate=baudrate), config


def radar_loop():
    frames, config = frames_and_config()
    run(frames, config, BridgeSink())


App.run(radar_loop)
