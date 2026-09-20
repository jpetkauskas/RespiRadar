"""Frame sources: the XM125 over serial, a recorded file, or a simulator.

Every source yields `Frame`s, so the pipeline and the UI never know which one is running.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

BASE_STEP_M = 0.0025  # A121 range quantum, ~2.5 mm
WAVELENGTH_M = 0.005  # 60 GHz
BYTES_PER_POINT = 4  # complex int16 on the wire
UART_BITS_PER_BYTE = 10  # 8N1


@dataclass(frozen=True)
class RadarConfig:
    """Sensor settings. The defaults match Acconeer's breathing reference app."""

    start_m: float = 0.3
    end_m: float = 1.5
    step_length: int = 24  # 24 * 2.5 mm = 6 cm between range points
    frame_rate: float = 20.0
    sweeps_per_frame: int = 16
    hwaas: int = 32
    profile: int = 3

    @property
    def start_point(self) -> int:
        return int(self.start_m / BASE_STEP_M)

    @property
    def num_points(self) -> int:
        # Floor, not ceil: a partial range point does not exist, and rounding up would push
        # the last point past end_m. The epsilon absorbs float error when the span divides
        # exactly (1.2 / 0.06 lands on 20.000000000000004 as often as 19.999999999999996).
        span = self.end_m - self.start_m
        steps = span / (self.step_length * BASE_STEP_M)
        return int(np.floor(steps + 1e-6)) + 1

    @property
    def distances_m(self) -> np.ndarray:
        return (self.start_point + self.step_length * np.arange(self.num_points)) * BASE_STEP_M

    @property
    def bits_per_second(self) -> float:
        """Raw IQ throughput this config demands of the UART."""
        per_frame = self.num_points * self.sweeps_per_frame * BYTES_PER_POINT
        return per_frame * UART_BITS_PER_BYTE * self.frame_rate

    def fits_in(self, baudrate: int) -> bool:
        """Leave 20% headroom for protocol overhead."""
        return self.bits_per_second <= 0.8 * baudrate


# Legacy module-level view of the default config. `ml_pipeline.py` reads these, so they stay.
# Prefer a RadarConfig instance in new code: these settings are per-session, since main.py
# lowers the frame rate and sweep count to fit the available baud rate.
_DEFAULT_CONFIG = RadarConfig()
START_M = _DEFAULT_CONFIG.start_m
END_M = _DEFAULT_CONFIG.end_m
STEP_LENGTH = _DEFAULT_CONFIG.step_length
FRAME_RATE_HZ = _DEFAULT_CONFIG.frame_rate
SWEEPS_PER_FRAME = _DEFAULT_CONFIG.sweeps_per_frame
HWAAS = _DEFAULT_CONFIG.hwaas


def _points() -> tuple[int, int]:
    """(start_point, num_points) for the default range. Kept for `ml_pipeline.py`."""
    return _DEFAULT_CONFIG.start_point, _DEFAULT_CONFIG.num_points


@dataclass
class Frame:
    t: float  # seconds since the source started
    iq: np.ndarray  # complex, (sweeps_per_frame, num_points)
    distances_m: np.ndarray  # (num_points,)
    delayed: bool = False  # sensor could not keep up with the requested frame rate


def _config_from_record(record) -> tuple[RadarConfig, np.ndarray]:
    sc = record.session_config.sensor_config
    distances = (sc.start_point + sc.step_length * np.arange(sc.num_points)) * BASE_STEP_M
    config = RadarConfig(
        start_m=sc.start_point * BASE_STEP_M,
        end_m=distances[-1],
        step_length=sc.step_length,
        frame_rate=sc.frame_rate or 20.0,
        sweeps_per_frame=sc.sweeps_per_frame,
        hwaas=sc.hwaas,
    )
    return config, distances


def recorded_config(path: Path | str) -> RadarConfig:
    from acconeer.exptool import a121

    config, _ = _config_from_record(a121.load_record(str(path)))
    return config


def replay_frames(path: Path | str, realtime: bool = False) -> Iterator[Frame]:
    """Replay a recorded .h5 session. Used for tests and for demoing without hardware."""
    from acconeer.exptool import a121

    record = a121.load_record(str(path))
    config, distances = _config_from_record(record)
    period = 1 / config.frame_rate
    wall_start = time.monotonic()

    for n, frame in enumerate(record.frames):
        t = n * period
        if realtime:
            time.sleep(max(0.0, wall_start + t - time.monotonic()))
        yield Frame(t=t, iq=np.asarray(frame), distances_m=distances)


def find_serial_port() -> str | None:
    """Guess the XM125's port. On macOS the CH34x driver creates /dev/cu.wchusbserial*."""
    from serial.tools import list_ports

    candidates = []
    for port in list_ports.comports():
        name = port.device
        if any(tag in name for tag in ("wchusbserial", "usbserial", "usbmodem", "SLAB")):
            candidates.append(name)
        elif port.vid == 0x1A86:  # WCH CH340/CH341
            candidates.append(name)
    # Prefer /dev/cu.* over /dev/tty.*: tty blocks waiting for carrier detect.
    candidates.sort(key=lambda n: ("/tty." in n, n))
    return candidates[0] if candidates else None


def radar_frames(
    port: str,
    config: RadarConfig | None = None,
    baudrate: int | None = None,
    flow_control: bool = True,
    record_to: Path | str | None = None,
) -> Iterator[Frame]:
    """Stream sparse IQ from an XM125 running Acconeer's Exploration Server firmware.

    `record_to` also saves every frame, untouched, to an Acconeer .h5 file (the same format
    `replay_frames` reads). Pass `port="mock"` for Acconeer's hardware-free mock server.
    """
    from acconeer.exptool import a121

    config = config or RadarConfig()
    sensor_config = a121.SensorConfig(
        start_point=config.start_point,
        num_points=config.num_points,
        step_length=config.step_length,
        profile=a121.Profile(config.profile),
        hwaas=config.hwaas,
        sweeps_per_frame=config.sweeps_per_frame,
        frame_rate=config.frame_rate,
    )
    distances = config.distances_m

    if port == "mock":
        client = a121.Client.open(mock=True)
    else:
        client = a121.Client.open(
            serial_port=port,
            override_baudrate=baudrate,
            flow_control=flow_control,
        )
    print(f"Connected: {client.server_info}")
    client.setup_session(sensor_config)
    # The recorder writes to disk as it goes, so a crash loses at most the last second.
    recorder = a121.H5Recorder(str(record_to), client) if record_to else None
    client.start_session()
    t0 = time.monotonic()
    try:
        while True:
            result = client.get_next()
            yield Frame(
                t=time.monotonic() - t0,
                iq=result.frame,
                distances_m=distances,
                delayed=bool(result.frame_delayed),
            )
    finally:
        client.stop_session()
        if recorder is not None:
            recorder.close()
        client.close()


def simulated_frames(
    config: RadarConfig | None = None,
    breaths_per_min: float | None = 14.0,
    target_m: float = 0.8,
    realtime: bool = True,
    holds: tuple[tuple[float, float], ...] = (),
) -> Iterator[Frame]:
    """Fake a chest at `target_m` moving ~4 mm peak-to-peak, plus static clutter and noise.

    Pass `breaths_per_min=None` for an empty room: clutter and noise, nobody there.
    `holds` is a list of (start_s, duration_s) breath-holds: the person stays in place but
    the chest stops moving, as in an apnea.
    """
    config = config or RadarConfig()
    distances = config.distances_m
    rng = np.random.default_rng()
    envelope = np.exp(-(((distances - target_m) / 0.08) ** 2))
    t0 = time.monotonic()
    n = 0
    while True:
        t = n / config.frame_rate
        if breaths_per_min is None:
            chest = np.zeros_like(distances, dtype=complex)
        else:
            breathing_t = t
            for start, duration in holds:
                if start <= t < start + duration:
                    breathing_t = start  # chest frozen where it was when the hold began
            displacement = 0.002 * np.sin(2 * np.pi * breaths_per_min / 60 * breathing_t)
            phase = 4 * np.pi * displacement / WAVELENGTH_M
            chest = 1000 * envelope * np.exp(1j * phase)
        static = 300 * np.exp(1j * distances * 50)
        noise = 20 * (
            rng.standard_normal((config.sweeps_per_frame, len(distances)))
            + 1j * rng.standard_normal((config.sweeps_per_frame, len(distances)))
        )
        yield Frame(t=t, iq=chest + static + noise, distances_m=distances)
        n += 1
        if realtime:
            time.sleep(max(0.0, t0 + n / config.frame_rate - time.monotonic()))
