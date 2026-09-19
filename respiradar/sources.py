"""Frame sources: the real XM125 over serial, or a simulator for working without hardware.

Every source yields `Frame`s: a complex IQ array of shape (sweeps_per_frame, num_points)
plus the distance (m) of each point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

# Sensor settings, roughly the Acconeer breathing reference app defaults.
START_M = 0.3
END_M = 1.5
STEP_LENGTH = 24  # in units of ~2.5 mm, so 6 cm between points
FRAME_RATE_HZ = 20.0
SWEEPS_PER_FRAME = 16
HWAAS = 32

BASE_STEP_M = 0.0025
WAVELENGTH_M = 0.005  # 60 GHz


@dataclass
class Frame:
    t: float  # seconds since the source started
    iq: np.ndarray  # complex, (sweeps_per_frame, num_points)
    distances_m: np.ndarray  # (num_points,)


def _points() -> tuple[int, int]:
    start_point = int(START_M / BASE_STEP_M)
    num_points = int(np.ceil((END_M - START_M) / (STEP_LENGTH * BASE_STEP_M))) + 1
    return start_point, num_points


def _distances() -> np.ndarray:
    start_point, num_points = _points()
    return (start_point + STEP_LENGTH * np.arange(num_points)) * BASE_STEP_M


def radar_frames(serial_port: str) -> Iterator[Frame]:
    """Stream sparse IQ frames from an XM125 running Acconeer's Exploration Server firmware."""
    from acconeer.exptool import a121

    start_point, num_points = _points()
    config = a121.SensorConfig(
        start_point=start_point,
        num_points=num_points,
        step_length=STEP_LENGTH,
        profile=a121.Profile.PROFILE_3,
        hwaas=HWAAS,
        sweeps_per_frame=SWEEPS_PER_FRAME,
        frame_rate=FRAME_RATE_HZ,
    )
    distances = _distances()

    client = a121.Client.open(serial_port=serial_port)
    print(f"Connected: {client.server_info}")
    client.setup_session(config)
    client.start_session()
    t0 = time.monotonic()
    try:
        while True:
            result = client.get_next()
            yield Frame(t=time.monotonic() - t0, iq=result.frame, distances_m=distances)
    finally:
        client.stop_session()
        client.close()


def simulated_frames(breaths_per_min: float = 14.0, target_m: float = 0.8) -> Iterator[Frame]:
    """Fake a chest at `target_m` moving ~4 mm peak-to-peak, plus noise."""
    distances = _distances()
    rng = np.random.default_rng()
    envelope = np.exp(-(((distances - target_m) / 0.08) ** 2))
    t0 = time.monotonic()
    n = 0
    while True:
        t = n / FRAME_RATE_HZ
        displacement = 0.002 * np.sin(2 * np.pi * breaths_per_min / 60 * t)
        phase = 4 * np.pi * displacement / WAVELENGTH_M
        chest = 1000 * envelope * np.exp(1j * phase)
        static = 300 * np.exp(1j * distances * 50)  # static clutter
        noise = 20 * (
            rng.standard_normal((SWEEPS_PER_FRAME, len(distances)))
            + 1j * rng.standard_normal((SWEEPS_PER_FRAME, len(distances)))
        )
        yield Frame(t=t, iq=chest + static + noise, distances_m=distances)
        n += 1
        time.sleep(max(0.0, t0 + n / FRAME_RATE_HZ - time.monotonic()))
