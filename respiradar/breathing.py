"""Breathing pipeline: IQ frames -> chest displacement -> breathing rate (+ alerts).

Each stage is a small replaceable function. The current versions are naive PLACEHOLDERS that
are just good enough for the dashboard to show something. Swap them for real algorithms.
Reference: acconeer/exptool/a121/algo/breathing/_processor.py (the Acconeer implementation).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from respiradar.sources import WAVELENGTH_M, Frame

HISTORY_S = 30.0  # displacement history kept for rate estimation and plotting
RATE_WINDOW_S = 20.0  # data used per rate estimate
MIN_BPM, MAX_BPM = 6.0, 40.0


def range_profile(iq: np.ndarray) -> np.ndarray:
    """Mean amplitude per distance point."""
    return np.abs(iq).mean(axis=0)


def select_range_bin(profile: np.ndarray, motion: np.ndarray) -> int:
    """PLACEHOLDER: choose the distance bin with the most slow motion.

    TODO: presence detection (acconeer.exptool.a121.algo.presence), track the person across
    bins, and combine several bins weighted by amplitude instead of picking just one.
    """
    return int(np.argmax(motion))


def estimate_rate_bpm(displacement: np.ndarray, fs: float) -> float | None:
    """PLACEHOLDER: breathing rate from the biggest FFT peak in the breathing band.

    TODO: bandpass filtering, peak interpolation, smoothing over time, confidence score.
    """
    n = int(RATE_WINDOW_S * fs)
    if len(displacement) < n:
        return None
    x = displacement[-n:] - displacement[-n:].mean()
    spectrum = np.abs(np.fft.rfft(x * np.hanning(n), n=4 * n))
    freqs_bpm = np.fft.rfftfreq(4 * n, 1 / fs) * 60
    band = (freqs_bpm >= MIN_BPM) & (freqs_bpm <= MAX_BPM)
    return float(freqs_bpm[band][np.argmax(spectrum[band])])


def detect_events(displacement: np.ndarray, rate_bpm: float | None) -> list[str]:
    """PLACEHOLDER: clinical alerts (apnea, bradypnea, tachypnea).

    TODO: apnea = no breathing motion for >10 s; also flag irregular breathing and
    person-absent (out of bed) so absence isn't mistaken for apnea.
    """
    events = []
    if rate_bpm is not None and rate_bpm < 8:
        events.append("low breathing rate")
    if rate_bpm is not None and rate_bpm > 30:
        events.append("high breathing rate")
    return events


class BreathingPipeline:
    def __init__(self, frame_rate: float):
        self.fs = frame_rate
        self.times: deque[float] = deque(maxlen=int(HISTORY_S * frame_rate))
        self.displacement_mm: deque[float] = deque(maxlen=int(HISTORY_S * frame_rate))
        self.static: np.ndarray | None = None  # slow average per bin, i.e. the static clutter
        self.motion: np.ndarray | None = None  # slow average of |non-static signal| per bin
        self.bin: int | None = None
        self.prev_phase: float | None = None
        self.phase_unwrapped = 0.0

    def process(self, frame: Frame) -> dict:
        mean_sweep = frame.iq.mean(axis=0)
        profile = range_profile(frame.iq)

        # Remove static reflections (walls, bed frame) so only moving things remain.
        if self.static is None:
            self.static = mean_sweep.copy()
            self.motion = np.zeros(len(mean_sweep))
        alpha = 1 / (10 * self.fs)  # ~10 s time constant
        self.static = (1 - alpha) * self.static + alpha * mean_sweep
        moving = mean_sweep - self.static
        self.motion = (1 - alpha) * self.motion + alpha * np.abs(moving)

        new_bin = select_range_bin(profile, self.motion)
        if new_bin != self.bin:
            self.bin, self.prev_phase = new_bin, None

        # Phase of the chest reflection -> displacement. 4*pi because the path is round-trip.
        phase = float(np.angle(moving[self.bin]))
        if self.prev_phase is not None:
            self.phase_unwrapped += (phase - self.prev_phase + np.pi) % (2 * np.pi) - np.pi
        self.prev_phase = phase
        self.times.append(frame.t)
        self.displacement_mm.append(self.phase_unwrapped * WAVELENGTH_M / (4 * np.pi) * 1000)

        displacement = np.asarray(self.displacement_mm)
        rate = estimate_rate_bpm(displacement, self.fs)
        return {
            "t": frame.t,
            "distances_m": frame.distances_m.tolist(),
            "range_profile": profile.tolist(),
            "target_m": float(frame.distances_m[self.bin]),
            "times": list(self.times),
            "displacement_mm": displacement.tolist(),
            "rate_bpm": rate,
            "events": detect_events(displacement, rate),
        }
