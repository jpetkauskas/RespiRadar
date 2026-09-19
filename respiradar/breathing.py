"""Breathing pipeline: sparse IQ frames -> chest displacement -> breathing rate.

The chain, once a person has been located:

1. Average the sweeps in each frame to get one complex sample per range point.
2. Take the phase of that sample. A chest moving `d` metres towards the sensor shifts the
   phase by `4*pi*d/lambda` - 4*pi rather than 2*pi because the path is there and back.
3. Unwrap the phase over time and convert to millimetres. That is the displacement trace.
4. Combine the few range points around the person, weighted by how strong each one is.
5. Band-pass the trace to the plausible breathing band, take its PSD, and read off the peak.

Stage 4 is why this beats watching a single range bin: a chest is wider than one 6 cm range
point, so neighbouring points carry correlated copies of the same motion and averaging them
cancels independent noise.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import signal

from respiradar.presence import PresenceDetector, PresenceResult
from respiradar.sources import WAVELENGTH_M, Frame, RadarConfig

MM_PER_RADIAN = WAVELENGTH_M / (4 * np.pi) * 1000


class AppState(Enum):
    NO_PRESENCE = "No presence detected"
    DETERMINE_DISTANCE = "Determining distance"
    ESTIMATE_BREATHING_RATE = "Estimating breathing rate"


@dataclass
class BreathingResult:
    t: float
    app_state: AppState
    presence: PresenceResult
    distances_being_analyzed: tuple[int, int] | None = None
    times: np.ndarray = field(default_factory=lambda: np.empty(0))
    displacement_mm: np.ndarray = field(default_factory=lambda: np.empty(0))
    psd_freqs_hz: np.ndarray = field(default_factory=lambda: np.empty(0))
    psd: np.ndarray = field(default_factory=lambda: np.empty(0))
    rate_bpm: float | None = None
    rate_history: np.ndarray = field(default_factory=lambda: np.empty(0))
    rate_times: np.ndarray = field(default_factory=lambda: np.empty(0))
    delayed: bool = False


class BreathingPipeline:
    def __init__(
        self,
        config: RadarConfig,
        lowest_breathing_rate: float = 6.0,
        highest_breathing_rate: float = 60.0,
        time_series_length_s: float = 20.0,
        distance_determination_duration_s: float = 5.0,
        num_distances_to_analyze: int = 3,
        presence: PresenceDetector | None = None,
    ) -> None:
        self.config = config
        self.fs = config.frame_rate
        self.low_hz = lowest_breathing_rate / 60
        self.high_hz = highest_breathing_rate / 60
        self.num_distances = num_distances_to_analyze
        self.presence = presence or PresenceDetector(config)

        self.window_length = int(time_series_length_s * self.fs)
        self.determination_frames = int(distance_determination_duration_s * self.fs)

        self.app_state = AppState.NO_PRESENCE
        self.frames_with_presence = 0
        self.distances_being_analyzed: tuple[int, int] | None = None

        self.times: deque[float] = deque(maxlen=self.window_length)
        self.displacement: deque[float] = deque(maxlen=self.window_length)
        self.rate_history: deque[float] = deque(maxlen=self.window_length)
        self.rate_times: deque[float] = deque(maxlen=self.window_length)

        self.prev_angles: np.ndarray | None = None
        self.unwrapped: np.ndarray | None = None
        self.amplitude: np.ndarray | None = None
        self.rate_bpm: float | None = None

        # Band-pass for the displayed trace. Breathing is slow, so a low order is plenty.
        nyquist = self.fs / 2
        self.sos = signal.butter(
            2,
            [self.low_hz / nyquist, min(self.high_hz / nyquist, 0.99)],
            btype="bandpass",
            output="sos",
        )

    def _reset_tracking(self) -> None:
        self.times.clear()
        self.displacement.clear()
        self.rate_history.clear()
        self.rate_times.clear()
        self.prev_angles = None
        self.unwrapped = None
        self.amplitude = None
        self.rate_bpm = None

    def _select_distances(self, peak_index: int) -> tuple[int, int]:
        half = self.num_distances // 2
        last = self.config.num_points - 1
        low = int(np.clip(peak_index - half, 0, max(last - self.num_distances + 1, 0)))
        high = min(low + self.num_distances - 1, last)
        return low, high

    def _update_displacement(self, frame: Frame) -> None:
        assert self.distances_being_analyzed is not None
        low, high = self.distances_being_analyzed
        segment = frame.iq[:, low : high + 1].mean(axis=0)

        angles = np.angle(segment)
        amplitude = np.abs(segment)

        if self.prev_angles is None:
            self.unwrapped = np.zeros_like(angles)
            self.amplitude = amplitude
        else:
            step = (angles - self.prev_angles + np.pi) % (2 * np.pi) - np.pi
            self.unwrapped = self.unwrapped + step
            # Slowly track how strong each range point is, for the weighting below.
            self.amplitude = 0.95 * self.amplitude + 0.05 * amplitude
        self.prev_angles = angles

        # Weight by power: a range point reflecting twice as strongly carries four times the
        # weight, which is where its phase is correspondingly more trustworthy.
        weights = self.amplitude**2
        total = weights.sum()
        if total <= 0:
            combined = float(np.mean(self.unwrapped))
        else:
            combined = float(np.sum(self.unwrapped * weights) / total)

        self.times.append(frame.t)
        self.displacement.append(combined * MM_PER_RADIAN)

    def _spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(self.displacement, dtype=float)
        x = x - x.mean()
        x = signal.detrend(x)
        n = len(x)
        padded = 8 * n  # interpolate the peak by zero padding
        spectrum = np.abs(np.fft.rfft(x * np.hanning(n), n=padded)) ** 2
        freqs = np.fft.rfftfreq(padded, 1 / self.fs)
        return freqs, spectrum

    def _estimate_rate(self) -> float | None:
        freqs, spectrum = self._spectrum()
        band = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        if not band.any():
            return None
        in_band = spectrum[band]
        peak = float(freqs[band][int(np.argmax(in_band))]) * 60

        # Reject a peak that is not meaningfully above the rest of the band: that means we are
        # looking at noise, not a chest.
        if in_band.max() <= 4 * np.median(in_band):
            return None
        return peak

    def process(self, frame: Frame) -> BreathingResult:
        presence = self.presence.process(frame)

        if not presence.detected:
            self.app_state = AppState.NO_PRESENCE
            self.frames_with_presence = 0
            self.distances_being_analyzed = None
            self._reset_tracking()
            return BreathingResult(
                t=frame.t, app_state=self.app_state, presence=presence, delayed=frame.delayed
            )

        self.frames_with_presence += 1

        if self.frames_with_presence < self.determination_frames:
            self.app_state = AppState.DETERMINE_DISTANCE
            return BreathingResult(
                t=frame.t, app_state=self.app_state, presence=presence, delayed=frame.delayed
            )

        if self.distances_being_analyzed is None:
            self.distances_being_analyzed = self._select_distances(presence.peak_index)

        self.app_state = AppState.ESTIMATE_BREATHING_RATE
        self._update_displacement(frame)

        freqs = np.empty(0)
        psd = np.empty(0)
        if len(self.displacement) >= self.window_length:
            freqs, psd = self._spectrum()
            band = (freqs >= self.low_hz) & (freqs <= self.high_hz)
            freqs, psd = freqs[band], psd[band]
            rate = self._estimate_rate()
            if rate is not None:
                # Smooth: breathing rate cannot jump, so a sudden change is an artefact.
                self.rate_bpm = rate if self.rate_bpm is None else 0.9 * self.rate_bpm + 0.1 * rate
                self.rate_history.append(self.rate_bpm)
                self.rate_times.append(frame.t)

        return BreathingResult(
            t=frame.t,
            app_state=self.app_state,
            presence=presence,
            distances_being_analyzed=self.distances_being_analyzed,
            times=np.asarray(self.times),
            displacement_mm=self._filtered_displacement(),
            psd_freqs_hz=freqs,
            psd=psd,
            rate_bpm=self.rate_bpm,
            rate_history=np.asarray(self.rate_history),
            rate_times=np.asarray(self.rate_times),
            delayed=frame.delayed,
        )

    def _filtered_displacement(self) -> np.ndarray:
        x = np.asarray(self.displacement, dtype=float)
        if len(x) < 30:
            return x - x.mean() if len(x) else x
        return signal.sosfiltfilt(self.sos, x - x.mean())
