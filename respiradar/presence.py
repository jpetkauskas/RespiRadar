"""Presence detection: is someone there, and at what distance?

Two independent scores per range point, both normalised by the sensor's own noise so the
thresholds mean the same thing regardless of gain:

- *intra* frame: how much the sweeps inside a single frame disagree. Fast motion - an arm,
  someone walking past, rolling over in bed.
- *inter* frame: how much the frame-to-frame average drifts. Slow motion - the chest of a
  person sitting still, which is what we actually want to find.

Breathing lives in the inter score, so that is what picks the distance to analyse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from respiradar.sources import Frame, RadarConfig


def _alpha(time_constant_s: float, frame_rate: float) -> float:
    """Exponential smoothing coefficient for a given time constant."""
    if time_constant_s <= 0:
        return 0.0
    return float(np.exp(-1 / (time_constant_s * frame_rate)))


@dataclass
class PresenceResult:
    intra: np.ndarray  # fast-motion score per range point
    inter: np.ndarray  # slow-motion score per range point
    score: np.ndarray  # the larger of the two, per range point
    detected: bool
    peak_index: int
    peak_distance_m: float
    distances_m: np.ndarray


class PresenceDetector:
    def __init__(
        self,
        config: RadarConfig,
        intra_threshold: float = 6.0,
        inter_threshold: float = 6.0,
        intra_time_const_s: float = 0.15,
        inter_fast_cutoff_hz: float = 20.0,
        inter_slow_cutoff_hz: float = 0.2,
        inter_deviation_time_const_s: float = 0.5,
    ) -> None:
        self.config = config
        self.intra_threshold = intra_threshold
        self.inter_threshold = inter_threshold

        fs = config.frame_rate
        self.a_intra = _alpha(intra_time_const_s, fs)
        self.a_fast = _alpha(1 / (2 * np.pi * inter_fast_cutoff_hz), fs)
        self.a_slow = _alpha(1 / (2 * np.pi * inter_slow_cutoff_hz), fs)
        self.a_dev = _alpha(inter_deviation_time_const_s, fs)
        self.a_noise = _alpha(5.0, fs)

        self.noise: np.ndarray | None = None
        self.fast: np.ndarray | None = None
        self.slow: np.ndarray | None = None
        self.intra: np.ndarray | None = None
        self.inter: np.ndarray | None = None

    def process(self, frame: Frame) -> PresenceResult:
        sweeps = frame.iq
        mean_sweep = sweeps.mean(axis=0)

        # Sweep-to-sweep difference is dominated by noise: the chest cannot move meaningfully
        # in the ~1 ms between sweeps. Divide by sqrt(2) because a difference of two
        # independent samples has twice the variance.
        if sweeps.shape[0] > 1:
            noise_now = np.abs(np.diff(sweeps, axis=0)).mean(axis=0) / np.sqrt(2)
        else:
            noise_now = np.abs(mean_sweep) * 0 + 1.0
        noise_now = np.maximum(noise_now, 1e-9)

        if self.noise is None:
            self.noise = noise_now
            self.fast = mean_sweep.copy()
            self.slow = mean_sweep.copy()
            self.intra = np.zeros_like(noise_now)
            self.inter = np.zeros_like(noise_now)
        self.noise = self.a_noise * self.noise + (1 - self.a_noise) * noise_now

        # Intra: spread of the sweeps about their own mean, in units of noise.
        intra_now = np.abs(sweeps - mean_sweep).mean(axis=0) / self.noise
        self.intra = self.a_intra * self.intra + (1 - self.a_intra) * intra_now

        # Inter: a fast and a slow view of the same signal. Anything moving slowly shows up
        # as a gap between them; a static wall does not.
        self.fast = self.a_fast * self.fast + (1 - self.a_fast) * mean_sweep
        self.slow = self.a_slow * self.slow + (1 - self.a_slow) * mean_sweep
        # Averaging the sweeps already cut the noise by sqrt(sweeps_per_frame).
        inter_now = np.abs(self.fast - self.slow) / (
            self.noise / np.sqrt(sweeps.shape[0])
        )
        self.inter = self.a_dev * self.inter + (1 - self.a_dev) * inter_now

        score = np.maximum(self.intra, self.inter)
        detected = bool(
            self.intra.max() > self.intra_threshold or self.inter.max() > self.inter_threshold
        )
        # Locate on the slow-motion score: breathing, not an arm waving.
        peak = int(np.argmax(self.inter))

        return PresenceResult(
            intra=self.intra.copy(),
            inter=self.inter.copy(),
            score=score,
            detected=detected,
            peak_index=peak,
            peak_distance_m=float(frame.distances_m[peak]),
            distances_m=frame.distances_m,
        )
