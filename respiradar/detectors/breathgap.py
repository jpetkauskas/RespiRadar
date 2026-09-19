"""Alarm on the gap between breaths, not on how strongly the chest moves.

Every amplitude-based detector in this bake-off transfers badly between people: chest size,
posture and distance all change how many millimetres the radar sees, so a threshold tuned on
one body misreads another. The time between breaths does not have that problem. A 15 s gap is
15 s whoever is lying there, and it is also what a clinician means by apnea.

Two things make it work that did not work before:

- A narrower band. The shared features band-pass at 0.10-0.70 Hz, but 0.10 Hz is 6 bpm, below
  any real breathing, so slow drift dominates the waveform - the dominant period came out at
  6.6-10.4 bpm on most recordings, which is wander, not breath. At 0.18-0.55 Hz every session
  shows a plausible 10.8-17.8 bpm line.
- Breath depth that cannot shrink during a hold. The reference is updated only when a breath
  is actually accepted, so a long quiet stretch never lowers the bar to meet itself. A
  reference that decays would find "breaths" in the noise of an apnea.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal

from respiradar.bakeoff import Clip
from respiradar.dataset import DATA, SESSIONS, FeatureExtractor
from respiradar.sources import recorded_config, replay_frames

CACHE = DATA / "breathgap_waveform.npz"
LOW_HZ, HIGH_HZ = 0.18, 0.55  # 10.8-33 bpm
MIN_BREATH_INTERVAL_S = 1.5  # 40 bpm ceiling; anything faster is not a breath


def build_cache(path: Path = CACHE) -> Path:
    arrays = {}
    for session in SESSIONS:
        extractor = FeatureExtractor(recorded_config(session.path))
        times, raw = [], []
        for frame in replay_frames(session.path):
            extractor.process(frame)
            times.append(frame.t)
            raw.append(extractor.raw[-1])
        arrays[f"{session.name}__t"] = np.asarray(times)
        arrays[f"{session.name}__x"] = np.asarray(raw)
    np.savez_compressed(path, **arrays)
    return path


def _waveform(name: str) -> tuple[np.ndarray, np.ndarray]:
    if not CACHE.exists():
        build_cache()
    with np.load(CACHE) as data:
        return data[f"{name}__t"], data[f"{name}__x"]


class BreathGapDetector:
    name = "breathgap/time-since-breath"

    def __init__(self, gap_s: float = 11.0, depth_fraction: float = 0.30,
                 floor_mm: float = 0.08) -> None:
        self.gap_s = gap_s
        self.depth_fraction = depth_fraction
        self.floor_mm = floor_mm

    def fit(self, clips) -> None:
        """Nothing is learned. The threshold is a duration, which needs no calibration."""

    def _breath_gaps(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        fs = 1 / max(float(np.median(np.diff(t))), 1e-6)
        sos = signal.butter(
            2, [LOW_HZ / (fs / 2), HIGH_HZ / (fs / 2)], btype="bandpass", output="sos"
        )
        y = signal.sosfilt(sos, x - x[0])  # causal: sosfilt, never sosfiltfilt

        gaps = np.zeros(len(t))
        depths: list[float] = []
        last_breath_t = t[0]
        threshold = self.floor_mm

        for i in range(2, len(t)):
            # A breath is a local maximum in the band-limited chest motion, deep enough to be
            # a breath rather than noise, and not too soon after the previous one.
            is_peak = y[i - 1] > y[i] and y[i - 1] >= y[i - 2]
            if is_peak and y[i - 1] > threshold and (t[i - 1] - last_breath_t) >= MIN_BREATH_INTERVAL_S:
                last_breath_t = t[i - 1]
                depths.append(float(y[i - 1]))
                if len(depths) > 8:
                    depths.pop(0)
                # Updated ONLY here, so a hold cannot lower the bar to meet its own noise.
                threshold = max(self.floor_mm, self.depth_fraction * float(np.median(depths)))
            gaps[i] = t[i] - last_breath_t
        return gaps

    def predict(self, clip: Clip) -> np.ndarray:
        session = clip.name.split("[")[0]
        t_full, x_full = _waveform(session)
        # The clip may be a time slice; run from the start of the session so the detector has
        # the same history it would have live, then return only the requested span.
        gaps = self._breath_gaps(t_full, x_full)
        idx = np.searchsorted(t_full, clip.t)
        idx = np.clip(idx, 0, len(gaps) - 1)
        return gaps[idx] > self.gap_s


def build():
    return BreathGapDetector()


if __name__ == "__main__":
    print("wrote", build_cache())
