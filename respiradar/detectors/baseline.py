"""Baseline: breathing energy against this person's own baseline, with a motion gate.

The simplest thing that could work, and the bar every learned model has to beat. Alarm when
band-limited chest motion stays far below this person's normal for long enough, unless
something is obviously moving, which means they are awake and fine.
"""

from __future__ import annotations

import numpy as np

from respiradar.bakeoff import Clip
from respiradar.dataset import FEATURE_NAMES

RATIO = FEATURE_NAMES.index("ratio_4s")
INTRA = FEATURE_NAMES.index("intra")


class EnergyThresholdDetector:
    name = "baseline/energy-threshold"

    # Tuned by grid search over the two folds, picking the lowest worst-case latency among
    # the configurations with zero false alarms. The motion gate turns out to be nearly
    # irrelevant - removing it entirely scores the same - because talking and moving never
    # looked like apnea in the first place. It stays as cheap insurance for postures the
    # three recordings do not cover.
    def __init__(self, ratio_threshold: float = 0.70, quiet_s: float = 10.0,
                 intra_gate: float = 3.0) -> None:
        self.ratio_threshold = ratio_threshold
        self.quiet_s = quiet_s
        self.intra_gate = intra_gate

    def fit(self, clips) -> None:
        """Nothing to learn - the thresholds are fixed."""

    def predict(self, clip: Clip) -> np.ndarray:
        fs = 1 / max(float(np.median(np.diff(clip.t))), 1e-6)
        need = int(self.quiet_s * fs)

        quiet = (clip.X[:, RATIO] < self.ratio_threshold) & (clip.X[:, INTRA] < self.intra_gate)

        # Alarm once the signal has been quiet continuously for quiet_s.
        alarms = np.zeros(len(clip.t), dtype=bool)
        run = 0
        for i, q in enumerate(quiet):
            run = run + 1 if q else 0
            alarms[i] = run >= need
        return alarms


def build():
    return EnergyThresholdDetector()
