"""The winning detector, with a presence gate: never alarm about an empty room.

Every recording used to build this system has a person in it the whole time, so nothing in
the bake-off ever tested what happens when someone walks away. The answer turned out to be
bad: on a synthetic empty room the change-point detector alarms on 44-54% of frames. Nobody
there means no chest motion, which is exactly what apnea looks like.

The gate cannot be instantaneous. Presence detection drops out on ~8% of frames *during* a
real breath hold - someone holding still genuinely does resemble an empty room for a moment -
so gating frame-by-frame would suppress the alarms we most want. It takes a sustained absence
to conclude the person has left.

This is the honest weak point of the whole system: apnea and absence are the same observation,
separated only by how long the radar has seen nothing at all. The threshold below is set from
the only empty-room data available, which is synthetic. Recording two minutes of a genuinely
empty room would be the single most valuable thing to add.
"""

from __future__ import annotations

import numpy as np

from respiradar.bakeoff import Clip
from respiradar.dataset import FEATURE_NAMES
from respiradar.detectors import changepoint

INTRA = FEATURE_NAMES.index("intra")
INTER = FEATURE_NAMES.index("inter")

# PresenceDetector's own thresholds, which separate every real recording (91-100% present)
# from an empty room (0%) with a wide margin.
PRESENCE_THRESHOLD = 6.0


class PresenceGatedDetector:
    name = "gated/cusum+presence"

    def __init__(self, inner=None, absent_s: float = 12.0) -> None:
        self.inner = inner or changepoint.build_conservative()
        self.absent_s = absent_s

    def fit(self, clips) -> None:
        if hasattr(self.inner, "fit"):
            self.inner.fit(clips)

    def _left_the_room(self, clip: Clip) -> np.ndarray:
        fs = 1 / max(float(np.median(np.diff(clip.t))), 1e-6)
        need = int(self.absent_s * fs)

        present = (clip.X[:, INTRA] > PRESENCE_THRESHOLD) | (
            clip.X[:, INTER] > PRESENCE_THRESHOLD
        )
        gone = np.zeros(len(clip.t), dtype=bool)
        run = 0
        for i, p in enumerate(present):
            run = 0 if p else run + 1
            gone[i] = run >= need
        return gone

    def predict(self, clip: Clip) -> np.ndarray:
        return self.inner.predict(clip) & ~self._left_the_room(clip)


def build():
    return PresenceGatedDetector()
