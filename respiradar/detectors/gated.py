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
from respiradar.detectors import changepoint, ensemble

INTRA = FEATURE_NAMES.index("intra")
INTER = FEATURE_NAMES.index("inter")

# Presence is decided on a SLOW statistic, not a per-frame one.
#
# The obvious test - is the presence score above a threshold right now - does not work, and
# a real recording of a wall is what proved it. Instantaneous slow-motion scores overlap
# badly: the wall spikes to 15.2 while a person holding their breath drops to 2.7. Per frame
# the two are not separable, because a perfectly still person and an empty room are very
# nearly the same observation for a motion sensor. Reflected amplitude does not help either
# (wall 21.0, person 18.8 - a wall is a strong reflector).
#
# What does separate them is time. Nobody holds still for a minute: heartbeat, sway and
# micro-motion keep accumulating. A wall does not. Over a 60 s trailing median the least
# active occupied minute of any recording scores 10.8 and the most active wall minute scores
# 8.1, which separates with every session on the right side.
PRESENCE_WINDOW_S = 60.0
PRESENCE_THRESHOLD = 9.5  # midway between wall 8.1 and occupied 10.8


class PresenceGatedDetector:
    name = "gated/cusum+presence"

    def __init__(self, inner=None, absent_s: float = 12.0, name: str | None = None) -> None:
        self.inner = inner or changepoint.build_conservative()
        if name:
            self.name = name
        self.absent_s = absent_s

    def fit(self, clips) -> None:
        if hasattr(self.inner, "fit"):
            self.inner.fit(clips)

    def _left_the_room(self, clip: Clip) -> np.ndarray:
        """True where nobody appears to be in front of the sensor."""
        fs = 1 / max(float(np.median(np.diff(clip.t))), 1e-6)
        window = int(PRESENCE_WINDOW_S * fs)
        inter = clip.X[:, INTER]

        # Trailing median, causal: each sample sees only its own past.
        #
        # Evaluated every STRIDE frames and held in between, rather than recomputed for each
        # one. Presence is a question about the last minute; resolving it to a twentieth of
        # a second is meaningless precision bought at 10x the cost. Holding the previous
        # value keeps it causal - a sample never sees anything newer than itself.
        stride = max(1, int(0.5 * fs))
        activity = np.empty(len(inter))
        last = 0.0
        for i in range(len(inter)):
            if i % stride == 0:
                last = float(np.median(inter[max(0, i - window + 1) : i + 1]))
            activity[i] = last

        # A partial window is used as-is rather than suppressed. An occupied scene reads high
        # from the first seconds, so there is no need to wait: blanket-suppressing the first
        # window swallows every hold that begins at 30 s, which cost five of thirteen.
        return activity < PRESENCE_THRESHOLD

    def predict(self, clip: Clip) -> np.ndarray:
        return self.inner.predict(clip) & ~self._left_the_room(clip)


def build():
    return PresenceGatedDetector()


def build_best():
    """The detector the live app runs: change-point charts plus the empty-room gate.

    Not simply the top of the bake-off table. `spectral` ties on score and has a slightly
    better median latency, but it resolves its features by looking them up per recorded
    session, so it cannot run on a live sensor at all without being rewritten. The
    change-point charts consume the streaming feature row directly, which is what a live
    detector has to do.

    `ensemble/fusion` is the only entry reaching 4/4 holds at zero false alarms, but like
    every other entry it was measured only on recordings containing a person, so ungated it
    alarms on an empty room.

    The full four-chart bank: 12/13 holds, zero false alarms across 23 minutes of negatives,
    median latency 18.0 s, leave-one-subject-out. The single miss is the 4.6 s hold that
    begins inside the filter warmup. Ungated it alarms on 44-54% of an empty room, which is
    what the gate is for.
    """
    return PresenceGatedDetector(inner=changepoint.build(), name="gated/best")
