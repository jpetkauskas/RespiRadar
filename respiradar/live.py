"""Run a bake-off detector on a live sensor.

The detectors are written against a `Clip`: a whole recording's worth of feature rows at
once. That is right for evaluation and wrong for a sensor, which produces one frame at a
time. Every detector is causal, though - `test_features_are_causal` and each entry's own
prefix-replay check pin that - so running one over a growing buffer and reading the last
value gives exactly the answer it would have given live.

Two practical concessions:

- The detector re-runs over the buffer rather than keeping incremental state, because the
  entries were not written to be resumable. That is O(n) per evaluation, so it runs every
  `interval_s` rather than every frame. An apnea alarm has ~20 s of latency by nature; half
  a second of evaluation granularity is not the bottleneck.
- The buffer is capped. CUSUM charts and trailing quantiles need history, but not unbounded
  history, and an all-night session would otherwise grow without limit.
"""

from __future__ import annotations

import numpy as np

from respiradar.bakeoff import Clip
from respiradar.dataset import SESSIONS, FeatureExtractor, load_cached
from respiradar.sources import Frame, RadarConfig

BUFFER_S = 300.0
EVALUATE_EVERY_S = 0.5


def fit_on_everything(detector):
    """Fit on every recording we have.

    Leave-one-subject-out exists to predict performance on a stranger. Once that number is
    known, the detector that actually runs should learn from all of it.
    """
    if not hasattr(detector, "fit"):
        return detector
    clips = []
    for session in SESSIONS:
        try:
            t, X, y = load_cached(session.name)
        except (KeyError, FileNotFoundError):
            continue
        clips.append(Clip(session.name, t, X, y, list(session.holds)))
    if clips:
        detector.fit(clips)
    return detector


class LiveDetector:
    """Feeds frames through the feature extractor and a detector, one at a time."""

    def __init__(self, config: RadarConfig, detector, buffer_s: float = BUFFER_S,
                 aux=("rhythm", "spectral")) -> None:
        self.config = config
        self.detector = fit_on_everything(detector)
        self.extractor = FeatureExtractor(config)
        self.max_frames = int(buffer_s * config.frame_rate)
        self.every = max(1, int(EVALUATE_EVERY_S * config.frame_rate))

        # Some detectors compute their own features from raw IQ and look them up by session
        # name - rhythm's band-fraction and spectral's range-STFT both do. Live there is no
        # session, so their extractors are run over the frame buffer and registered under
        # the name the live clip uses. A detector that needs neither is unaffected.
        self.aux = []
        for name in aux:
            try:
                module = __import__(f"respiradar.detectors.{name}", fromlist=["x"])
                if hasattr(module, "extract") and hasattr(module, "register"):
                    self.aux.append(module)
            except Exception:
                pass
        self.frames_buffer: list = []

        self.times: list[float] = []
        self.rows: list[np.ndarray] = []
        self.alarm = False
        self.alarms: list[bool] = []
        self._since_eval = 0

    def process(self, frame: Frame) -> np.ndarray:
        row = self.extractor.process(frame)
        self.times.append(frame.t)
        self.rows.append(row)
        if self.aux:
            self.frames_buffer.append(frame)
        if len(self.times) > self.max_frames:
            self.times.pop(0)
            self.rows.pop(0)
            self.alarms.pop(0)
            if self.frames_buffer:
                self.frames_buffer.pop(0)

        self._since_eval += 1
        if self._since_eval >= self.every and len(self.times) > 40:
            self._since_eval = 0
            self.alarm = self._evaluate()
        self.alarms.append(self.alarm)
        return row

    def _evaluate(self) -> bool:
        t = np.asarray(self.times)
        X = np.asarray(self.rows)
        clip = Clip("live", t, X, np.zeros(len(t), dtype=bool), [])
        for module in self.aux:
            try:
                at, aX = module.extract(self.frames_buffer, self.config)
                module.register("live", at, aX)
            except Exception:
                pass  # a detector that cannot supply its own features falls back
        try:
            return bool(np.asarray(self.detector.predict(clip))[-1])
        except Exception:
            # A detector that cannot run live must not take the dashboard down with it.
            return self.alarm
