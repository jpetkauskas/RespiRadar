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

import threading

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

    def __init__(
        self,
        config: RadarConfig,
        detector,
        buffer_s: float = BUFFER_S,
        evaluate_every_s: float | None = EVALUATE_EVERY_S,
        fit_now: bool = True,
    ) -> None:
        self.config = config
        # The extractor is cheap and needs nothing on disk. Fitting is the slow half: it
        # reads the feature cache, and building that cache from the recordings takes minutes
        # on a Cortex-A53. `fit_now=False` lets a caller start consuming frames immediately
        # and fit on another thread - which is not a nicety on a real sensor, because a
        # process that is not calling `get_next()` is a process backing frames up on the wire.
        self.extractor = FeatureExtractor(config)
        self.detector = detector
        self.fitted = False
        if fit_now:
            self.fit()
        self.max_frames = int(buffer_s * config.frame_rate)
        # `None` means never evaluate from `process`. Re-running a detector over the buffer
        # costs O(n): measured on this corpus, a 300 s buffer takes ~1.9 s per evaluation on
        # a laptop, against the 0.5 s cadence asked of it. A caller that cannot afford to
        # block - anything driving a display - takes `None` and calls `evaluate()` on its own
        # thread instead.
        self.every = None if evaluate_every_s is None else max(
            1, int(evaluate_every_s * config.frame_rate)
        )

        self.times: list[float] = []
        self.rows: list[np.ndarray] = []
        self.alarm = False
        self.alarms: list[bool] = []
        self._since_eval = 0

    def process(self, frame: Frame) -> np.ndarray:
        row = self.extractor.process(frame)
        self.times.append(frame.t)
        self.rows.append(row)
        if len(self.times) > self.max_frames:
            self.times.pop(0)
            self.rows.pop(0)
            self.alarms.pop(0)

        self._since_eval += 1
        if self.every is not None and self._since_eval >= self.every and len(self.times) > 40:
            self._since_eval = 0
            self.alarm = self._evaluate()
        self.alarms.append(self.alarm)
        return row

    def fit(self) -> None:
        """Fit the detector on every recording. Reads the feature cache, so it is the slow one."""
        self.detector = fit_on_everything(self.detector)
        self.fitted = True

    def evaluate(self, snapshot=None) -> bool:
        """Run the detector and return its current verdict.

        `snapshot` is an optional (times, rows) pair taken by the caller, so a background
        thread can copy the buffer under its own lock and then spend the O(n) prediction
        without holding anything the frame loop needs.
        """
        return self._evaluate(snapshot)

    def _evaluate(self, snapshot=None) -> bool:
        # An unfitted detector has no opinion worth having. Returning the current state means
        # "no alarm yet" rather than a verdict from a detector that has not seen the data.
        if not self.fitted:
            return self.alarm
        times, rows = snapshot if snapshot is not None else (self.times, self.rows)
        if len(times) <= 40:
            return self.alarm
        t = np.asarray(times)
        X = np.asarray(rows)
        clip = Clip("live", t, X, np.zeros(len(t), dtype=bool), [])
        try:
            return bool(np.asarray(self.detector.predict(clip))[-1])
        except Exception:
            # A detector that cannot run live must not take the dashboard down with it.
            return self.alarm


class AlarmWorker:
    """Runs the detector on its own thread, so a slow verdict never stalls the display.

    The detectors are written against a whole `Clip` and re-run from scratch over the live
    buffer, which costs O(n). Measured on this corpus: 49 ms over a 30 s buffer, 373 ms over
    60 s, 1.9 s over the 300 s `live.BUFFER_S` default - against a 0.5 s cadence, on a laptop.
    The UNO Q's A53s are several times slower again, so evaluating inline would drop the frame
    rate to whatever the detector managed, and a frozen matrix is worse than a late alarm.

    So: the frame loop only extracts features (0.63 ms/frame) and draws, and this thread loops
    over the buffer as fast as it can, publishing a single bool. An apnea alarm has ~20 s of
    latency by nature - a verdict that is a couple of seconds stale changes nothing, and the
    wave stays at full frame rate, which is the part a person actually watches.
    """

    def __init__(self, live: LiveDetector) -> None:
        self.live = live
        self.alarm = False
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self.lock:
                snapshot = (list(self.live.times), list(self.live.rows))
            if len(snapshot[0]) <= 40:
                self._stop.wait(0.2)
                continue
            try:
                # Outside the lock: this is the expensive part, and the frame loop must not
                # wait on it.
                self.alarm = bool(self.live.evaluate(snapshot))
            except Exception:
                # A detector that cannot run live must not take the display down with it.
                # Keep the last verdict and try again.
                pass
            self._stop.wait(0.1)

    def stop(self) -> None:
        self._stop.set()
