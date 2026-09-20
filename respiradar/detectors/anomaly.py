"""Apnea detection as novelty detection: learn normal breathing, flag deviation from it.

The negative class here is ~17 minutes across three people; the positive class is four
events. So the model is fitted only on normal (non-hold) frames and a hold is whatever
does not look like them. Nothing is ever fitted on a breath hold.

What the bake-off taught us
---------------------------
Two results shaped this detector, and both are worth knowing before changing it.

1. Plain (symmetric) novelty detection is the wrong tool. IsolationForest, OneClassSVM,
   EllipticEnvelope and Mahalanobis all agree that the most abnormal frames in these
   recordings are talking and gross movement, not apnea. A breath hold sits *inside* the
   normal cloud on the quiet side - it is a low-energy state, and low-energy states are
   what a still, sleeping person produces all night. Scored symmetrically, the noisy
   session outranks both holds. The fix is to make the decision one-sided: only deviation
   in the direction of *less* breathing counts as anomalous. That is the `np.maximum(z, 0)`
   in `_score` and it is the single most important line in the file.

2. "Normal" is personal, not universal. Fitting on other people's absolute chest-motion
   features and applying the result to a stranger does not work - amplitude depends on
   body, posture and distance far more than on whether the person is breathing. What does
   transfer is a *drop relative to that person's own recent quiet breathing*. The supplied
   `baseline` feature is the repo's version of this, but it ratchets towards a subject's
   best breathing, so any shallow-breathing stretch sits below it forever and reads as
   apnea. A causal rolling low quantile of the last 90 s does not ratchet, re-references
   itself when the person's breathing genuinely changes, and measurably outperformed both
   `ratio_4s` and raw RMS across subjects.

So: a one-sided Mahalanobis novelty model, fitted on other people's normal frames, over a
feature that is already expressed in that person's own units.

Causality
---------
Every step is a function of the past only. The rolling quantile looks back, never forward;
the smoother is a trailing mean; the threshold is fitted in `fit()` on training clips and
never touched during `predict`. `predict` is written as an explicit forward loop so that
this is checkable by reading it rather than by trusting an array trick.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from respiradar.dataset import FEATURE_NAMES

RMS_4S = FEATURE_NAMES.index("rms_4s")
RMS_8S = FEATURE_NAMES.index("rms_8s")

LOG_FLOOR = 1e-6
WARMUP_S = 25.0  # matches evaluation.DEFAULT_WARMUP_S; features are transients before it


def _rolling_low_quantile(
    x: np.ndarray, fs: float, window_s: float, q: float, min_s: float, step: int
) -> np.ndarray:
    """Causal rolling q-quantile of `x` over the last `window_s` seconds.

    Sample i sees x[:i+1] and nothing else. The quantile is recomputed every `step`
    frames and held between recomputations (a zero-order hold, never an interpolation
    onto a future value). Before `min_s` of data exists the window is simply expanding.
    """
    n = len(x)
    win = int(window_s * fs)
    warm = int(min_s * fs)
    out = np.empty(n)
    held = x[0] if n else 0.0
    for i in range(n):
        if i % step == 0 or i == n - 1:
            lo = max(0, i - win + 1) if i >= warm else 0
            held = float(np.quantile(x[lo : i + 1 : max(1, step // 2)], q))
        out[i] = held
    return out


def _drop(X: np.ndarray, fs: float, window_s: float, q: float, min_s: float, step: int) -> np.ndarray:
    """How far below this person's own recent quiet breathing the last 4 s sit (in nepers)."""
    log_4s = np.log(np.maximum(X[:, RMS_4S], LOG_FLOOR))
    log_8s = np.log(np.maximum(X[:, RMS_8S], LOG_FLOOR))
    reference = _rolling_low_quantile(log_8s, fs, window_s, q, min_s, step)
    return reference - log_4s


def _trailing_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Mean of the last n samples, partial windows at the start. Causal."""
    if n <= 1:
        return x.astype(float)
    cumulative = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    i = np.arange(len(x))
    lo = np.maximum(i - n + 1, 0)
    return (cumulative[i + 1] - cumulative[lo]) / (i - lo + 1)


def _frame_rate(t: np.ndarray) -> float:
    if len(t) < 2:
        return 20.0
    dt = float(np.median(np.diff(t)))
    return 1.0 / dt if dt > 0 else 20.0


class AnomalyDetector:
    """One-sided Mahalanobis novelty model over person-relative breathing features."""

    name = "anomaly/one-sided-maha"

    def __init__(
        self,
        window_s: float = 90.0,
        quantile: float = 0.25,
        smooth_s: float = 3.0,
        threshold_quantile: float = 0.9999,
        margin: float = 0.90,
        dwell_s: float = 0.0,
        dwell_fraction: float = 1.0,
        reference_min_s: float = 30.0,
        quantile_step: int = 10,
    ) -> None:
        self.window_s = window_s
        self.quantile = quantile
        self.smooth_s = smooth_s
        self.threshold_quantile = threshold_quantile
        self.margin = margin
        self.dwell_s = dwell_s
        self.dwell_fraction = dwell_fraction
        self.reference_min_s = reference_min_s
        self.quantile_step = quantile_step

        self.mean_ = 0.0
        self.scale_ = 1.0
        self.threshold_ = np.inf

    # -- fitting -----------------------------------------------------------------

    def _features(self, X: np.ndarray, fs: float) -> np.ndarray:
        return _drop(X, fs, self.window_s, self.quantile, self.reference_min_s, self.quantile_step)

    def _score(self, drop: np.ndarray, fs: float) -> np.ndarray:
        """One-sided Mahalanobis distance from normal, then a trailing smoother.

        In one dimension the Mahalanobis distance is |z|; taking `max(z, 0)` keeps only
        the "breathing quieter than normal" half-space, which is what apnea looks like
        and what talking and movement do not.
        """
        z = np.maximum((drop - self.mean_) / self.scale_, 0.0)
        return _trailing_mean(z, max(1, int(self.smooth_s * fs)))

    def fit(self, clips: Sequence) -> None:
        normal: list[np.ndarray] = []
        for clip in clips:
            if len(clip.t) < 2:
                continue
            fs = _frame_rate(clip.t)
            drop = self._features(clip.X, fs)
            settled = clip.t >= clip.t[0] + WARMUP_S
            keep = settled & ~np.asarray(clip.y, dtype=bool)
            if keep.any():
                normal.append(drop[keep])
        if not normal:
            self.mean_, self.scale_, self.threshold_ = 0.0, 1.0, np.inf
            return

        pooled = np.concatenate(normal)
        self.mean_ = float(np.mean(pooled))
        self.scale_ = float(np.std(pooled)) or 1.0

        # Calibrate the alarm level on the same normal frames, after smoothing, so the
        # threshold is in the units the running detector actually sees. A high quantile
        # rather than the maximum keeps one freak training frame from deafening us; the
        # margin then allows for a body we have not met.
        smoothed: list[np.ndarray] = []
        for clip in clips:
            if len(clip.t) < 2:
                continue
            fs = _frame_rate(clip.t)
            score = self._score(self._features(clip.X, fs), fs)
            settled = clip.t >= clip.t[0] + WARMUP_S
            keep = settled & ~np.asarray(clip.y, dtype=bool)
            if keep.any():
                smoothed.append(score[keep])
        pooled_scores = np.concatenate(smoothed)
        level = float(np.quantile(pooled_scores, self.threshold_quantile))
        self.threshold_ = level * self.margin if level > 0 else np.inf

    # -- prediction --------------------------------------------------------------

    def predict(self, clip) -> np.ndarray:
        n = len(clip.t)
        if n == 0:
            return np.zeros(0, dtype=bool)
        fs = _frame_rate(clip.t)
        score = self._score(self._features(clip.X, fs), fs)
        over = score > self.threshold_

        dwell = int(self.dwell_s * fs)
        if dwell <= 1:
            return over

        # Sustained-duration requirement: the score must have been over threshold for
        # `dwell_fraction` of the last `dwell_s` seconds. Strictly backward-looking.
        alarms = np.zeros(n, dtype=bool)
        run = 0.0
        for i in range(n):
            run += float(over[i])
            if i >= dwell:
                run -= float(over[i - dwell])
            if i >= dwell - 1 and run >= self.dwell_fraction * dwell - 1e-9:
                alarms[i] = True
        return alarms


def build() -> AnomalyDetector:
    return AnomalyDetector()
