"""Apnea detection as novelty detection: learn normal breathing, flag deviation from it.

The negative class is ~23 minutes across three people; the positive class is thirteen
events. So the model is fitted only on normal (non-hold) frames and a hold is whatever
does not look like them. Nothing is ever fitted on a breath hold.

Three findings shaped this file
-------------------------------
1. **Symmetric novelty detection is the wrong tool.** IsolationForest, OneClassSVM,
   EllipticEnvelope and Mahalanobis all agree that the most abnormal frames in these
   recordings are talking and gross movement, not apnea. A breath hold sits *inside* the
   normal cloud, on the quiet side - it is a low-energy state, and low-energy states are
   what a still, sleeping person produces all night. Scored symmetrically the noisy
   session outranks the holds. Every deviation here is therefore clipped to the
   "quieter than normal" half-space (`np.maximum(..., 0.0)` in `_deviations`). That one
   change is what makes the approach work at all.

2. **"Normal" is personal, not universal.** Absolute chest-motion amplitude depends on
   body, posture and distance far more than on whether someone is breathing, so a model
   fitted on other people's absolute features and applied to a stranger does not
   transfer. What transfers is a drop relative to *that person's own recent quiet
   breathing*. This detector's earlier version had to build that reference itself; the
   shared features now supply it as `ratio_4s_q` / `ratio_8s_q`, and those are the
   primary inputs here.

3. **Almost every false alarm is recovery breathing.** Listing the sustained low-energy
   excursions across all eleven sessions, every single one in a hold session lands in the
   seconds *immediately after a hold ends* - 39-46 s after a hold ending at 33 s, 114-121 s
   after one ending at 114 s, and so on. The pure negative sessions (sleeping, talking,
   noisy) contain essentially none. Breathing does not resume cleanly after a hold: it
   overshoots and then goes briefly quiet, and that dip looks exactly like the apnea that
   preceded it. `_refractory` therefore suppresses alarms for a while after an alarm
   clears, which is both physiologically honest and worth several holds of recall,
   because it lets the threshold drop without buying the recovery dips.

Causality
---------
Every step looks backwards only. Feature statistics and the alarm threshold are fitted in
`fit()` on training clips and never touched during `predict`. The smoother is a trailing
mean, the dwell test is a trailing count, and the refractory is a forward scan that can
only ever consult frames it has already passed. Verified by checking that
`predict(clip)[:k] == predict(clip[:k])` and that perturbing the future leaves past
alarms unchanged.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from respiradar.dataset import FEATURE_NAMES

# A hold makes each of these fall. `ratio_*_q` are the person-relative ones and carry most
# of the signal; `disp_std_4s` and `inter` add holds the ratios alone miss, because they
# see the broadband stillness rather than just the breathing band.
DEVIATION_FEATURES = ("ratio_4s_q", "ratio_8s_q", "disp_std_4s", "inter")

LOG_FLOOR = 1e-6
WARMUP_S = 25.0  # matches evaluation.DEFAULT_WARMUP_S; features are transients before it


def _log_features(X: np.ndarray, columns: Sequence[int]) -> np.ndarray:
    """Log of the selected columns. Log because these are scale-like quantities: a halving
    is a halving whether the person reflects strongly or weakly."""
    return np.log(np.maximum(X[:, list(columns)], LOG_FLOOR))


def _trailing_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Mean of the last n samples, partial windows at the start. Causal."""
    if n <= 1:
        return np.asarray(x, dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    i = np.arange(len(x))
    lo = np.maximum(i - n + 1, 0)
    return (cumulative[i + 1] - cumulative[lo]) / (i - lo + 1)


def _frame_rate(t: np.ndarray) -> float:
    """Frame rate, rounded so that a prefix of a clip yields exactly the same window
    lengths as the whole clip. Without the rounding, the median sample interval differs
    in its last floating-point digit and a window occasionally comes out one sample
    shorter, which looks like a causality violation and is really just arithmetic."""
    if len(t) < 2:
        return 20.0
    dt = float(np.median(np.diff(t)))
    return round(1.0 / dt, 6) if dt > 0 else 20.0


def _refractory(alarms: np.ndarray, n: int) -> np.ndarray:
    """Suppress alarms for n frames after an alarm run ends.

    Breathing is transiently irregular for a few tens of seconds after a hold; without
    this the recovery dip is indistinguishable from a second apnea.
    """
    out = np.zeros(len(alarms), dtype=bool)
    blocked_until = -1
    for i in range(len(alarms)):
        if alarms[i]:
            if i > blocked_until:
                out[i] = True
        elif i > 0 and out[i - 1]:
            blocked_until = i + n
    return out


class AnomalyDetector:
    """One-sided novelty model over person-relative breathing features."""

    name = "anomaly/one-sided"

    def __init__(
        self,
        features: Sequence[str] = DEVIATION_FEATURES,
        smooth_s: float = 6.0,
        dwell_s: float = 3.0,
        dwell_fraction: float = 0.9,
        refractory_s: float = 45.0,
        threshold_quantile: float = 0.99,
        margin: float = 1.1,
        recovery_s: float = 45.0,
    ) -> None:
        self.features = list(features)
        self.columns = [FEATURE_NAMES.index(n) for n in self.features]
        self.smooth_s = smooth_s
        self.dwell_s = dwell_s
        self.dwell_fraction = dwell_fraction
        self.refractory_s = refractory_s
        self.threshold_quantile = threshold_quantile
        self.margin = margin
        self.recovery_s = recovery_s

        self.mean_ = np.zeros(len(self.columns))
        self.scale_ = np.ones(len(self.columns))
        self.threshold_ = np.inf

    # -- scoring -----------------------------------------------------------------

    def _deviations(self, X: np.ndarray) -> np.ndarray:
        """Per-feature standardised shortfall below normal, clipped at zero.

        Only the "less breathing than normal" direction counts. Talking and movement push
        these features *up*, which is why an unclipped score flags the noisy session.
        """
        logged = _log_features(X, self.columns)
        return np.maximum((self.mean_ - logged) / self.scale_, 0.0)

    def _score(self, X: np.ndarray, fs: float) -> np.ndarray:
        combined = self._deviations(X).mean(axis=1)
        return _trailing_mean(combined, max(1, int(round(self.smooth_s * fs))))

    def _normal_mask(self, clip) -> np.ndarray:
        """Frames that are genuinely representative of normal breathing.

        Settled, not inside a hold, and - crucially - not in the recovery window just
        after one. Those recovery dips are as deep as the holds themselves, so leaving
        them in the calibration set drags the threshold up above the events we are
        trying to catch. They are not normal breathing; they are the aftermath of an
        apnea, and at run time `_refractory` is suppressing that same window anyway.
        """
        settled = clip.t >= clip.t[0] + WARMUP_S
        in_hold = np.asarray(clip.y, dtype=bool)
        recovering = np.zeros(len(clip.t), dtype=bool)
        for hold in getattr(clip, "holds", []):
            recovering |= (clip.t >= hold.end_s) & (clip.t < hold.end_s + self.recovery_s)
        return settled & ~in_hold & ~recovering

    # -- fitting -----------------------------------------------------------------

    def fit(self, clips: Sequence) -> None:
        normal = []
        for clip in clips:
            if len(clip.t) < 2:
                continue
            keep = self._normal_mask(clip)
            if keep.any():
                normal.append(_log_features(clip.X, self.columns)[keep])
        if not normal:
            self.threshold_ = np.inf
            return

        pooled = np.vstack(normal)
        self.mean_ = pooled.mean(axis=0)
        self.scale_ = pooled.std(axis=0)
        self.scale_[self.scale_ <= 0] = 1.0

        # Calibrate the alarm level on those same normal frames, after smoothing, so the
        # threshold is in the units the running detector actually sees. A high quantile
        # rather than the maximum stops one freak training frame from deafening us.
        scores = []
        for clip in clips:
            if len(clip.t) < 2:
                continue
            keep = self._normal_mask(clip)
            if keep.any():
                scores.append(self._score(clip.X, _frame_rate(clip.t))[keep])
        pooled_scores = np.concatenate(scores)
        level = float(np.quantile(pooled_scores, self.threshold_quantile))
        self.threshold_ = level * self.margin if level > 0 else np.inf

    # -- prediction --------------------------------------------------------------

    def predict(self, clip) -> np.ndarray:
        n = len(clip.t)
        if n == 0:
            return np.zeros(0, dtype=bool)
        fs = _frame_rate(clip.t)
        over = self._score(clip.X, fs) > self.threshold_

        # Sustained-duration requirement: the score must have been over threshold for
        # `dwell_fraction` of the last `dwell_s` seconds. A single quiet breath is not
        # apnea; half a minute of quiet is.
        dwell = int(round(self.dwell_s * fs))
        if dwell > 1:
            held = _trailing_mean(over.astype(float), dwell)
            alarms = held >= self.dwell_fraction - 1e-9
            alarms[: dwell - 1] = False
        else:
            alarms = over.copy()

        refractory = int(round(self.refractory_s * fs))
        if refractory > 0:
            alarms = _refractory(alarms, refractory)
        return alarms


def build() -> AnomalyDetector:
    return AnomalyDetector()
