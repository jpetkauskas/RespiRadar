"""Temporal / sequence models over the feature history.

A single frame is ambiguous: the gap between two breaths looks exactly like the start of a
hold. What separates them is the trajectory, so everything here is computed over a trailing
window and nothing ever reads past the current sample.

Two detectors live in this file.

`BreathGapDetector` is the interpretable one: seconds since the chest last moved like a
breath. A breath is a rise of the presence slow-motion score above a fraction of this
person's own recent level, so the statistic carries no unit and no body size - a gap of
twelve seconds means the same thing on a large chest at 0.6 m as on a small one at 0.9 m.
The alarm gap itself is fitted, on the *other* subjects' negatives, as the longest gap that
ordinary breathing ever produced there.

`StackedWindowDetector` is the learned one: the twelve features at t, and again at t-2 s,
t-5 s, t-10 s and t-15 s, plus trailing mean / minimum / slope over 4, 8 and 16 s windows,
fed to an ordinary logistic regression. That gives a plain estimator the information a
sequence model would have. Its alarm threshold is not a constant either - it is set at the
highest score the model produced anywhere in the *training subjects'* negative recordings,
so the model has to be more certain about a new body than it ever was about a quiet minute
of a body it has seen.

Causality. Every window here is trailing (`_roll_*`, `_slope`, `_asym_env` all step forward
in time), the threshold is a constant fixed at fit time, and the persistence rule only looks
backwards. There is no whole-array normalisation and no zero-phase filtering.

Cross-subject honesty. Both detectors choose their thresholds inside `fit`, from the clips
they are handed, so under `bakeoff.folds()` (leave-one-subject-out) nothing about the
held-out body leaks into them. See the module-level note in `bakeoff` about how little
training accuracy is worth here: there are four labelled holds and the stacked-window model
has hundreds of parameters, so its in-sample separation is perfect and meaningless.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from respiradar.bakeoff import Clip
from respiradar.dataset import FEATURE_NAMES

FS = 20.0  # frames per second
WARMUP_S = 25.0  # matches evaluation.DEFAULT_WARMUP_S
IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


# --------------------------------------------------------------------------- causal windows


def _roll_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Mean of the trailing n samples (fewer at the start of the clip)."""
    cumulative = np.cumsum(np.insert(x, 0, 0.0))
    i = np.arange(len(x))
    lo = np.maximum(i - n + 1, 0)
    return (cumulative[i + 1] - cumulative[lo]) / (i - lo + 1)


def _roll_extreme(x: np.ndarray, n: int, want_min: bool) -> np.ndarray:
    """Running min/max over the trailing n samples, via a monotone deque."""
    out = np.empty(len(x))
    window: deque[int] = deque()
    for i, v in enumerate(x):
        while window and ((x[window[-1]] >= v) if want_min else (x[window[-1]] <= v)):
            window.pop()
        window.append(i)
        while window[0] <= i - n:
            window.popleft()
        out[i] = x[window[0]]
    return out


def _roll_min(x: np.ndarray, n: int) -> np.ndarray:
    return _roll_extreme(x, n, True)


def _slope(x: np.ndarray, n: int) -> np.ndarray:
    """Least-squares slope per second over the trailing n samples."""
    k = np.arange(n)
    centred = k - k.mean()
    weights = (centred / (centred**2).sum() * FS)[::-1]
    out = np.convolve(x, weights, mode="full")[: len(x)]
    out[: n - 1] = 0.0
    return out


def _ewma(x: np.ndarray, tau_s: float) -> np.ndarray:
    """Exponentially weighted trailing mean - 'what has this person been doing lately'."""
    a = 1.0 / (tau_s * FS)
    out = np.empty(len(x))
    level = x[0]
    for i, v in enumerate(x):
        level = (1 - a) * level + a * v
        out[i] = level
    return out


def _asym_env(x: np.ndarray, up_s: float, down_s: float) -> np.ndarray:
    """One-sided envelope: rises on a short time constant, falls on a long one.

    This is what makes a hold visible at all. A symmetric average would follow the signal
    down into the hold within a few seconds and declare the new, quieter level normal.
    """
    a_up, a_down = 1.0 / (up_s * FS), 1.0 / (down_s * FS)
    out = np.empty(len(x))
    level = x[0]
    for i, v in enumerate(x):
        a = a_up if v > level else a_down
        level = (1 - a) * level + a * v
        out[i] = level
    return out


def _persist(flag: np.ndarray, need: int) -> np.ndarray:
    """True once `flag` has been continuously true for `need` frames."""
    out = np.zeros(len(flag), dtype=bool)
    run = 0
    for i, q in enumerate(flag):
        run = run + 1 if q else 0
        out[i] = run >= need
    return out


def _scored(clip: Clip) -> np.ndarray:
    """Frames the bake-off will actually score, given the clip's own start time."""
    return clip.t >= clip.t[0] + WARMUP_S


# ------------------------------------------------------------------ breath-gap (no learning
#                                                                      beyond one threshold)

GAP_BASES = ("inter", "rms_4s", "disp_std_4s")


def _breath_gap(
    X: np.ndarray,
    base: str = "inter",
    frac: float = 0.4,
    smooth_s: float = 1.0,
    up_s: float = 4.0,
    down_s: float = 120.0,
) -> np.ndarray:
    """Seconds since the chest last moved like a breath.

    `base` is a per-frame chest-motion score; `inter` (the presence slow-motion score) is
    the one that actually oscillates once per breath - on the sleeping recordings its
    dominant period sits at 14-17 breaths per minute for all three subjects.

    A breath is registered whenever that score rises above `frac` of this person's own
    recent level. The reference level is a one-sided envelope, so a hold cannot drag it
    down to meet itself. The output is a duration, which is the point: it does not care
    how strong anyone's reflection is.
    """
    x = _roll_mean(X[:, IDX[base]].astype(float), max(int(smooth_s * FS), 1))
    log_x = np.log(np.maximum(x, 1e-6))
    relative = log_x - _asym_env(log_x, up_s, down_s)

    breath = relative > np.log(frac)
    breath[: int(WARMUP_S * FS)] = True  # do not accrue a gap while the envelope settles

    gap = np.empty(len(x))
    running = 0.0
    for i, is_breath in enumerate(breath):
        running = 0.0 if is_breath else running + 1 / FS
        gap[i] = running
    return gap


_GAP_GRID = [
    dict(base=b, frac=f, smooth_s=s, up_s=u, down_s=d)
    for b in GAP_BASES
    for f in (0.3, 0.4, 0.5, 0.6)
    for s in (1.0, 3.0)
    for u in (4.0, 8.0, 20.0)
    for d in (120.0, 400.0, 1500.0)
]


class BreathGapDetector:
    """Alarm when nobody has taken a breath for longer than anyone else ever went without."""

    name = "temporal/breath-gap"

    def __init__(self, margin: float = 1.0) -> None:
        self.margin = margin
        self.config = _GAP_GRID[0]
        self.alarm_gap_s = float("inf")

    def fit(self, clips) -> None:
        best = None
        for config in _GAP_GRID:
            gaps = {id(c): _breath_gap(c.X, **config) for c in clips}

            longest_normal = 0.0
            for clip in clips:
                quiet = _scored(clip) & ~clip.y
                if quiet.any():
                    longest_normal = max(longest_normal, float(gaps[id(clip)][quiet].max()))
            alarm_gap = longest_normal * self.margin + 1e-6

            missed, latencies = 0, []
            for clip in clips:
                alarm = (gaps[id(clip)] > alarm_gap) & _scored(clip)
                for hold in clip.holds:
                    inside = (clip.t >= hold.start_s) & (clip.t < hold.end_s) & alarm
                    fired = np.where(inside)[0]
                    if len(fired):
                        latencies.append(clip.t[fired[0]] - hold.start_s)
                    else:
                        missed += 1
            key = (missed, max(latencies) if latencies else 1e6, alarm_gap)
            if best is None or key < best[0]:
                best = (key, config, alarm_gap)

        _, self.config, self.alarm_gap_s = best

    def predict(self, clip: Clip) -> np.ndarray:
        return _breath_gap(clip.X, **self.config) > self.alarm_gap_s


# ------------------------------------------------------------------------- stacked windows

STACK_BASES = (
    "rms_4s",
    "rms_8s",
    "ratio_4s",
    "ratio_8s",
    "disp_std_4s",
    "autocorr",
    "intra",
    "inter",
)
STACK_WINDOWS = (4, 8, 16)  # seconds
STACK_LAGS = (2, 5, 10, 15)  # seconds


def _stack(X: np.ndarray) -> np.ndarray:
    """The feature history as one row per frame: lags, trailing stats and their differences."""
    columns = []
    for base in STACK_BASES:
        x = X[:, IDX[base]].astype(float)
        columns.append(x)

        # Dimensionless: where this feature sits against the same person a minute ago.
        columns.append(x / (_ewma(x, 60.0) + 1e-6))
        columns.append(x / (_ewma(x, 20.0) + 1e-6))

        for w in STACK_WINDOWS:
            n = int(w * FS)
            columns.append(_roll_mean(x, n))
            columns.append(_roll_min(x, n))
            columns.append(_slope(x, n))

        for lag_s in STACK_LAGS:
            k = int(lag_s * FS)
            lagged = np.concatenate([np.full(k, x[0]), x[:-k]])
            columns.append(lagged)
            columns.append(x - lagged)
    return np.column_stack(columns)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


class StackedWindowDetector:
    """Logistic regression over the stacked feature history, thresholded conservatively."""

    name = "temporal/stacked-window"

    def __init__(self, C: float = 0.03, dwell_s: float = 2.0, margin: float = 0.0) -> None:
        self.C = C
        self.dwell = int(dwell_s * FS)
        self.margin = margin
        self.threshold = float("inf")

    def fit(self, clips) -> None:
        rows, labels = [], []
        for clip in clips:
            keep = _scored(clip)
            rows.append(_stack(clip.X)[keep])
            labels.append(clip.y[keep])
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.C, max_iter=5000, class_weight="balanced"),
        )
        self.model.fit(np.vstack(rows), np.concatenate(labels))

        # The alarm level is whatever the model's most apnea-like *negative* moment scored
        # on the subjects it trained on, after the same dwell it will use at run time.
        # With four holds and this many parameters, any threshold read off the positives
        # would just be memorising them.
        worst_negative = -1e9
        for clip in clips:
            score = _logit(self.model.predict_proba(_stack(clip.X))[:, 1])
            sustained = _roll_min(score, self.dwell)
            quiet = _scored(clip) & ~clip.y
            if quiet.any():
                worst_negative = max(worst_negative, float(sustained[quiet].max()))
        self.threshold = worst_negative + self.margin

    def predict(self, clip: Clip) -> np.ndarray:
        score = _logit(self.model.predict_proba(_stack(clip.X))[:, 1])
        return _persist(score > self.threshold, self.dwell)


def build():
    """The entry the bake-off runs.

    The stacked-window model wins on the bake-off's own priority order (no false alarms
    first): leave-one-subject-out it catches 2 of the 4 holds with zero false alarms in
    16.8 minutes of negatives, where the breath-gap detector catches 3 but cries wolf
    twice. `BreathGapDetector` is kept beside it because it is the one that explains
    itself, and on a longer negative set it is the one worth re-measuring.
    """
    return StackedWindowDetector(C=0.03, dwell_s=2.0)
