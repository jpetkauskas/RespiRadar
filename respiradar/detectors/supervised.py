"""Supervised apnea detector: a gradient-boosted classifier behind a causal duration gate.

Leave-one-subject-out over 13 holds and 23 minutes of negatives: 7/13 holds, ZERO false
alarms, worst-case latency 27.7 s, median 23.9 s. That is well behind the hand-built
change-point and spectral entries, and the reason is in the features rather than in the
model - see the assessment at the bottom of this file before reading the number as a verdict
on supervised learning.

Causality
---------
The shared features are already causal, so the only ways to leak the future are window
statistics that reach forwards and normalisation fitted on the clip being scored. Neither
happens here: every rolling statistic (`_sliding`, `_time_since_above`, `_down_cusum`) ends
its window at the current index, and the scaler is fitted in `fit()` on training clips alone.
`predict(clip.X[:k])` equals `predict(clip.X)[:k]` exactly, which is the property that matters
for running live on a sensor, and it is checked rather than assumed.

Why the raw feature row is not enough
-------------------------------------
A single 50 ms frame cannot tell a hold from the gap between two breaths, and a fixed level
cannot tell one person's hold from another person's shallow sleeping. Both are context
problems, so each row is expanded with backwards-looking context: how long it has been quiet,
how quiet it is relative to how quiet this person has recently been, whether anything
periodic is still happening, and one-sided CUSUM charts of the drop itself. `augment` says
which column is there for which reason.

The alarm gate
--------------
The per-frame probability is smoothed over the last second, must stay above `on_threshold`
for `min_duration_s` continuously to arm, and then latches until it falls below
`off_threshold`. The duration requirement is the load-bearing part: negative recordings
contain quiet stretches of 8-26 s, and demanding eight unbroken seconds of high probability
is what keeps them silent. Hysteresis is deliberately NOT load-bearing - the score is
identical for every `off_threshold` from 0.10 to 0.60 - so nothing here depends on one alarm
bridging the gap between two nearby events, which is how an earlier version of this file
bought its zeroes.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from respiradar.bakeoff import Clip
from respiradar.dataset import FEATURE_NAMES

F = {n: i for i, n in enumerate(FEATURE_NAMES)}

FS = 20.0  # frames per second; every session is recorded at this rate


def _sliding(x: np.ndarray, width: int) -> np.ndarray:
    """(n, width) view where row i is x[i-width+1 : i+1], edge-padded with x[0].

    Strictly causal: row i never contains a sample later than i.
    """
    width = max(1, int(width))
    if width == 1:
        return x[:, None]
    pad = np.full(width - 1, x[0], dtype=float)
    padded = np.concatenate([pad, np.asarray(x, dtype=float)])
    return np.lib.stride_tricks.sliding_window_view(padded, width)


def _causal_mean(x: np.ndarray, width: int) -> np.ndarray:
    return _sliding(x, width).mean(axis=1)


def _causal_min(x: np.ndarray, width: int) -> np.ndarray:
    return _sliding(x, width).min(axis=1)


def _causal_max(x: np.ndarray, width: int) -> np.ndarray:
    return _sliding(x, width).max(axis=1)


QUIET_LEVELS = (0.35, 0.50, 0.70)
MEAN_WINDOWS_S = (2.0, 5.0, 10.0, 20.0)
MIN_WINDOWS_S = (4.0, 10.0)
QUIET_WINDOWS_S = (5.0, 10.0, 20.0, 40.0)
LOUD_LEVELS = (0.70, 0.90, 1.10)
SILENCE_CAP_S = 45.0


def _time_since_above(x: np.ndarray, level: float, cap_s: float = SILENCE_CAP_S) -> np.ndarray:
    """Seconds since x was last at or above `level`, capped. Causal by construction.

    This is the feature an apnea detector really wants: how long it has been since the chest
    last moved like a breath. It is also what separates a real hold from the two-second pause
    between two breaths, without any threshold tuning by hand.
    """
    n = len(x)
    idx = np.arange(n, dtype=float)
    hit = np.where(x >= level, idx, -1.0)
    last = np.maximum.accumulate(hit)
    # Before the first loud frame, treat the clip start as the reference point.
    last = np.where(last < 0, 0.0, last)
    return np.minimum((idx - last) / FS, cap_s)


def _down_cusum(z: np.ndarray, drift: float, cap: float = 60.0) -> np.ndarray:
    """One-sided CUSUM accumulating evidence that `z` has shifted downwards.

    S[i] = clip(S[i-1] - (z[i] + drift), 0, cap). It rises only while z sits more than
    `drift` below zero, and decays as soon as the signal comes back, so it measures the
    depth-times-duration of a drop rather than its instantaneous depth. Causal by shape,
    and it is the statistic the hand-built change-point entries in this bake-off win on.
    """
    n = len(z)
    out = np.empty(n)
    s = 0.0
    for i in range(n):
        s = min(max(s - (z[i] + drift), 0.0), cap)
        out[i] = s
    return out


def augment(X: np.ndarray) -> np.ndarray:
    """Expand the shared feature row into the causal context the classifier needs.

    Column choices, in one line each:

    - Nothing absolute. Reflection amplitude and millimetre rms differ by body and posture,
      and a model given them learns "this recording" rather than "this hold".
    - `ratio_*_q`, the shared 25th-percentile self-reference, is preferred over `ratio_*`,
      which divides by the ratcheting baseline. The ratchet tracks a subject's *best*
      breathing, so a shallow sleeper sits far below it all night and reads as apnea.
    - Duration, in several forms. A frame cannot tell a hold from the gap between two
      breaths; how long it has been quiet can.
    - Periodicity. Shallow breathing keeps an autocorrelation peak at the breathing period
      however small its amplitude; a hold has nothing to be periodic about. This is the part
      that needs no per-body calibration.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n == 0:
        return np.zeros((0, 1))

    rms4 = X[:, F["rms_4s"]]
    rms8 = X[:, F["rms_8s"]]
    rms16 = X[:, F["rms_16s"]]
    intra = X[:, F["intra"]]
    inter = X[:, F["inter"]]
    amp = X[:, F["amplitude"]]
    disp = X[:, F["disp_std_4s"]]
    flat = X[:, F["flatness"]]
    ac = X[:, F["autocorr"]]
    ref = np.maximum(X[:, F["ref_q25"]], 1e-6)
    warm = X[:, F["warm"]]
    history = X[:, F["seconds_of_history"]]

    r4 = np.clip(X[:, F["ratio_4s"]], 0.0, 4.0)  # against the ratcheting baseline
    r8 = np.clip(X[:, F["ratio_8s"]], 0.0, 4.0)
    q4 = np.clip(X[:, F["ratio_4s_q"]], 0.0, 4.0)  # against the trailing 25th percentile
    q8 = np.clip(X[:, F["ratio_8s_q"]], 0.0, 4.0)

    cols: list[np.ndarray] = [
        r4, r8, q4, q8,
        np.clip(rms4 / np.maximum(rms16, 1e-6), 0.0, 4.0),
        np.clip(rms4 / np.maximum(rms8, 1e-6), 0.0, 4.0),
        intra, inter, flat, ac,
        # Gross motion relative to in-band motion: talking and fidgeting push this up.
        np.clip(disp / np.maximum(rms4, 1e-6), 0.0, 8.0),
        # Reflection strength against its own recent level - catches the person leaving
        # without letting the model key on how far away they happened to be that day.
        np.clip(amp / np.maximum(_causal_mean(amp, int(30 * FS)), 1e-6), 0.0, 4.0),
        # How much the model should trust the rest of the row.
        warm, np.minimum(history, 180.0),
    ]

    for w in MEAN_WINDOWS_S:
        k = int(w * FS)
        cols.append(_causal_mean(q4, k))
        cols.append(_causal_mean(q8, k))
        cols.append(_causal_mean(r4, k))
        cols.append(_causal_mean(intra, k))
    for w in MIN_WINDOWS_S:
        k = int(w * FS)
        cols.append(_causal_min(q4, k))
        cols.append(_causal_max(q4, k))
        cols.append(_causal_max(intra, k))

    # How much of the recent past was spent below each "quiet" level, at several horizons.
    # This is what a threshold detector implements by hand, handed over at several scales -
    # on the self-referenced ratio, so the levels mean the same thing on every body.
    for level in QUIET_LEVELS:
        for source in (q4, r4):
            q = (source < level).astype(float)
            for w in QUIET_WINDOWS_S:
                cols.append(_causal_mean(q, int(w * FS)))
    for level in LOUD_LEVELS:
        cols.append(_time_since_above(q4, level))
        cols.append(_time_since_above(q8, level))
        cols.append(_time_since_above(r4, level))

    for w in (5.0, 10.0, 20.0, 40.0):
        cols.append(_causal_max(ac, int(w * FS)))
        cols.append(_causal_mean(ac, int(w * FS)))
    for level in (0.35, 0.50, 0.65):
        cols.append(_time_since_above(ac, level))
    for w in (10.0, 30.0):
        cols.append(_causal_min(flat, int(w * FS)))

    # Drop detectors. The percentile reference needs a minute or two of history; these need
    # only their own window, which is what covers a hold early in a recording.
    for w in (20.0, 40.0, 60.0):
        peak = np.maximum(_causal_max(rms8, int(w * FS)), 1e-6)
        cols.append(np.clip(rms4 / peak, 0.0, 2.0))
        cols.append(np.clip(rms8 / peak, 0.0, 2.0))

    z4 = np.log(np.maximum(rms4, 1e-9)) - np.log(ref)
    z8 = np.log(np.maximum(rms8, 1e-9)) - np.log(ref)
    for drift in (0.15, 0.35, 0.70):
        cols.append(_down_cusum(z4, drift))
    for drift in (0.15, 0.35):
        cols.append(_down_cusum(z8, drift))

    return np.column_stack(cols)


class SupervisedApneaDetector:
    name = "supervised/hgb-duration"

    def __init__(
        self,
        on_threshold: float = 0.92,
        off_threshold: float = 0.20,
        smooth_s: float = 1.0,
        min_duration_s: float = 8.0,
        onset_grace_s: float = 6.0,
        recovery_grace_s: float = 6.0,
        model: str = "hgb",
        class_weight: float = 5.0,
        name: str | None = None,
    ) -> None:
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.smooth_s = smooth_s
        self.min_duration_s = min_duration_s
        self.onset_grace_s = onset_grace_s
        self.recovery_grace_s = recovery_grace_s
        self.model_kind = model
        self.class_weight = class_weight
        self.model = None
        self.scaler = None
        if name:
            self.name = name

    # ----------------------------------------------------------------- fitting

    def _new_model(self):
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        if self.model_kind == "hgb":
            return HistGradientBoostingClassifier(
                max_depth=5,
                max_iter=100,
                learning_rate=0.05,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=0,
                # Off deliberately: sklearn turns early stopping on above 10k rows, which
                # carves out a random validation split and makes the fitted model depend on
                # the seed. With thirteen labelled events in the whole dataset, a scoring
                # difference that comes from a seed is noise being mistaken for tuning.
                early_stopping=False,
            )
        if self.model_kind == "rf":
            return RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=0,
                n_jobs=-1,
            )
        if self.model_kind == "logreg":
            return LogisticRegression(
                C=0.3, max_iter=2000, class_weight="balanced", random_state=0
            )
        raise ValueError(self.model_kind)

    def fit(self, clips: Sequence[Clip]) -> None:
        from sklearn.preprocessing import StandardScaler

        rows, labels, weights = [], [], []
        for clip in clips:
            if not len(clip.t):
                continue
            feats = augment(clip.X)
            # Frames before the 16 s analysis buffer has filled are computed over a partial
            # window and are not comparable to the rest. They are never alarmed on, so they
            # are not trained on either.
            usable = clip.X[:, F["warm"]] >= 1.0
            if not usable.any():
                continue
            # The features lag the event: the first seconds of a hold still look like
            # breathing (a 4 s rms window is still full of the last breath), and the seconds
            # after a hold ends are the subject catching their breath, which looks quiet.
            # Training on either teaches the model to call ordinary pauses apnea, so those
            # frames are given zero weight instead of a wrong label.
            w = np.ones(len(clip.t))
            for hold in clip.holds:
                w[(clip.t >= hold.start_s) & (clip.t < hold.start_s + self.onset_grace_s)] = 0.0
                w[(clip.t >= hold.end_s) & (clip.t < hold.end_s + self.recovery_grace_s)] = 0.0
            rows.append(feats[usable])
            labels.append(clip.y[usable])
            weights.append(w[usable])
        if not rows:
            raise ValueError("no usable training frames")

        Xf = np.vstack(rows)
        yf = np.concatenate(labels).astype(int)
        wf = np.concatenate(weights)
        if self.class_weight != 1.0 and yf.any():
            wf = np.where(yf == 1, wf * self.class_weight, wf)

        self.scaler = StandardScaler().fit(Xf)  # fitted on TRAINING clips only
        model = self._new_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.scaler.transform(Xf), yf, sample_weight=wf)
        self.model = model

    # -------------------------------------------------------------- prediction

    def probabilities(self, clip: Clip) -> np.ndarray:
        """Smoothed per-frame apnea probability. Exposed for tuning and inspection."""
        if self.model is None:
            raise RuntimeError("fit() first")
        n = len(clip.t)
        if n == 0:
            return np.zeros(0)
        feats = self.scaler.transform(augment(clip.X))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = self.model.predict_proba(feats)[:, 1]
        k = max(1, int(self.smooth_s * FS))
        return _causal_mean(p, k)  # trailing mean: causal smoothing, no phase peeking

    def predict(self, clip: Clip) -> np.ndarray:
        n = len(clip.t)
        if n == 0:
            return np.zeros(0, dtype=bool)
        p = self.probabilities(clip)

        # Gate: never alarm before the feature buffers have filled, in absolute session time
        # as well as time since the clip started (a clip may begin mid-session).
        ready = clip.X[:, F["warm"]] >= 1.0
        ready &= (clip.t - clip.t[0]) >= self.smooth_s

        need = max(1, int(self.min_duration_s * FS))
        alarms = np.zeros(n, dtype=bool)
        run = 0
        latched = False
        for i in range(n):
            hot = bool(ready[i]) and p[i] >= self.on_threshold
            if latched:
                if p[i] < self.off_threshold or not ready[i]:
                    latched = False
                    run = 0
            else:
                run = run + 1 if hot else 0
                if run >= need:
                    latched = True
            alarms[i] = latched
        return alarms


def build():
    return SupervisedApneaDetector()


# ---------------------------------------------------------------------------------------
# Honest assessment
# ---------------------------------------------------------------------------------------
# Cross-subject (bakeoff.folds, leave-one-subject-out): 7/13 holds, 0 false alarms in 23.0
# minutes, worst latency 27.7 s, median 23.9 s. Fit on everything and run whole sessions and
# it is 13/13 with 0 false alarms - which is exactly the optimistic number the fold protocol
# exists to disbelieve.
#
# Why it loses to the hand-built entries. It is not the classifier and it is not the alarm
# gate; it is that the shared feature row does not describe most of these holds. Taking the
# mean of rms_4s inside each labelled hold over its own session's mean outside, across all 13
# holds, gives 0.46 to 1.47 - four holds have MORE band-limited chest energy while the
# subject is holding their breath than while they are breathing. `ratio_4s_q`, the trailing
# 25th-percentile self-reference, averages above 1.0 inside 11 of the 13 holds. A detector
# that reads these columns as levels therefore cannot see most of the events at any
# threshold, and the per-hold probabilities show exactly that: within one session the model
# is at 0.99 for one hold and 0.01 for the two before it. The entries that score 12/13 and
# 13/13 recompute their own statistic from the range data instead of reading these columns.
#
# The binding false alarm is vishnu. Sustained over 12 s, the highest apnea probability
# anywhere in the negative recordings is 0.86, and it is in vishnu-sleeping - the one subject
# with no holds at all, so the model never sees a body like that labelled either way. Every
# other negative session sits below 0.3 at that duration. The operating point is set by that
# one recording; without it the same model would run at a much lower threshold and catch
# more.
#
# Would it survive a live demo? For a hold on a body resembling one it trained on, and taken
# after a minute of normal breathing, yes - and it will not cry wolf, which is the property
# it was tuned for. But it misses about half the holds, so it should not be the thing on
# stage. If this approach is wanted in the demo, the useful shape is as a veto or a second
# opinion beside a detector that computes its own drop statistic, not as the primary alarm.
#
# What is worth keeping from it regardless of which detector ships: dimensionless
# subject-relative features rather than absolute amplitudes (a model given raw rms learns
# which recording it is looking at), and duration rather than depth as the thing that buys
# off false alarms. Both survived every revision of the features in this project.
#
# Caveats on the numbers themselves. Thirteen holds from two subjects is three times what
# this file was first tuned on and still not a test set; frames inside a hold are correlated,
# so the effective n is 13. The 0-false-alarm result holds for `on_threshold` from 0.90 to
# 0.96 at `min_duration_s` 8 and for every `off_threshold` tried, and early stopping is
# switched off so the fit does not move with the seed - but 0.88 costs a false alarm, so the
# margin on the low side is one grid step, not a comfortable band.
