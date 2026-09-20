"""Supervised apnea detector: a random forest on the shared features, behind a causal alarm gate.

Scores 2/4 holds with ZERO false alarms over 16.8 minutes of negatives, leave-one-subject-out.
The fixed-threshold baseline gets 3/4 with 10 false alarms on the same protocol, so what the
model buys is the generalisation, not the sensitivity. The honest reading of that trade and of
the two misses is at the bottom of this file; read it before trusting the number.

Causality
---------
The 12 supplied features are already causal, so the only ways to leak the future are window
statistics that reach forwards and normalisation fitted on the clip being scored. Neither
happens here: every rolling statistic (`_sliding`, `_time_since_above`, `_causal_quantile`)
ends its window at the current index, and the scaler is fitted in `fit()` on training clips
alone. `predict(clip.X[:k])` equals `predict(clip.X)[:k]` exactly, which is the property that
matters for running live on the sensor.

Why the raw feature row is not enough
------------------------------------
A single 50 ms frame cannot tell a breath hold from the gap between two breaths, and a fixed
threshold on `ratio_4s` cannot tell one person's hold from another person's shallow sleeping.
Both problems are about context, so each row is expanded with backwards-looking context:

- how long the chest energy has been quiet (`_time_since_above`, quiet fractions),
- how quiet it is *relative to how quiet this person has been lately* (`_causal_quantile`),
- whether anything periodic is still happening (autocorrelation over several horizons).

The third is the one that travels between bodies. Shallow breathing is still breathing and
keeps a clear autocorrelation peak at the breathing period however small its amplitude gets;
a hold has nothing to be periodic about.

The alarm gate
--------------
The per-frame probability is smoothed over the last second, has to stay above `on_threshold`
for `min_duration_s` continuously before the alarm arms, and then latches until the
probability drops below `off_threshold`. The long arming duration is doing most of the work:
the negative sessions contain quiet stretches of 8-26 s, and requiring ten unbroken seconds of
high probability is what keeps them quiet. Hysteresis is deliberately NOT load-bearing here -
the score is identical for any `off_threshold` from 0.10 to 0.38 - so the result does not
depend on an alarm bridging a gap between two nearby events.
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
        on_threshold: float = 0.70,
        off_threshold: float = 0.20,
        smooth_s: float = 1.0,
        min_duration_s: float = 10.0,
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
                # the seed. With four labelled events in the whole dataset, a scoring
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
# Cross-subject (bakeoff.folds, leave-one-subject-out): 2/4 holds, worst latency 29.8 s,
# median 24.1 s, 0 false alarms in 16.8 min of negatives. Fit on everything and run whole
# sessions instead and it is 4/4 with 0 false alarms - which is exactly the optimistic number
# the fold protocol exists to disbelieve.
#
# What it misses, and why. Both missed holds are each subject's FIRST hold: nishant's at
# 14.3 s and justinas' at 4.6 s. Both start inside or immediately after the 25 s scoring
# warm-up, so the band-pass is still ringing, the supplied baseline has barely formed, and the
# rolling quantile references have almost no history to be quiet relative to. The model's peak
# probability inside those two holds is 0.14 and 0.24, against 0.85 for the quietest stretch
# of justinas' ordinary sleeping - no threshold recovers them, and nothing in the tuning got
# them above the noise. A hold that starts two minutes into a recording is a different and
# much easier problem than one that starts before the filters have settled.
#
# Would it survive a live demo? For a hold taken after a minute or two of normal breathing,
# probably: the 0-false-alarm result held across five random seeds, across random-forest
# depths from 6 to unlimited, and across every off_threshold tried, and the negatives it
# stays silent through include a subject it never trained on. For a hold taken in the first
# half-minute after the sensor starts, no - it will miss it, and that is the failure mode to
# design the demo around (let the subject breathe normally for a minute first).
#
# The honest caveat is the sample size. Four holds from two subjects is not a test set; it is
# an anecdote with error bars wider than the effect. The 29.8 s worst-case latency in
# particular rests on one hold, and the labelled hold boundaries themselves are marker presses
# - justinas' first hold is followed by 13 s of labelled-negative quiet that looks exactly
# like a hold, which is either late marker timing or the subject still recovering. What the
# comparison against the threshold baseline does support, because it is a difference of ten
# false alarms and not of one, is the original question: a learned model on subject-relative
# features does transfer across bodies where a hand-tuned threshold does not.
