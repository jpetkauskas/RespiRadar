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


def _causal_quantile(
    x: np.ndarray, window_s: float, q: float, valid: np.ndarray | None = None, decim: int = 10
) -> np.ndarray:
    """Rolling quantile of the last `window_s` seconds, evaluated on a decimated copy.

    Why a quantile and not the supplied `baseline`: the baseline ratchets up and decays only
    very slowly, so it tracks the subject's *best* breathing. A person who breathes shallowly
    for half a minute - which is most of what "sleeping" looks like on some bodies - sits far
    below that baseline the whole time and is indistinguishable from a hold. A rolling low
    quantile instead asks "is this quiet even by the standard of how quiet this person has
    been lately", which a long shallow plateau answers with "no" and a real hold with "yes".
    Strictly causal: window ends at the current (decimated) sample.
    """
    n = len(x)
    if n == 0:
        return np.zeros(0)
    d = np.asarray(x, dtype=float)[::decim]
    v = np.asarray(valid, dtype=bool)[::decim] if valid is not None else np.ones(len(d), bool)
    width = max(2, int(window_s * FS / decim))
    # Expanding until the window is full, rather than padding with the first sample: at the
    # start of a recording that sample is a filter transient near zero, and padding with it
    # would make the first two minutes - which is where a hold can hide - look loud. Frames
    # from before the analysis buffer filled are skipped outright for the same reason.
    qd = np.empty(len(d))
    for j in range(len(d)):
        lo = max(0, j - width + 1)
        window = d[lo : j + 1][v[lo : j + 1]]
        qd[j] = np.quantile(window, q) if len(window) >= 3 else d[j]
    return qd[np.minimum(np.arange(n) // decim, len(qd) - 1)]


def augment(X: np.ndarray) -> np.ndarray:
    """Expand the 12 raw features into the causal context the classifier actually needs."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n == 0:
        return np.zeros((0, 1))

    rms4 = X[:, F["rms_4s"]]
    rms8 = X[:, F["rms_8s"]]
    rms16 = X[:, F["rms_16s"]]
    base = np.maximum(X[:, F["baseline"]], 1e-6)
    intra = X[:, F["intra"]]
    inter = X[:, F["inter"]]
    amp = X[:, F["amplitude"]]
    disp = X[:, F["disp_std_4s"]]
    flat = X[:, F["flatness"]]
    ac = X[:, F["autocorr"]]
    valid = rms16 > 0  # the 16 s analysis buffer has filled; before that the row is a stub

    # The raw ratio columns blow up to ~1e8 in the first seconds, before the baseline exists.
    # Clip rather than drop: the same rows are refused an alarm by the warm-up gate anyway.
    r4 = np.clip(rms4 / base, 0.0, 4.0)
    r8 = np.clip(rms8 / base, 0.0, 4.0)
    r16 = np.clip(rms16 / base, 0.0, 4.0)

    # Everything below is dimensionless. Absolute millimetre amplitudes are deliberately
    # excluded: the breath-hold recording sits at a lower overall amplitude than the other
    # two sessions, so a model given raw rms learns "this session" instead of "this hold",
    # and then calls the subject's ordinary breathing apnea. Ratios against the subject's
    # own slow baseline do not have that hole.
    cols: list[np.ndarray] = [
        r4, r8, r16,
        np.clip(rms4 / np.maximum(rms16, 1e-6), 0.0, 4.0),
        np.clip(rms4 / np.maximum(rms8, 1e-6), 0.0, 4.0),
        intra, inter, flat, ac,
        # Gross motion relative to in-band motion: talking and fidgeting push this up.
        np.clip(disp / np.maximum(rms4, 1e-6), 0.0, 8.0),
        # Reflection strength relative to its own recent level - catches the person leaving
        # without letting the model key on how far away they happened to be that day.
        np.clip(amp / np.maximum(_causal_mean(amp, int(30 * FS)), 1e-6), 0.0, 4.0),
    ]

    for w in MEAN_WINDOWS_S:
        k = int(w * FS)
        cols.append(_causal_mean(r4, k))
        cols.append(_causal_mean(r8, k))
        cols.append(_causal_mean(intra, k))
        cols.append(_causal_mean(ac, k))
    for w in MIN_WINDOWS_S:
        k = int(w * FS)
        cols.append(_causal_min(r4, k))
        cols.append(_causal_max(r4, k))
        cols.append(_causal_max(intra, k))
    # How much of the recent past was spent below each "quiet" level. This is the feature that
    # a fixed threshold detector implements by hand, handed to the model at several scales.
    for level in QUIET_LEVELS:
        q = (r4 < level).astype(float)
        for w in QUIET_WINDOWS_S:
            cols.append(_causal_mean(q, int(w * FS)))
    for level in LOUD_LEVELS:
        cols.append(_time_since_above(r4, level))
        cols.append(_time_since_above(r8, level))

    # Self-referencing quietness: the current 4 s energy against low quantiles of the recent
    # past. This is the part that has to carry across bodies - see _causal_quantile.
    for window_s in (60.0, 120.0):
        for q in (0.1, 0.25, 0.5):
            ref = np.maximum(_causal_quantile(rms8, window_s, q, valid), 1e-6)
            cols.append(np.clip(rms4 / ref, 0.0, 6.0))
            cols.append(np.clip(rms8 / ref, 0.0, 6.0))
    ref50 = np.maximum(_causal_quantile(rms8, 120.0, 0.5, valid), 1e-6)
    rel = np.clip(rms4 / ref50, 0.0, 6.0)
    for w in (5.0, 15.0, 30.0):
        cols.append(_causal_mean(rel, int(w * FS)))
    # The same "how long has it been quiet" battery as above, but measured on `rel` rather
    # than on the supplied ratio. Fixed levels on ratio_4s are not comparable between bodies -
    # one subject breathes at ratio 1.3 and another at 0.65, so "below 0.5" means different
    # things to each - whereas `rel` is 1.0 by construction whenever a subject is doing what
    # they have recently been doing.
    for level in (0.30, 0.50, 0.70):
        q = (rel < level).astype(float)
        for w in QUIET_WINDOWS_S:
            cols.append(_causal_mean(q, int(w * FS)))
    for level in (0.6, 0.8, 1.0, 1.2):
        cols.append(_time_since_above(rel, level))
    for w in (4.0, 10.0):
        cols.append(_causal_min(rel, int(w * FS)))
        cols.append(_causal_max(rel, int(w * FS)))

    # Periodicity. Shallow breathing is still breathing: the autocorrelation of the band-
    # limited signal keeps a clear peak at the breathing period even when its amplitude has
    # collapsed. A hold has nothing to be periodic about. This is the feature that separates
    # one subject's quiet sleeping plateau from another subject's breath hold, and it needs
    # no per-body calibration at all.
    for w in (5.0, 10.0, 20.0, 40.0):
        cols.append(_causal_max(ac, int(w * FS)))
        cols.append(_causal_mean(ac, int(w * FS)))
    for level in (0.35, 0.50, 0.65):
        cols.append(_time_since_above(ac, level))
    for w in (10.0, 30.0):
        cols.append(_causal_min(flat, int(w * FS)))

    # A short-horizon drop detector. The quantile references above need a minute or two of
    # history, which a hold in the first half-minute of a recording does not have; this one
    # only needs its own window, so it is what catches an early hold.
    for w in (20.0, 40.0, 60.0):
        peak = np.maximum(_causal_max(rms8, int(w * FS)), 1e-6)
        cols.append(np.clip(rms4 / peak, 0.0, 2.0))
        cols.append(np.clip(rms8 / peak, 0.0, 2.0))

    return np.column_stack(cols)


class SupervisedApneaDetector:
    name = "supervised/rf-hysteresis"

    def __init__(
        self,
        on_threshold: float = 0.40,
        off_threshold: float = 0.20,
        smooth_s: float = 1.0,
        min_duration_s: float = 10.0,
        onset_grace_s: float = 6.0,
        recovery_grace_s: float = 6.0,
        model: str = "rf",
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
                max_depth=3,
                max_iter=150,
                learning_rate=0.08,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=0,
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
            # Frames before the 16 s analysis buffer is full carry placeholder features
            # (rms_16s is exactly 0). They are never alarmed on, so they are not trained on.
            usable = clip.X[:, F["rms_16s"]] > 0
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
        ready = clip.X[:, F["rms_16s"]] > 0
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
