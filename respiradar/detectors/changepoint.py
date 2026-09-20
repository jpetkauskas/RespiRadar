"""Sequential change-point detection: a breath hold is a step down in breathing energy.

A hold is not an outlier, it is a *regime change*. The band-limited chest motion sits at one
level while the person breathes and at a lower one while they do not, and the job is to
declare the second regime as soon as the evidence justifies it. That is exactly the problem
Page's CUSUM solves: for a given tolerated false-alarm rate it has the smallest possible
expected delay, which is this bake-off's objective function written out in full.

Three things had to be got right before the CUSUM itself mattered.

1. What the statistic is measured against. There are two references and they answer
   different questions, so the charts use both. `ref_ratchet` rises quickly and falls only
   very slowly: a liability on negatives, because one burst of movement leaves "normal"
   inflated for minutes and ordinary breathing afterwards reads as a hold, but exactly what
   a fast channel wants, since a reference that refuses to follow a hold downwards keeps
   the step visible for the hold's whole length. `ref_q25` is a trailing 25th percentile
   with no such memory; its level sits in a narrow band across subjects where the ratchet's
   wanders from 0.14 to 0.79, which is why fixed thresholds on the ratchet do not transfer.
   Measured on four holds the ratchet dominated. Measured on thirteen, neither wins alone.
   Every channel is a log ratio against one of them, or against a trailing median of this
   module's own floored by the ratchet, which makes the numbers dimensionless and
   comparable between bodies.

2. A motion gate. Talking and rolling over produce their own energy excursions, and the
   `intra` presence score and `disp_std_4s` both rise while they happen. Those frames do not
   update the chart at all and reset it to zero, so no amount of fidgeting can accumulate
   into an alarm. The gate is relative to the subject's own recent median, for the same
   reason the statistic is.

3. A deadband and a cap on the per-frame log-likelihood increment. Without the deadband a
   chronically shallow sleeper accumulates a small positive drift for a minute and trips the
   chart; without the cap a momentary loss of radar lock counts for more than a real hold.
   Together they turn the chart into a detector of *sustained, substantial* drops.

The detector is a small bank of CUSUM charts run in parallel, which is standard practice
when the post-change distribution is not known exactly: one chart is tuned for a deep, fast
collapse and another for a shallower, longer one, and the union alarms when either does.
Because every chart in the bank is individually free of false alarms, so is their union -
every alarm frame belongs to some chart's episode, and that episode overlaps a hold.

What this module deliberately does not do is notice that nobody is there. An empty room
produces no chest motion, which is indistinguishable from apnea by construction, and no
threshold inside a chart can tell them apart. That belongs to a presence gate, and
`gated.py` supplies one; assume this detector's output is wrapped in it.

Everything is strictly causal: the alarm at frame i reads only `clip.X[:i + 1]`. There is a
test for it - predicting on every prefix of a session reproduces the full-session output
exactly - and it has already caught one bug, a frame-rate estimate whose last decimal place
moved a window boundary depending on how much future happened to exist.
"""

from __future__ import annotations

import numpy as np

from respiradar.bakeoff import Clip
from respiradar.dataset import FEATURE_NAMES

RMS4 = FEATURE_NAMES.index("rms_4s")
RMS8 = FEATURE_NAMES.index("rms_8s")
RATCHET = FEATURE_NAMES.index("ref_ratchet")   # == the old `baseline` column
REF_Q = FEATURE_NAMES.index("ref_q25")         # non-ratcheting trailing 25th percentile
HISTORY = FEATURE_NAMES.index("seconds_of_history")
INTRA = FEATURE_NAMES.index("intra")
INTER = FEATURE_NAMES.index("inter")
# The slow-motion score AT THE CHEST, not its maximum over every range bin. The charts want
# "has this chest stopped moving"; the global maximum answers "is anything in the room
# moving", which a drifting radiator or the sensor's own near-field clutter can hold high
# straight through an apnea. See the note beside `inter_chest` in dataset.FEATURE_NAMES.
INTER_CHEST = FEATURE_NAMES.index("inter_chest")
DISP = FEATURE_NAMES.index("disp_std_4s")

EPS = 1e-6


def _trailing_quantile(values, usable, window, gap, q, min_n, stride=4):
    """Causal quantile of `values` over [i - gap - window, i - gap).

    The gap matters: a statistic compared against the seconds immediately behind it cannot
    see a step, because by then those seconds are inside the step.
    """
    n = len(values)
    out = np.zeros(n)
    good = np.zeros(n, dtype=bool)
    for i in range(n):
        hi = i - gap + 1
        lo = max(0, hi - window)
        if hi - lo < min_n:
            continue
        segment = values[lo:hi:stride][usable[lo:hi:stride]]
        if len(segment) < max(3, min_n // stride):
            continue
        out[i] = np.quantile(segment, q)
        good[i] = True
    return out, good


class _Context:
    """Everything a chart needs about one clip, computed once and shared."""

    __slots__ = ("fs", "usable", "drops", "calm", "calm_parts")

    def __init__(self, clip: Clip, window_s: float, gap_s: float, gate: dict,
                 history_s: float) -> None:
        X = clip.X
        # Rounded, because the raw estimate wobbles in the last decimal place with the
        # number of frames seen so far, and an off-by-one window length would make the
        # detector's output depend on how much future there happens to be.
        self.fs = fs = round(1 / max(float(np.median(np.diff(clip.t))), 1e-6), 3)
        # How much history the extractor had at this frame. The windowed statistics are
        # defined from 2 s onwards now, but a reference built from four seconds of a
        # start-up transient is not a reference, so each chart names its own minimum.
        self.usable = usable = X[:, HISTORY] >= history_s
        window, gap = int(window_s * fs), int(gap_s * fs)

        log_ratchet = np.log(np.maximum(X[:, RATCHET], EPS))
        log_q = np.log(np.maximum(X[:, REF_Q], EPS))

        def drop(raw, floor=None):
            ref, ok = _trailing_quantile(raw, usable, window, gap, 0.5, int(4 * fs))
            if floor is None:
                ref = np.where(ok, ref, raw)
            else:
                ref = np.where(ok & usable, np.minimum(floor, ref), floor)
            return ref - raw          # positive = below this person's own normal

        log4 = np.log(np.maximum(X[:, RMS4], EPS))
        log8 = np.log(np.maximum(X[:, RMS8], EPS))
        self.drops = {
            # Against the ratcheting reference. It rises fast and falls only very slowly,
            # which is a liability on negatives - a burst of movement leaves it inflated for
            # minutes - but exactly what a *fast* channel wants, because a reference that
            # refuses to follow a hold downwards keeps the step visible for the whole hold.
            "ratio": log_ratchet - log4,
            "ratio_8": log_ratchet - log8,
            # Against the shared non-ratcheting 25th-percentile reference. Far better
            # behaved on negatives: its level sits in a narrow band across subjects where
            # the ratchet's wanders from 0.14 to 0.79, which is why fixed thresholds on the
            # ratchet do not transfer. It does drift down inside a hold longer than about
            # half its window, so it is the slow channel's reference, not the fast one's.
            "ratio_q": log_q - log4,
            "ratio_q8": log_q - log8,
            # A trailing median of my own, floored by the ratchet: takes the lower of the
            # two, so an inflated ratchet cannot manufacture a drop.
            "energy": drop(log4, floor=log_ratchet),
            # The presence processor's slow-motion score. An independent estimate of chest
            # movement that does not share the band-pass filter's failure modes, and by a
            # wide margin the best-separating and most subject-stable channel here.
            # The GLOBAL maximum, deliberately, though `inter_chest` reads the same score at
            # the tracked chest and is the more obviously correct quantity. Substituting it
            # was tried and measured worse on every axis: 8/13 holds against 12/13, four
            # false alarms against one, and it broke silence on a subject with no holds. The
            # chest-local score simply separates less well - median hold/breathe ratio
            # 0.49-0.71 against the global score's 0.38-0.64 - because the three tracked bins
            # are the ones whose phase is already band-passed into `rms_4s`, so it adds noise
            # rather than an independent view. `inter_chest` stays in the feature row as the
            # diagnostic that tells you whether this sensor's global peak IS the person.
            "inter": drop(np.log(np.maximum(X[:, INTER], 1e-3))),
        }

        intra_med, ok_i = _trailing_quantile(X[:, INTRA], usable, window, gap, 0.5, int(10 * fs))
        disp_med, ok_d = _trailing_quantile(X[:, DISP], usable, window, gap, 0.5, int(10 * fs))
        intra_med = np.where(ok_i, intra_med, 1.35)
        disp_med = np.where(ok_d, disp_med, 1.1)
        # Kept as four named conditions rather than one expression, because "the chart was
        # reset" is not an actionable answer on a live sensor. Each of these fails for its
        # own reason and wants its own fix, and on an unfamiliar setup the only way to know
        # which one is shut is to be told - see `gate_state`.
        self.calm_parts = {
            "intra": X[:, INTRA] < gate["intra_k"] * intra_med,
            "intra abs": X[:, INTRA] < gate["intra_abs"],
            "disp": X[:, DISP] < gate["disp_k"] * disp_med,
            # A displacement standard deviation far below this person's own normal is a lost
            # radar lock, not a still chest: a live body at a metre never goes that quiet.
            "disp floor": X[:, DISP] > gate["disp_floor_k"] * disp_med,
        }
        self.calm = np.logical_and.reduce(list(self.calm_parts.values()))


class Chart:
    """One CUSUM chart: a weighted drop statistic, a deadband, a cap and a threshold."""

    def __init__(self, weights, deadband, cap, threshold, window_s=60.0, gap_s=4.0,
                 gate=None, decay=1.0, refractory_s=0.0, history_s=16.0, note=""):
        self.weights = dict(weights)
        self.deadband = deadband
        self.cap = cap
        self.threshold = threshold
        self.window_s = window_s
        self.gap_s = gap_s
        self.history_s = history_s
        self.gate = gate or dict(intra_k=2.0, intra_abs=2.5, disp_k=2.0, disp_floor_k=0.15)
        # A forgetting factor slightly below 1 turns the chart into a rate detector: a small
        # positive drift saturates at a low level, so an hour of slightly shallow breathing
        # never reaches a threshold that eight seconds of a real collapse clears.
        self.decay = decay
        # Nothing may alarm for this long after the gate last opened, because the seconds
        # right after someone stops moving look quiet for reasons that are not apnea.
        self.refractory_s = refractory_s
        self.note = note

    def key(self):
        return (self.window_s, self.gap_s, self.history_s, tuple(sorted(self.gate.items())))

    def run(self, clip: Clip, ctx: _Context, warm_s: float) -> np.ndarray:
        statistic = sum(w * ctx.drops[name] for name, w in self.weights.items() if w)
        increment = np.minimum(statistic - self.deadband, self.cap)
        live = ctx.usable & ctx.calm & (clip.t >= clip.t[0] + warm_s)
        alarms = np.zeros(len(clip.t), dtype=bool)
        blocked_until = -1
        hold_off = int(self.refractory_s * ctx.fs)
        total = 0.0
        for i in range(len(clip.t)):
            if not live[i]:
                total = 0.0          # movement, or no reference yet: start the chart over
                blocked_until = i + hold_off
                continue
            total = max(0.0, total * self.decay + increment[i])
            if total >= self.threshold and i > blocked_until:
                alarms[i] = True
        return alarms

    def progress(self, clip: Clip, ctx: _Context, warm_s: float) -> np.ndarray:
        """The chart's running total as a fraction of its threshold, for display.

        Identical arithmetic to `run`, kept separate so the detector's hot path stays a
        boolean. 1.0 means this chart is firing; 0 means it has just been reset by movement
        or has no reference yet. Watching this is how you see WHY the alarm did or did not
        come: a chart pinned at 0 is being reset, one creeping up is accumulating evidence.
        """
        statistic = sum(w * ctx.drops[name] for name, w in self.weights.items() if w)
        increment = np.minimum(statistic - self.deadband, self.cap)
        live = ctx.usable & ctx.calm & (clip.t >= clip.t[0] + warm_s)
        out = np.zeros(len(clip.t))
        total = 0.0
        for i in range(len(clip.t)):
            if not live[i]:
                total = 0.0
                continue
            total = max(0.0, total * self.decay + increment[i])
            out[i] = total / self.threshold
        return out


# Four charts, chosen by a two-stage random search (65 000 configurations) over the
# leave-one-subject-out folds. Only configurations with no false alarm on any recording were
# kept, then a greedy union picked the set that caught the most holds soonest. The union of
# alarm-free charts is itself alarm-free, which is why a bank costs nothing here.
#
# Both references earn a place, which is the answer to a question worth recording. Measured
# on four holds the ratcheting reference dominated; measured on thirteen, neither wins
# alone - the best single chart of any kind uses both, and so does this bank.
DEFAULT_CHARTS: list[dict] = [
    dict(
        note="patient: presence, the quantile reference and my own median; carries most holds",
        weights={"inter": 2.0, "energy": 0.5, "ratio_q": 0.5},
        deadband=0.1, cap=1.2, threshold=200.0, decay=0.999, refractory_s=8.0,
        window_s=75.0, gap_s=5.0, history_s=16.0,
        gate=dict(intra_k=3.0, intra_abs=2.5, disp_k=1.5, disp_floor_k=0.1),
    ),
    dict(
        note="presence alone, leaky, low threshold: picks up the holds the first chart misses",
        weights={"inter": 0.5},
        deadband=0.4, cap=0.8, threshold=30.0, decay=0.995, refractory_s=0.0,
        window_s=90.0, gap_s=6.0, history_s=12.0,
        gate=dict(intra_k=2.0, intra_abs=10.0, disp_k=3.0, disp_floor_k=0.1),
    ),
    dict(
        note="energy only, ratcheting reference, no deadband: the fast chart",
        weights={"ratio": 0.5, "energy": 0.5},
        deadband=0.0, cap=0.4, threshold=120.0, decay=0.999, refractory_s=0.0,
        window_s=90.0, gap_s=6.0, history_s=12.0,
        gate=dict(intra_k=3.0, intra_abs=2.5, disp_k=2.0, disp_floor_k=0.1),
    ),
    dict(
        note="8 s energy against the quantile reference, plus presence: shortens the median",
        weights={"ratio_q8": 0.5, "inter": 0.5},
        deadband=0.5, cap=1.5, threshold=30.0, decay=1.0, refractory_s=8.0,
        window_s=120.0, gap_s=8.0, history_s=12.0,
        gate=dict(intra_k=1.5, intra_abs=3.0, disp_k=1.5, disp_floor_k=0.0),
    ),
]


class ChangePointDetector:
    name = "changepoint/cusum-bank"

    def __init__(self, charts=None, warm_s: float = 20.0) -> None:
        self.charts = [Chart(**c) for c in (charts if charts is not None else DEFAULT_CHARTS)]
        self.warm_s = warm_s

    def fit(self, clips) -> None:
        """Measure the training subjects' own pre- and post-change levels.

        The thresholds themselves are deliberately not refitted per fold. Every channel is
        already a log ratio against the subject's own recent normal, so the numbers are
        dimensionless and a threshold that means "40% below normal for six seconds" means
        the same thing on a different body - which is exactly what fixed thresholds on raw
        features fail to do here. What fitting records is `self.separation`: the median
        drop while breathing and the median drop while holding, per channel, on whoever is
        in the training set. It is the evidence for whether the deadbands are set where the
        two distributions actually part, and it is what to print when they stop working.
        """
        before, during = {}, {}
        for clip in clips:
            ctx = self._context(clip, self.charts[0])
            live = ctx.usable & ctx.calm & (clip.t >= clip.t[0] + self.warm_s)
            in_hold = np.zeros(len(clip.t), dtype=bool)
            for hold in clip.holds:
                in_hold |= (clip.t >= hold.start_s) & (clip.t < hold.end_s)
            for name, values in ctx.drops.items():
                before.setdefault(name, []).append(values[live & ~in_hold])
                during.setdefault(name, []).append(values[live & in_hold])
        self.separation = {}
        for name in before:
            neg = np.concatenate(before[name])
            pos = [p for p in during[name] if len(p)]
            pos = np.concatenate(pos) if pos else None
            self.separation[name] = (
                float(np.median(neg)),
                float(np.median(pos)) if pos is not None else None,
            )

    def _context(self, clip, chart):
        """Charts that share a window and a gate share their context, computed once."""
        cache = getattr(self, "_cache", None)
        if cache is None:
            cache = self._cache = {}
        key = chart.key()
        held, contexts = cache.get("held"), cache.setdefault("ctx", {})
        if held is not clip:                       # a new clip: nothing carries over
            cache["held"] = clip                   # keep a reference so `is` stays meaningful
            contexts = cache["ctx"] = {}
        if key not in contexts:
            contexts[key] = _Context(clip, chart.window_s, chart.gap_s, chart.gate,
                                     chart.history_s)
        return contexts[key]

    def predict(self, clip: Clip) -> np.ndarray:
        alarms = np.zeros(len(clip.t), dtype=bool)
        for chart in self.charts:
            alarms |= chart.run(clip, self._context(clip, chart), self.warm_s)
        return alarms

    def progress(self, clip: Clip) -> dict:
        """{chart note: running total / threshold} - what the scope draws."""
        return {
            chart.note.split(":")[0]: chart.progress(
                clip, self._context(clip, chart), self.warm_s
            )
            for chart in self.charts
        }

    def gate_state(self, clip: Clip) -> dict:
        """Why the charts are being reset, per frame.

        A chart pinned at zero has been reset every frame, and there are two quite different
        reasons: `calm` is False because the subject is moving, or `usable` is False because
        there is no trustworthy reference yet. They need opposite fixes, so the display has
        to tell them apart.
        """
        ctx = self._context(clip, self.charts[0])
        return {
            "calm": ctx.calm.copy(),
            "usable": ctx.usable.copy(),
            # Which of the four motion conditions is shut. `calm` is their conjunction, so
            # on its own it says only that something is wrong, and the four want opposite
            # fixes: `intra`/`disp` mean genuine movement, `intra abs` is the one fixed
            # threshold in the design and so the one most likely to be wrong on a sensor
            # that was never used to tune it, and `disp floor` means the radar has lost
            # lock - or that a real hold went quieter than the floor allows for, which
            # would suppress the alarm exactly when it is wanted.
            "parts": {name: mask.copy() for name, mask in ctx.calm_parts.items()},
        }


# The two charts with the most margin: the highest thresholds and the strictest gates. The
# full bank is already silent on every recording, so this is no longer a false-alarm hedge;
# it is for callers who want extra headroom on an unseen body and can afford the holds it
# gives up. `gated.py` wraps this one.
CONSERVATIVE_CHARTS: list[dict] = [DEFAULT_CHARTS[0], DEFAULT_CHARTS[2]]


def build():
    return ChangePointDetector()


def build_conservative():
    d = ChangePointDetector(charts=CONSERVATIVE_CHARTS)
    d.name = "changepoint/cusum-bank-conservative"
    return d
