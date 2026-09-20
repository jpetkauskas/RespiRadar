"""Combining the bake-off entries, and the ablation that says how much the combining is worth.

`changepoint` runs CUSUM charts on the shared scalar features. `spectral` builds its own
range-Doppler representation straight from the IQ and never touches the shared features.
Structurally independent errors are the condition under which voting helps, so the pair is
the obvious candidate for an ensemble. This module builds every combination of them worth
building, and then measures what the second member is actually contributing.

What the members do, per hold
-----------------------------
=========================  ===========  ===========  ===========  ===========  =====
hold                       nishant #1   nishant #2   justinas #1  justinas #2  false
=========================  ===========  ===========  ===========  ===========  =====
cusum-bank                  15.3 s       23.5 s       26.2 s       17.5 s       3
cusum-bank-conservative     19.1 s       22.0 s       MISS         17.4 s       0
range-stft                  25.6 s       30.1 s       MISS         30.8 s       0
one-sided-maha              MISS         26.6 s       MISS         33.1 s       2
=========================  ===========  ===========  ===========  ===========  =====

Two facts in that table decide everything below.

**The members miss the same hold.** The brief expected `spectral` and
`cusum-bank-conservative` to be complementary. They are not: both miss justinas #1, the
hold that starts at 4.6 s, inside the 25 s scoring warmup. Only the aggressive `cusum-bank`
ever catches it, and it catches it with a single short blip. So no boolean vote of any
m-of-n can reach 4/4 - exactly one voter ever votes for that hold, and a majority of one is
that voter. OR(conservative, spectral) is byte-for-byte the conservative bank;
AND/cascade(either bank, spectral) is byte-for-byte spectral.

**Spectral is quieter at the false alarms than inside the real hold.** The asymmetric idea -
aggressive changepoint gated by spectral's absolute millimetre floor - is backwards on this
data. At the three frames `cusum-bank` cries wolf on vishnu-sleeping, against the frames of
justinas #1 that only it catches:

    justinas #1 @ 28-30 s    A = 1.2-1.6 mm   ratio_p = 1.00 (spectral still in warmup)
    vishnu FA   @ 131 s      A = 0.71 mm      ratio_p = 0.02
    vishnu FA   @ 147 s      A = 0.30 mm      ratio_p = 0.01
    vishnu FA   @ 180 s      A = 2.7 mm       ratio_p = 0.62

A quiet-gate confirms two of the three false alarms and vetoes the detection. Those vishnu
stretches look like genuinely very shallow sleeping breath, which is a labelling problem
rather than a fusion problem.

What does work, and the honest ablation
---------------------------------------
Boolean voting can only remove alarms, and the zero-false-alarm members have none to
remove. The room is in *latency* and in the three blips: both members are usually partly
convinced before either crosses its own threshold. So the entry here is a **weighted score
fusion**, not a boolean vote:

    S(t) = max_over_charts(cusum_total / chart_threshold) + w * spectral_progress(t)

`spectral_progress` is spectral's own dwell counter as a fraction of the dwell it requires:
how far along it is towards its own alarm, reaching exactly 1.0 at the frame spectral would
fire. With w = 1 and threshold 1 either member alone still fires, so the fusion contains
their OR; the cross terms are the only thing it adds. A dwell on S itself buys back the
persistence that a lowered effective threshold gives away.

That reaches **4/4 holds with zero false alarms**, which no single entry had. But the
ablation is unkind. Setting w = 0 - deleting spectral entirely and keeping only the dwell -
scores *identically*:

    aggressive bank, w = 0.5, threshold 1.0, dwell 2 s   4/4, 0 false, worst 28.1 s
    aggressive bank, w = 0.0, threshold 1.0, dwell 2 s   4/4, 0 false, worst 28.1 s

So at the headline operating point **`spectral` contributes nothing**. All of the gain is a
2 s persistence requirement on `cusum-bank`, whose three false alarms last 0.2 s, 1.7 s and
1.6 s. That is a post-filter on one member, not an ensemble, and it is a knife edge: 1 s of
dwell leaves two false alarms, 3 s leaves one, 5 s drops to 2/4. Three false alarms is not
enough evidence to place a threshold between 1.7 s and 2.0 s and expect it to hold on a
fourth body.

`build_conservative()` is where the second member earns its place. At a longer dwell and a
higher threshold - more margin against a negative set that has not been recorded yet -
changepoint alone loses a hold and gains a false alarm, and spectral's partial evidence is
what carries the hold across:

    aggressive bank, w = 0.5, threshold 1.2, dwell 4 s   4/4, 0 false, worst 30.1 s
    aggressive bank, w = 0.0, threshold 1.2, dwell 4 s   3/4, 1 false, worst 30.7 s

That is a real ensemble gain, on one fold, worth roughly one recording. Treat it as a
hypothesis, not a result.

Causality
---------
Both members are causal. The fusion is a pointwise sum of their running state plus a
backward-looking dwell and a hysteresis latch; no statistic of the whole clip is used
anywhere. Nothing here re-tunes a member: every chart weight, deadband, cap, threshold,
gate and spectral constant is read off the imported member objects at run time, so when the
shared features change underneath, the members move and this moves with them. The only
numbers owned by this module are the fusion weight, its threshold and its dwell.
"""

from __future__ import annotations

import numpy as np

from respiradar.bakeoff import Clip
from respiradar.detectors import anomaly as A
from respiradar.detectors import changepoint as CP
from respiradar.detectors import spectral as SP

SF = SP.F


def _dwell_fraction(flags: np.ndarray, need: int) -> np.ndarray:
    """How far a run of `flags` has got towards `need` samples, capped at 1.0.

    The continuous form of `spectral._sustained`: at the frame where `_sustained` turns
    true this is exactly 1.0, so a fusion weight of 1 reproduces spectral's own alarm and
    anything above it fires earlier.
    """
    out = np.zeros(len(flags))
    run = 0
    for i, flag in enumerate(flags):
        run = run + 1 if flag else 0
        out[i] = min(run / need, 1.0) if need > 0 else float(flag)
    return out


def _dilate(alarm: np.ndarray, width: int) -> np.ndarray:
    """"has fired within the last `width` samples" - a causal dilation, never a future one."""
    out = np.zeros(len(alarm), dtype=bool)
    run = 0
    for i, flag in enumerate(alarm):
        run = width if flag else max(0, run - 1)
        out[i] = run > 0
    return out


def _hysteresis(trigger: np.ndarray, recovered: np.ndarray) -> np.ndarray:
    """Latch on `trigger`, release on `recovered`.

    The same rule spectral uses, for the same reason: without it the quiet tail of a long
    hold breaks into several alarm episodes and every one after the first scores as a false
    alarm.
    """
    alarms = np.zeros(len(trigger), dtype=bool)
    on = False
    for i in range(len(trigger)):
        if trigger[i]:
            on = True
        elif on and recovered[i]:
            on = False
        alarms[i] = on
    return alarms


class _ChangePointEvidence:
    """`changepoint`'s CUSUM totals as a fraction of each chart's own threshold.

    `Chart.run` with the running total kept instead of thresholded. Every parameter comes
    from the `Chart` objects the member built, so this follows the member rather than
    pinning a copy of its tuning.
    """

    def __init__(self, detector: CP.ChangePointDetector) -> None:
        self.detector = detector

    def __call__(self, clip: Clip) -> np.ndarray:
        best = np.zeros(len(clip.t))
        for chart in self.detector.charts:
            ctx = self.detector._context(clip, chart)
            statistic = sum(w * ctx.drops[n] for n, w in chart.weights.items() if w)
            increment = np.minimum(statistic - chart.deadband, chart.cap)
            live = ctx.usable & ctx.calm & (clip.t >= clip.t[0] + self.detector.warm_s)
            hold_off = int(chart.refractory_s * ctx.fs)
            blocked_until, total = -1, 0.0
            trace = np.zeros(len(clip.t))
            for i in range(len(clip.t)):
                if not live[i]:
                    total = 0.0
                    blocked_until = i + hold_off
                    continue
                total = max(0.0, total * chart.decay + increment[i])
                if i > blocked_until:
                    trace[i] = total / chart.threshold
            best = np.maximum(best, trace)
        return best


class _SpectralEvidence:
    """How far `spectral` has got towards its own alarm, in [0, 1].

    1.0 at exactly the frame `SpectralApneaDetector` would fire. The gating conditions are
    reproduced from the member's own attributes, so a frame spectral refuses to judge - in
    warmup, or with obvious motion - contributes nothing rather than contributing evidence
    that the person is breathing.
    """

    def __init__(self, detector: SP.SpectralApneaDetector) -> None:
        self.detector = detector

    def __call__(self, clip: Clip) -> np.ndarray:
        d = self.detector
        X = SP.features_for(clip)
        fs = 1 / max(float(np.median(np.diff(clip.t))), 1e-6)
        a, ratio = X[:, SF["A"]], X[:, SF["ratio_p"]]
        ready = (a > 0) & (X[:, SF["base_p"]] > 0) & (X[:, SF["intra"]] < d.intra_gate)
        quiet = _dwell_fraction(ready & (a < d.quiet_mm), int(d.quiet_s * fs))
        relative = _dwell_fraction(ready & (ratio < d.rel_ratio), int(d.rel_s * fs))
        return np.maximum(quiet, relative)


def _evidence_for(member):
    if isinstance(member, CP.ChangePointDetector):
        return _ChangePointEvidence(member)
    if isinstance(member, SP.SpectralApneaDetector):
        return _SpectralEvidence(member)
    raise TypeError(f"no continuous evidence available for {member!r}")


class _Combination:
    """Shared plumbing: hold the members, fit each of them on the training clips.

    Fitting every member on the fold's training clips is what keeps the test subject unseen.
    In practice neither member learns a threshold from labels - `spectral.fit` is a no-op
    and `changepoint.fit` only records the separation it achieved - so the fold makes no
    difference to what they predict, which is itself the reason they transfer.
    """

    def __init__(self, members: dict) -> None:
        self.members = members

    def fit(self, clips) -> None:
        for member in self.members.values():
            if hasattr(member, "fit"):
                member.fit(clips)


class BooleanEnsemble(_Combination):
    """m-of-n over the members' boolean alarms, with slack so the votes need not coincide.

    `slack_s` is what makes this more than a frame-wise AND, and it is the cascade idea
    stated once for all members: a vote counts at frame i if that member alarmed at any
    point in the last `slack_s` seconds, so whoever fires first arms and the others confirm.
    A plain AND of `cusum-bank` and `range-stft` detects nothing at all on nishant #1,
    because changepoint's episode (29.5-39.2 s) and spectral's (39.9-51.5 s) do not overlap
    by a single frame.
    """

    def __init__(self, members: dict, need: int, slack_s: float = 0.0, dwell_s: float = 0.0):
        super().__init__(members)
        self.need = need
        self.slack_s = slack_s
        self.dwell_s = dwell_s
        self.name = f"ensemble/{need}-of-{len(members)}"

    def predict(self, clip: Clip) -> np.ndarray:
        fs = 1 / max(float(np.median(np.diff(clip.t))), 1e-6)
        width = max(1, int(self.slack_s * fs))
        votes = np.zeros(len(clip.t), dtype=int)
        for member in self.members.values():
            alarm = np.asarray(member.predict(clip), dtype=bool)
            votes += _dilate(alarm, width) if width > 1 else alarm
        passed = votes >= self.need
        if self.dwell_s > 0:
            passed = _dwell_fraction(passed, int(self.dwell_s * fs)) >= 1.0
        return passed


class FusionEnsemble(_Combination):
    """Weighted sum of the members' partial evidence, thresholded with a dwell.

    Each member's evidence is scaled so that 1.0 is the frame it would have alarmed on its
    own. With unit weights and `threshold` 1.0 the fusion is therefore a superset of the
    members' OR up to the dwell, and everything it adds is a frame where two members were
    each partly, but individually insufficiently, convinced.
    """

    def __init__(
        self,
        members: dict,
        weights: dict,
        threshold: float = 1.0,
        dwell_s: float = 0.0,
        release: float = 0.5,
        name: str = "ensemble/score-fusion",
    ) -> None:
        super().__init__(members)
        self.evidence = {key: _evidence_for(m) for key, m in members.items()}
        self.weights = weights
        self.threshold = threshold
        self.dwell_s = dwell_s
        self.release = release
        self.name = name

    def score(self, clip: Clip) -> np.ndarray:
        """The fused evidence, one value per frame. Continuous, causal, unnormalised."""
        total = np.zeros(len(clip.t))
        for key, weight in self.weights.items():
            if weight:
                total += weight * self.evidence[key](clip)
        return total

    def predict(self, clip: Clip) -> np.ndarray:
        fs = 1 / max(float(np.median(np.diff(clip.t))), 1e-6)
        score = self.score(clip)
        trigger = score >= self.threshold
        if self.dwell_s > 0:
            trigger = _dwell_fraction(trigger, int(self.dwell_s * fs)) >= 1.0
        return _hysteresis(trigger, score < self.release * self.threshold)


def _members(aggressive: bool = True) -> dict:
    """The two structurally independent members, freshly built so each entry owns its own."""
    changepoint = CP.build() if aggressive else CP.build_conservative()
    return {"changepoint": changepoint, "spectral": SP.build()}


def build():
    """4/4 holds, zero false alarms, worst 28.1 s, median 22.4 s.

    The best-scoring configuration found, and the first entry with both all four holds and
    no false alarms. Read the ablation in the module docstring before believing in it: an
    identical score is reached with the spectral weight set to zero, so on today's data this
    is `cusum-bank` plus a 2 s persistence filter and the second member is inert. It is
    kept in the sum because it costs nothing here and is what rescues the longer-dwell
    operating point below, but the honest description of this entry is "one member, with a
    dwell", and the dwell sits on three false alarms of 0.2 s, 1.7 s and 1.6 s.
    """
    return FusionEnsemble(
        _members(),
        weights={"changepoint": 1.0, "spectral": 0.5},
        threshold=1.0,
        dwell_s=2.0,
        release=0.5,
        name="ensemble/fusion",
    )


def build_conservative():
    """4/4 holds, zero false alarms, worst 30.1 s, median 26.2 s - with margin to spare.

    Two seconds slower in the worst case, in exchange for a 4 s persistence requirement and
    a 20% higher threshold, which is the variant to prefer if the negative set ever grows
    past nineteen minutes. It is also the only operating point where the ensemble is doing
    real work: with the spectral weight zeroed, the same threshold and dwell score 3/4 with
    one false alarm, so spectral's partial evidence is what carries the fourth hold over the
    longer dwell. One fold, one hold - a hypothesis to re-test when there are more subjects.
    """
    return FusionEnsemble(
        _members(),
        weights={"changepoint": 1.0, "spectral": 0.5},
        threshold=1.2,
        dwell_s=4.0,
        release=0.5,
        name="ensemble/fusion-conservative",
    )


# The combinations below are kept because their *numbers* are the finding - they are what
# rules the boolean ensemble out - not because any of them should be shipped.


def build_or():
    """OR of the two members. Identical to the conservative bank: 3/4, 0 false, 22.0 s."""
    d = BooleanEnsemble(_members(aggressive=False), need=1)
    d.name = "ensemble/or"
    return d


def build_and(slack_s: float = 20.0):
    """Cascade: either member arms, the other confirms within `slack_s`.

    Identical to spectral alone at every slack from 10 s to 30 s: 3/4, 0 false, 30.8 s.
    Spectral is always the later of the two, so the cascade always waits for it.
    """
    d = BooleanEnsemble(_members(aggressive=False), need=2, slack_s=slack_s)
    d.name = "ensemble/cascade"
    return d


def build_vote():
    """2-of-3 over changepoint, spectral and anomaly, with slack. 3/4, 30.8 s, 1 false alarm.

    Adding a third voter makes it worse. `anomaly` fires later than spectral on both holds
    it catches and never fires on justinas #1 either, so the second vote is still
    spectral's and the detections are unchanged - but anomaly's own vishnu-sleeping alarms
    now land within slack of changepoint's, and two wrong votes are a false alarm. A vote
    is only a safeguard when the voters' *errors* are independent, and on the one negative
    recording that matters they are not.
    """
    members = {"changepoint": CP.build(), "spectral": SP.build(), "anomaly": A.build()}
    d = BooleanEnsemble(members, need=2, slack_s=20.0)
    d.name = "ensemble/2-of-3"
    return d
