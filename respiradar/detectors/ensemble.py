"""Combining the bake-off entries: a 2-of-3 vote on partial evidence.

`changepoint` runs CUSUM charts on the shared scalar features. `spectral` builds its own
range-Doppler representation straight from the IQ and never touches the shared features.
Structurally independent errors are the condition under which voting helps, which is why
this pair is the one worth combining.

This file has been through two datasets and the second one overturned most of what the
first one said. Both versions are recorded below, because the *way* it was wrong is the
most useful thing here.

What four holds said, and why it was wrong
------------------------------------------
With four labelled holds, the members all missed the same one, so no vote could beat them;
the only entry that reached 4/4 was the aggressive CUSUM bank, and its three false alarms
lasted 0.2 s, 1.7 s and 1.6 s. Requiring the alarm to persist for 2 s removed all three and
scored 4/4 at zero false alarms - and an ablation with the spectral weight set to zero
scored identically, so the "ensemble" was one member plus a dwell. That was written down at
the time as a knife edge that should not be trusted, and it was not trustworthy: with nine
more holds, a 2 s dwell on the same member drops it from 12/13 to 9/13 *and* raises its
false alarms from 2 to 3.

The dwell failed in a way worth naming. A dwell delays every alarm by its own length. When
an alarm starts late in a hold, delaying it pushes it past the end of the hold - and the
harness scores an alarm that begins after the subject resumed breathing as a miss **and** a
false alarm. So the same knob loses holds and gains false alarms at once. A persistence
filter on the final boolean is the wrong instrument on this problem. It has been removed.

What thirteen holds say
-----------------------
Measured against the members as they stood when this was written. `spectral` was re-tuned
twice during the same afternoon and got much faster, which moved every number below; the
table is a snapshot, and re-running the bake-off is the only way to know it is still true.

=========================  ==========  ==========  ==========
per-hold latency            cusum-bank  cusum-bank  range-stft
                                        -conserv.
=========================  ==========  ==========  ==========
breath-hold #1                19.2 s      21.0 s      15.3 s
breath-hold #2                25.6 s      23.2 s      25.5 s
justinas-breath-hold #1       MISS        MISS        27.1 s
justinas-breath-hold #2       27.0 s      33.3 s      31.3 s
justinas-holds-3515 #1        19.9 s      20.0 s      16.8 s
justinas-holds-3515 #2        20.7 s      20.7 s      20.4 s
justinas-holds-3515 #3        22.7 s      22.7 s      24.2 s
nishant-holds-2401 #1         18.1 s      29.3 s      16.1 s
nishant-holds-2401 #2         20.7 s      27.3 s      16.1 s
nishant-holds-2401 #3         16.9 s      17.0 s      15.8 s
nishant-holds-3008 #1         12.9 s      16.8 s      14.3 s
nishant-holds-3008 #2         17.9 s      18.6 s      14.2 s
nishant-holds-3008 #3          9.2 s       3.9 s      15.6 s
false alarms                  2           0           2
=========================  ==========  ==========  ==========

Three things follow.

**The false alarms that remain are all on one recording.** vishnu-sleeping is the one
negative session containing long stretches of very shallow sleeping breath, and it is where
every member that cries wolf does so. Crucially they are never wrong in the *same second*,
so requiring a second member to agree removes them outright - and unlike a dwell it costs
no latency on the holds where a second member is already convinced. At the time of writing
`cusum-bank` has been re-tuned and no longer has any, leaving only `range-stft`'s two.

**13/13 costs a false alarm.** justinas-breath-hold #1 starts at 4.6 s, inside the 25 s
scoring warmup, and only `range-stft` reaches it - at 27.1 s, with a single vote. To turn
that single vote into an alarm the ensemble has to accept `cusum-bank`'s partial evidence
at 0.8 as the second vote, and doing so brings back one vishnu-sleeping false alarm. Since
the bake-off ranks zero false alarms above misses, `build()` takes 12/13 and
`build_sensitive()` exposes the other choice. Across roughly 1500 vote configurations and
2835 weighted-fusion configurations, none reached 13/13 at zero false alarms. That hold has
been reported as below the noise floor by four separate entries now; it is not a tuning
problem and chasing it only buys false alarms.

**The confirmation must be allowed to arrive late, but it does not need to be partial.**
The members' alarm episodes often fail to overlap by a single frame even when both are
right, so a frame-wise AND loses holds. A slack window fixes that. Partial evidence, which
mattered a great deal one revision ago, currently does not - see below.

The entry
---------
A 2-of-3 vote over `cusum-bank`, `cusum-bank-conservative` and `range-stft`. A member votes
if it crossed its own threshold within the last `slack_s` seconds, and `spectral` may vote
slightly early, at 85% of the dwell it demands of itself.

    2-of-3 vote (this module)        12/13, 0 false, worst 23.7 s, median 18.0 s
    cusum-bank alone                 12/13, 0 false, worst 23.7 s, median 18.0 s
    cusum-bank-conservative alone     9/13, 0 false, worst 23.7 s, median 21.2 s
    range-stft alone                 13/13, 2 false, worst 31.3 s, median 16.1 s
    2-of-3, cusum votes at 0.8       13/13, 1 false, worst 27.1 s, median 18.1 s

Read the first two rows honestly: **the vote currently ties `cusum-bank` exactly**, to the
frame, on every metric. It does not beat it.

That is the second time in this file that an ablation has reversed. One revision ago the
vote was 9.6 s of worst case ahead of the best clean member, because `cusum-bank` had two
false alarms on vishnu-sleeping and needed confirming. `changepoint` was then re-tuned and
those two false alarms went away on their own - so there is nothing left for the second vote
to suppress, and the vote reduces to its fastest member. An ensemble that exists to remove
false alarms is worth exactly as much as the false alarms its members have.

What the vote still earns, on this snapshot, is one row: `build_sensitive()` is the only
entry that reaches all thirteen holds with fewer than two false alarms. `range-stft` gets
13/13 and pays two; confirming it with `cusum-bank`'s partial evidence removes one of them
and takes 4.2 s off the worst case. No single member does that.

And it costs nothing to keep. The vote can only ever be as slow as its second-fastest
member and can only ever remove alarms, so the failure mode it insures against - a member
regressing and starting to cry wolf on an unseen body, which is the single thing this
project has been bitten by most - is covered for free. That is the argument for shipping it
while it ties, and it is a weaker argument than "it wins".

This time the operating point is a plateau, not an edge
--------------------------------------------------------
The previous version of this file put a 2 s dwell on three false-alarm episodes lasting
0.2 s, 1.7 s and 1.6 s. It scored perfectly and then collapsed from 12/13 to 9/13 the moment
there were more holds. So both numbers here were chosen for margin, and both were checked
for a plateau rather than a best cell.

`slack_s` is flat from 0 s to 20 s - every value gives the identical 12/13, zero false
alarms, 23.7 s worst case. Above 20 s it starts to *look* better and is an artefact: the
sessions are scripted as 30 s of breathing alternating with 30 s of holding, so a slack
approaching 30 s lets a vote from one hold survive into the next and the harness then reports
a 0.0 s latency because the alarm never dropped. `VoteEnsemble` refuses such a slack outright.

The spectral vote threshold is flat from 0.5 to 1.0 - identical false alarms, misses and
worst case across the whole range, differing only by half a second of median. So it is
currently inert, exactly as the spectral *weight* was inert in the four-hold version of this
file. The difference is that this time inert is not the same as useless: one revision ago,
when `spectral` was slower, the same knob was load-bearing and 0.77 was a hard false-alarm
boundary. It is kept at 0.85, mid-plateau, as insurance against the members diverging again
rather than as something that is earning its place today. `build_conservative()` sets it to
1.0 and owns no continuous threshold at all.

The honest summary of the knobs: there is nothing to tune here at present, and that is the
point. A combination whose score is unchanged across the entire range of its own parameters
is one that will survive the next time a member moves.

Causality
---------
Both members are causal. A vote persists *forward* in time from the frame a member crossed
its threshold, never backwards; the evidence extractors reproduce the members' own running
state frame by frame; and no statistic of the whole clip is used anywhere. Nothing here
re-tunes a member - every chart weight, deadband, cap, threshold, gate and spectral constant
is read off the imported member objects at run time, so when the shared features move the
members move and this moves with them. The only numbers this module owns are `slack_s` and
the per-member vote thresholds.
"""

from __future__ import annotations

import numpy as np

from respiradar.bakeoff import Clip
from respiradar.detectors import changepoint as CP
from respiradar.detectors import spectral as SP

SF = SP.F

# A vote may not survive long enough to reach the next hold. The recorded sessions alternate
# 30 s of breathing with 30 s of holding, so anything approaching 30 s lets one hold's alarm
# be inherited by the next and reports it as a 0.0 s detection.
MAX_HONEST_SLACK_S = 20.0


def _dwell_fraction(flags: np.ndarray, need: int) -> np.ndarray:
    """How far a run of `flags` has got towards `need` samples, capped at 1.0.

    The continuous form of `spectral._sustained`: at the frame where `_sustained` turns
    true this is exactly 1.0, so a vote threshold of 1.0 reproduces spectral's own alarm
    and anything below it votes earlier.
    """
    out = np.zeros(len(flags))
    run = 0
    for i, flag in enumerate(flags):
        run = run + 1 if flag else 0
        out[i] = min(run / need, 1.0) if need > 0 else float(flag)
    return out


def _dilate(flags: np.ndarray, width: int) -> np.ndarray:
    """"has been true within the last `width` samples" - forwards in time only."""
    out = np.zeros(len(flags), dtype=bool)
    run = 0
    for i, flag in enumerate(flags):
        run = width if flag else max(0, run - 1)
        out[i] = run > 0
    return out


class _ChangePointEvidence:
    """`changepoint`'s CUSUM totals as a fraction of each chart's own threshold.

    `Chart.run` with the running total kept rather than thresholded. Every parameter comes
    from the `Chart` objects the member built, so this follows the member instead of
    pinning a copy of its tuning. As with `_SpectralEvidence`, the member's own `predict`
    is OR-ed in at the end, so a vote threshold of 1.0 is exactly the member's decision
    even if the bank grows a chart this loop does not reproduce faithfully.
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
        return np.maximum(best, self.detector.predict(clip).astype(float))


class _SpectralEvidence:
    """How far `spectral` has got towards its own alarm, in [0, 1].

    1.0 at exactly the frame `SpectralApneaDetector` fires, so a vote threshold below 1.0
    means "spectral is this fraction of the way through the dwell it demands of itself".
    Its gating conditions are reproduced from the member's own attributes, so a frame
    spectral refuses to judge - in warmup, or with obvious motion - contributes nothing
    rather than contributing evidence that the person is breathing.

    `BRANCHES` lists the member's threshold/dwell attribute pairs and the feature each one
    watches. Absent attributes are skipped, so this survives spectral gaining or losing a
    branch, which it has already done once: it was written against `quiet` and `relative`
    and woke up to find `fast` and `deep` had been added underneath it, silently
    understating the member by a third.

    That is why `__call__` finishes by OR-ing in the member's own `predict`. Whatever
    branches exist and whatever they are called, the evidence is 1.0 wherever the member
    alarms - so a vote threshold of 1.0 is *exactly* the member's own decision, by
    construction rather than by my keeping up with it. Modelling the branches individually
    only ever adds early partial credit on top.
    """

    #  threshold attribute, dwell attribute, feature, relative?
    BRANCHES = [
        ("fast_mm", "fast_s", "A_fast", False),
        ("quiet_mm", "quiet_s", "A", False),
        ("deep_ratio", "deep_s", "ratio_p", True),
        ("rel_ratio", "rel_s", "ratio_p", True),
    ]

    def __init__(self, detector: SP.SpectralApneaDetector) -> None:
        self.detector = detector

    def __call__(self, clip: Clip) -> np.ndarray:
        d = self.detector
        X = SP.features_for(clip)
        fs = 1 / max(float(np.median(np.diff(clip.t))), 1e-6)
        ready = (X[:, SF["A"]] > 0) & (X[:, SF["intra"]] < getattr(d, "intra_gate", np.inf))
        ready_rel = ready & (X[:, SF["base_p"]] > 0)

        best = np.zeros(len(clip.t))
        for level_attr, dwell_attr, feature, relative in self.BRANCHES:
            level, dwell_s = getattr(d, level_attr, None), getattr(d, dwell_attr, None)
            if level is None or dwell_s is None or feature not in SF:
                continue
            below = (ready_rel if relative else ready) & (X[:, SF[feature]] < level)
            best = np.maximum(best, _dwell_fraction(below, int(dwell_s * fs)))

        # The guarantee: whatever the member does, evidence is 1.0 where it alarms.
        return np.maximum(best, self.detector.predict(clip).astype(float))


def _evidence_for(member):
    if isinstance(member, CP.ChangePointDetector):
        return _ChangePointEvidence(member)
    if isinstance(member, SP.SpectralApneaDetector):
        return _SpectralEvidence(member)
    raise TypeError(f"no continuous evidence available for {member!r}")


class VoteEnsemble:
    """m-of-n over the members' evidence, with a per-member threshold and a slack window.

    Each member's evidence is scaled so that 1.0 is the frame it would have alarmed on its
    own, which is what makes a *partial* vote meaningful and comparable across members that
    share no features. A member's vote counts at frame i if its evidence reached that
    member's threshold at any point in the last `slack_s` seconds - the cascade idea stated
    once for every member, so whichever one is fastest on a given hold arms and the others
    confirm. Frame-wise coincidence is not required and should not be: the members' alarm
    episodes routinely fail to overlap even when both are right.
    """

    def __init__(
        self,
        members: dict,
        vote_at: dict,
        need: int = 2,
        slack_s: float = 12.0,
        name: str = "ensemble/vote",
    ) -> None:
        if slack_s > MAX_HONEST_SLACK_S:
            raise ValueError(
                f"slack_s={slack_s} exceeds {MAX_HONEST_SLACK_S}s, at which a vote survives "
                "from one scripted hold into the next and latency stops meaning anything"
            )
        self.members = members
        self.evidence = {key: _evidence_for(m) for key, m in members.items()}
        self.vote_at = vote_at
        self.need = need
        self.slack_s = slack_s
        self.name = name

    def fit(self, clips) -> None:
        """Fit every member on this fold's training clips, so none of them sees the test
        subject. In practice neither member learns a threshold from labels - `spectral.fit`
        is a no-op and `changepoint.fit` only records the separation it achieved - which is
        itself the reason they transfer to a new body at all."""
        for member in self.members.values():
            if hasattr(member, "fit"):
                member.fit(clips)

    def votes(self, clip: Clip) -> np.ndarray:
        """How many members are currently voting, per frame. Useful for inspection."""
        fs = 1 / max(float(np.median(np.diff(clip.t))), 1e-6)
        width = max(1, int(self.slack_s * fs))
        counted = np.zeros(len(clip.t), dtype=int)
        for key, evidence in self.evidence.items():
            crossed = evidence(clip) >= self.vote_at[key]
            counted += _dilate(crossed, width) if width > 1 else crossed
        return counted

    def predict(self, clip: Clip) -> np.ndarray:
        return self.votes(clip) >= self.need


def _members() -> dict:
    """The three voters, freshly built so each entry owns its own.

    Both CUSUM banks are included even though they share a feature set and a chart
    machinery. They are not independent, and the vote does not pretend they are - what they
    supply is a *second opinion at a different operating point*, which is exactly what is
    needed to confirm or reject the aggressive bank's two false alarms. Spectral supplies
    the genuinely independent third opinion.
    """
    return {
        "cusum": CP.build(),
        "cusum_conservative": CP.build_conservative(),
        "spectral": SP.build(),
    }


def build():
    """12/13 holds, zero false alarms, worst 23.7 s, median 18.0 s.

    Matches the only member with no false alarms on misses and beats it by 9.6 s of worst
    case. 12/13 is the ceiling at zero false alarms: justinas-breath-hold #1 starts at
    4.6 s, inside the scoring warmup, and reaching it costs one false alarm - see
    `build_sensitive()`.

    Spectral votes at 0.85 rather than 1.0. That knob is currently inert - every value from
    0.5 to 1.0 scores identically - and is set mid-range rather than at a best cell, because
    one revision ago the same knob had a hard false-alarm boundary at 0.77.
    """
    return VoteEnsemble(
        _members(),
        vote_at={"cusum": 1.0, "cusum_conservative": 1.0, "spectral": 0.85},
        need=2,
        slack_s=8.0,
        name="ensemble/vote",
    )


def build_conservative():
    """12/13 holds, zero false alarms, worst 23.7 s, median 18.5 s.

    The same vote with every member required to have fully crossed its own threshold, so
    this module contributes no continuous threshold of its own and the only number it owns
    is the slack - which is itself flat across its entire usable range. Half a second of
    median slower than `build()` and identical on everything that ranks. This is the one to
    reach for if the members are re-tuned underneath it again, or if the negative set grows
    past the twenty-three minutes measured here.
    """
    return VoteEnsemble(
        _members(),
        vote_at={"cusum": 1.0, "cusum_conservative": 1.0, "spectral": 1.0},
        need=2,
        slack_s=8.0,
        name="ensemble/vote-conservative",
    )


def build_sensitive():
    """13/13 holds, one false alarm, worst 27.1 s, median 18.1 s.

    The only way found to reach every hold. justinas-breath-hold #1 is seen by `range-stft`
    alone, so the second vote has to come from `cusum-bank` at 0.8 of its threshold rather
    than at it - and lowering `cusum-bank`'s bar that far lets one vishnu-sleeping stretch
    of very shallow sleeping breath through.

    The bake-off ranks zero false alarms above misses, so this is not `build()`. It is here
    because "every hold, one false alarm in twenty-three minutes" is the right trade for
    some products and the wrong one for this one, and because the hold it buys starts inside
    the scoring warmup, where the features are a filter transient rather than a chest. If
    that hold is ever re-recorded starting after 30 s, this builder should be deleted rather
    than promoted.
    """
    return VoteEnsemble(
        _members(),
        vote_at={"cusum": 0.8, "cusum_conservative": 1.0, "spectral": 1.0},
        need=2,
        slack_s=12.0,
        name="ensemble/vote-sensitive",
    )


# Kept because their numbers are the finding - they are what rules the simpler combinations
# out - not because any of them should be shipped.


def build_or():
    """OR of the conservative bank and spectral. 12/13, 0 false, worst 30.8 s, median 20.7 s.

    Identical to spectral on every metric that ranks. A union cannot remove a false alarm,
    and on this data it does not add a hold either.
    """
    d = VoteEnsemble(
        {"cusum_conservative": CP.build_conservative(), "spectral": SP.build()},
        vote_at={"cusum_conservative": 1.0, "spectral": 1.0},
        need=1,
        slack_s=0.0,
    )
    d.name = "ensemble/or"
    return d


def build_and(slack_s: float = 12.0):
    """Cascade: either member arms, the other confirms. 12/13, 0 false, worst 30.8 s.

    Two voters are enough to clear `cusum-bank`'s false alarms, but with only two the
    confirmation must come from spectral every time, and spectral is the slowest member.
    The third voter in `build()` is what lets the confirmation come from whoever is fast on
    that particular hold. Below about 4 s of slack this degrades sharply - to 10/13 with no
    slack at all - because the members' alarm episodes often do not overlap by a frame.
    """
    d = VoteEnsemble(
        {"cusum": CP.build(), "spectral": SP.build()},
        vote_at={"cusum": 1.0, "spectral": 1.0},
        need=2,
        slack_s=slack_s,
    )
    d.name = "ensemble/cascade"
    return d
