"""The contract every apnea detector is measured against, and the runner that scores them.

Write a detector, register it, run `python -m respiradar.bakeoff`. Everyone gets the same
features, the same folds and the same metrics.

Evaluation protocol
-------------------
There are only two labelled holds, so there is no honest way to do a large cross-validation.
Instead there are two folds, and each hold is tested by a detector that never saw it:

    fold A: fit on hold 2 (+ first half of the negative sessions), test on hold 1
    fold B: fit on hold 1 (+ second half of the negative sessions), test on hold 2

A detector that needs no fitting scores the same either way, which is itself informative.
Negatives are split by time so that a fitted detector cannot memorise the exact minute it
will be tested on.

What counts as good
-------------------
1. Zero false alarms. Eight minutes of talking, moving and sleeping is the negative set.
2. Both holds detected, while they are still happening.
3. Lowest worst-case latency from hold onset.

In that order. A detector that catches both holds in 8 s but cries wolf once during the
noisy session is worse than one that takes 15 s and never does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from respiradar.dataset import SESSIONS, load_cached, session_by_name
from respiradar.evaluation import DEFAULT_WARMUP_S, Episode, EvaluationResult, evaluate_alarms


@dataclass
class Clip:
    """A time slice of one session: features, labels and the holds inside it."""

    name: str
    t: np.ndarray
    X: np.ndarray
    y: np.ndarray
    holds: list[Episode]


class Detector(Protocol):
    name: str

    def fit(self, clips: Sequence[Clip]) -> None:
        """Optional. Detectors that need no training may do nothing here."""

    def predict(self, clip: Clip) -> np.ndarray:
        """One boolean per frame. Must depend only on the past - see test_features_are_causal."""


def _clip(name: str, lo: float, hi: float) -> Clip:
    session = session_by_name(name)
    t, X, y = load_cached(name)
    mask = (t >= lo) & (t < hi)
    holds = [h for h in session.holds if h.start_s >= lo and h.end_s <= hi]
    return Clip(f"{name}[{lo:.0f}-{hi:.0f}s]", t[mask], X[mask], y[mask], holds)


def folds() -> list[tuple[list[Clip], list[Clip]]]:
    """[(train_clips, test_clips), ...] - one fold per labelled hold."""
    hold_1, hold_2 = session_by_name("breath-hold").holds
    # Split the hold session between the two holds, and the negative sessions in half.
    split = (hold_1.end_s + hold_2.start_s) / 2
    early = _clip("breath-hold", 0.0, split)
    late = _clip("breath-hold", split, 1e9)
    negatives = [("sleeping", 185.0), ("noisy", 185.0)]

    fold_a = (
        [late] + [_clip(n, 0.0, d / 2) for n, d in negatives],
        [early] + [_clip(n, d / 2, 1e9) for n, d in negatives],
    )
    fold_b = (
        [early] + [_clip(n, d / 2, 1e9) for n, d in negatives],
        [late] + [_clip(n, 0.0, d / 2) for n, d in negatives],
    )
    return [fold_a, fold_b]


def score(detector: Detector, warmup_s: float = DEFAULT_WARMUP_S) -> EvaluationResult:
    """Fit and test the detector on every fold, summing the results."""
    total = EvaluationResult()
    for train, test in folds():
        if hasattr(detector, "fit"):
            detector.fit(train)
        for clip in test:
            # Each clip starts partway through the session, so warmup is relative to it.
            result = evaluate_alarms(
                clip.t, detector.predict(clip), clip.holds, warmup_s=clip.t[0] + warmup_s
            )
            total.detected += result.detected
            total.missed += result.missed
            total.latencies_s += result.latencies_s
            total.false_alarms += result.false_alarms
            total.false_alarm_s += result.false_alarm_s
            total.negative_s += result.negative_s
    return total


def full_session_check(detector: Detector) -> dict[str, EvaluationResult]:
    """Sanity pass: fit on everything, run each whole session. Optimistic by design."""
    clips = [_clip(s.name, 0.0, 1e9) for s in SESSIONS]
    if hasattr(detector, "fit"):
        detector.fit(clips)
    return {
        clip.name: evaluate_alarms(clip.t, detector.predict(clip), clip.holds)
        for clip in clips
    }


def report(detectors: Sequence[Detector]) -> None:
    rows = []
    for detector in detectors:
        try:
            result = score(detector)
        except Exception as exc:  # a broken entry must not sink the whole table
            print(f"{detector.name}: FAILED ({type(exc).__name__}: {exc})")
            continue
        rows.append((detector.name, result))

    # Zero false alarms first, then worst-case latency. That is the product's priority order.
    rows.sort(key=lambda r: (r[1].false_alarms, r[1].missed, r[1].worst_latency_s or 1e9))

    header = f"{'detector':<26} {'holds':>7} {'worst lat':>10} {'med lat':>9} {'false':>7} {'neg min':>8}"
    print(header)
    print("-" * len(header))
    for name, r in rows:
        worst = "-" if r.worst_latency_s is None else f"{r.worst_latency_s:.1f}s"
        med = "-" if r.median_latency_s is None else f"{r.median_latency_s:.1f}s"
        print(
            f"{name:<26} {r.detected}/{r.total_holds:<5} {worst:>10} {med:>9} "
            f"{r.false_alarms:>7} {r.negative_s/60:>8.1f}"
        )
