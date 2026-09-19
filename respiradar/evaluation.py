"""Scoring apnea detectors against labelled sessions.

Every approach in the bake-off is judged here, on the same rules, so the numbers are
comparable. A detector is reduced to one boolean per frame - did it alarm? - and the rest
is bookkeeping.

The two rules that matter:

- An alarm counts as detecting a hold only if it fires *during* the hold. Alarming after the
  person started breathing again is a false alarm, not a late catch. The existing pipeline
  scores 0/2 under this rule even though it does alarm once in the session.
- Latency is measured from the onset of the hold, because that is the number the product
  lives or dies on: how long someone stops breathing before anyone is told.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Filters, baselines and presence tracking all need time to settle. Alarms inside this
# window are a known start-up artefact and are excluded from scoring entirely.
DEFAULT_WARMUP_S = 25.0


@dataclass(frozen=True)
class Episode:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def contains(self, t: float) -> bool:
        return self.start_s <= t < self.end_s


@dataclass
class EvaluationResult:
    detected: int = 0
    missed: int = 0
    latencies_s: list[float] = field(default_factory=list)
    false_alarms: int = 0
    false_alarm_s: float = 0.0
    negative_s: float = 0.0

    @property
    def total_holds(self) -> int:
        return self.detected + self.missed

    @property
    def worst_latency_s(self) -> float | None:
        return max(self.latencies_s) if self.latencies_s else None

    @property
    def median_latency_s(self) -> float | None:
        return float(np.median(self.latencies_s)) if self.latencies_s else None

    @property
    def false_alarms_per_hour(self) -> float:
        if self.negative_s <= 0:
            return 0.0
        return self.false_alarms * 3600 / self.negative_s

    def __str__(self) -> str:
        latency = "n/a" if self.worst_latency_s is None else f"{self.worst_latency_s:.1f}s"
        return (
            f"{self.detected}/{self.total_holds} detected, worst latency {latency}, "
            f"{self.false_alarms} false alarms in {self.negative_s/60:.1f} min "
            f"({self.false_alarms_per_hour:.1f}/h)"
        )


def _episodes(t: np.ndarray, alarms: np.ndarray) -> list[tuple[float, float]]:
    """Contiguous runs of True, as (start_time, end_time)."""
    if not len(t):
        return []
    flags = np.asarray(alarms, dtype=bool).astype(np.int8)
    edges = np.diff(np.concatenate(([0], flags, [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.05
    spans = []
    for s, e in zip(starts, ends):
        start_t = float(t[s])
        end_t = float(t[e]) if e < len(t) else float(t[-1]) + dt
        spans.append((start_t, end_t))
    return spans


def evaluate_alarms(
    t: np.ndarray,
    alarms: np.ndarray,
    holds: list[Episode],
    warmup_s: float = DEFAULT_WARMUP_S,
) -> EvaluationResult:
    """Score one session's per-frame alarm flags against its labelled holds."""
    t = np.asarray(t, dtype=float)
    alarms = np.asarray(alarms, dtype=bool)
    result = EvaluationResult()

    scored = t >= warmup_s
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.05

    in_hold = np.zeros(len(t), dtype=bool)
    for hold in holds:
        in_hold |= (t >= hold.start_s) & (t < hold.end_s)
    result.negative_s = float(np.count_nonzero(scored & ~in_hold) * dt)

    for hold in holds:
        window = (t >= hold.start_s) & (t < hold.end_s) & scored
        fired = np.where(window & alarms)[0]
        if len(fired):
            result.detected += 1
            result.latencies_s.append(float(t[fired[0]] - hold.start_s))
        else:
            result.missed += 1

    # A run of alarm frames is a false alarm unless some part of it lands inside a hold.
    for start_t, end_t in _episodes(t, alarms & scored):
        overlaps = any(
            start_t < hold.end_s and end_t > hold.start_s for hold in holds
        )
        if not overlaps:
            result.false_alarms += 1
            result.false_alarm_s += end_t - start_t

    return result
