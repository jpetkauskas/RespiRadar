import numpy as np
import pytest

from respiradar.evaluation import Episode, evaluate_alarms


def _alarms(n, spans, fs=20.0):
    t = np.arange(n) / fs
    a = np.zeros(n, dtype=bool)
    for lo, hi in spans:
        a |= (t >= lo) & (t < hi)
    return t, a


def test_reports_no_detection_when_the_alarm_never_fires():
    t, alarms = _alarms(4000, [])
    result = evaluate_alarms(t, alarms, holds=[Episode(20.0, 50.0)], warmup_s=0.0)

    assert result.detected == 0
    assert result.missed == 1
    assert result.latencies_s == []


def test_measures_latency_from_the_onset_of_the_hold():
    t, alarms = _alarms(4000, [(32.0, 55.0)])
    result = evaluate_alarms(t, alarms, holds=[Episode(20.0, 50.0)], warmup_s=0.0)

    assert result.detected == 1
    assert result.latencies_s[0] == pytest.approx(12.0, abs=0.1)


def test_an_alarm_only_after_the_hold_ended_is_not_a_detection():
    """Alarming after the person resumed breathing is a false alarm, not a late catch."""
    t, alarms = _alarms(4000, [(52.0, 60.0)])
    result = evaluate_alarms(t, alarms, holds=[Episode(20.0, 50.0)], warmup_s=0.0)

    assert result.detected == 0
    assert result.missed == 1
    assert result.false_alarms == 1


def test_counts_false_alarms_outside_any_hold():
    t, alarms = _alarms(4000, [(5.0, 8.0), (100.0, 104.0)])
    result = evaluate_alarms(t, alarms, holds=[Episode(20.0, 50.0)], warmup_s=0.0)

    assert result.false_alarms == 2
    assert result.false_alarm_s == pytest.approx(7.0, abs=0.2)


def test_ignores_everything_inside_the_warmup_period():
    """Filters need time to settle; alarms there are a known artefact, not a result."""
    t, alarms = _alarms(4000, [(5.0, 8.0)])
    result = evaluate_alarms(t, alarms, holds=[], warmup_s=25.0)

    assert result.false_alarms == 0


def test_negative_seconds_exclude_holds_and_warmup():
    t, alarms = _alarms(4000, [])  # 200 s at 20 Hz
    result = evaluate_alarms(t, alarms, holds=[Episode(20.0, 50.0)], warmup_s=25.0)

    # 200 total - 25 warmup - 25 of the hold that falls after warmup
    assert result.negative_s == pytest.approx(150.0, abs=0.5)
