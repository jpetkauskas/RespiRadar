import itertools

import h5py
import numpy as np
import pytest

from respiradar.breathing import AppState, BreathingPipeline
from respiradar.sources import RadarConfig, recorded_config, replay_frames, simulated_frames


def _run_recording(path):
    pipeline = BreathingPipeline(recorded_config(path))
    return [pipeline.process(frame) for frame in replay_frames(path)]


def _settled_rates(results, tail=200):
    rates = [r.rate_bpm for r in results if r.rate_bpm is not None]
    return np.array(rates[-tail:])


def test_matches_acconeer_rate_on_the_sitting_recording(sitting_recording, acconeer_reference):
    with h5py.File(acconeer_reference, "r") as f:
        reference = f["breathing_rate"][:]
    expected = float(np.mean(reference[np.isfinite(reference)]))

    ours = _settled_rates(_run_recording(sitting_recording))

    assert np.median(ours) == pytest.approx(expected, abs=1.5)


def test_reaches_rate_estimation_on_the_sitting_recording(sitting_recording):
    states = [r.app_state for r in _run_recording(sitting_recording)]

    assert states[0] != AppState.ESTIMATE_BREATHING_RATE
    assert states[-1] == AppState.ESTIMATE_BREATHING_RATE


def test_analyses_several_distances_around_the_person(sitting_recording):
    final = _run_recording(sitting_recording)[-1]

    assert final.distances_being_analyzed is not None
    low, high = final.distances_being_analyzed
    assert high - low == 2  # three range points


@pytest.mark.parametrize("bpm", [10.0, 14.0, 22.0])
def test_recovers_a_known_simulated_breathing_rate(bpm):
    config = RadarConfig()
    pipeline = BreathingPipeline(config)
    frames = itertools.islice(
        simulated_frames(config, breaths_per_min=bpm, realtime=False), 900
    )
    results = [pipeline.process(f) for f in frames]

    assert np.median(_settled_rates(results)) == pytest.approx(bpm, abs=1.0)


def test_reports_no_rate_in_an_empty_room():
    config = RadarConfig()
    pipeline = BreathingPipeline(config)
    frames = itertools.islice(
        simulated_frames(config, breaths_per_min=None, realtime=False), 900
    )
    results = [pipeline.process(f) for f in frames]

    assert all(r.rate_bpm is None for r in results)


def _run_simulated(seconds, **kwargs):
    config = RadarConfig()
    pipeline = BreathingPipeline(config)
    frames = itertools.islice(
        simulated_frames(config, breaths_per_min=14.0, realtime=False, **kwargs),
        int(seconds * config.frame_rate),
    )
    return [pipeline.process(f) for f in frames]


def test_breath_hold_raises_apnea_instead_of_losing_the_person():
    results = _run_simulated(90, holds=((40.0, 25.0),))
    states = {round(r.t, 2): r.app_state for r in results}

    assert AppState.APNEA in states.values()
    # Never mistaken for an empty bed while the person holds their breath.
    assert all(states[round(t, 2)] != AppState.NO_PRESENCE for t in np.arange(40, 65, 0.5))
    # Alarm clears once breathing resumes, and tracking was not reset.
    assert results[-1].app_state == AppState.ESTIMATE_BREATHING_RATE
    assert results[-1].rate_bpm == pytest.approx(14.0, abs=1.0)


def test_short_pause_does_not_alarm():
    results = _run_simulated(70, holds=((40.0, 5.0),))

    assert all(r.app_state != AppState.APNEA for r in results)


def test_rate_is_not_dragged_down_by_a_breath_hold():
    results = _run_simulated(120, holds=((40.0, 25.0),))
    rates = [r.rate_bpm for r in results if r.t > 30 and r.rate_bpm is not None]

    assert min(rates) == pytest.approx(14.0, abs=1.0)
