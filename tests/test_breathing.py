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
