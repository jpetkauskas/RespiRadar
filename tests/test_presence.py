import itertools

import numpy as np

from respiradar.presence import PresenceDetector
from respiradar.sources import RadarConfig, recorded_config, replay_frames, simulated_frames

SETTLE = 100  # frames for the exponential filters to reach steady state


def _run_recording(path):
    detector = PresenceDetector(recorded_config(path))
    return [detector.process(frame) for frame in replay_frames(path)][SETTLE:]


def test_detects_the_person_in_the_sitting_recording(sitting_recording):
    assert all(r.detected for r in _run_recording(sitting_recording))


def test_locates_the_person_where_acconeer_found_them(sitting_recording):
    peaks = np.array([r.peak_distance_m for r in _run_recording(sitting_recording)])

    assert 0.55 <= np.median(peaks) <= 0.85


def test_detects_the_person_without_relying_on_the_recorded_presence_config(
    no_presence_processor_recording,
):
    assert all(r.detected for r in _run_recording(no_presence_processor_recording))


def test_reports_no_presence_in_an_empty_room():
    config = RadarConfig()
    detector = PresenceDetector(config)
    frames = itertools.islice(
        simulated_frames(config, breaths_per_min=None, realtime=False), 400
    )
    results = [detector.process(f) for f in frames][SETTLE:]

    assert np.mean([r.detected for r in results]) < 0.05
